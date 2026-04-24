# agent/brain.py — Cerebro de Sofía: conexión con Claude API
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Lógica de IA de Sofía. Soporta prompts dinámicos por proyecto desde el CRM
y un prompt genérico de bienvenida cuando el cliente aún no tiene proyecto asignado.
"""

import os
import time
import yaml
import logging
import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Caché del prompt global ────────────────────────────────────────────────────
_cache_global_prompt: str | None = None
_cache_timestamp: float = 0.0
_CACHE_TTL = 300  # 5 minutos


async def _obtener_prompt_global() -> str:
    """
    Lee el prompt global configurado por el admin desde sofia_config en Supabase.
    Usa caché de 5 minutos para no consultar en cada mensaje.
    Retorna "" ante cualquier error o si el valor está vacío.
    """
    global _cache_global_prompt, _cache_timestamp

    ahora = time.monotonic()
    if _cache_global_prompt is not None and (ahora - _cache_timestamp) < _CACHE_TTL:
        return _cache_global_prompt

    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

    if not supabase_url or not supabase_key:
        logger.warning("SUPABASE_URL / SUPABASE_SERVICE_KEY no configurados — prompt global omitido")
        _cache_global_prompt = ""
        _cache_timestamp = ahora
        return ""

    try:
        url = f"{supabase_url}/rest/v1/sofia_config?select=global_prompt&limit=1"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }
        async with httpx.AsyncClient(timeout=5) as http:
            r = await http.get(url, headers=headers)
            r.raise_for_status()
            rows = r.json()
            logger.info(f"sofia_config respuesta cruda: {rows}")
            valor = rows[0]["global_prompt"].strip() if rows and rows[0].get("global_prompt") else ""
    except Exception as e:
        logger.warning(f"No se pudo leer prompt global de Supabase: {e}")
        valor = ""

    _cache_global_prompt = valor
    _cache_timestamp = ahora
    logger.info(f"Prompt global cargado ({len(valor)} chars): {valor[:80]!r}")
    return valor

# Prompt base de Sofía — se usa cuando el CRM no tiene proyecto para este cliente
_PROMPT_BIENVENIDA = """Eres Sofía, la asesora virtual de Sucol Soluciones Urbanísticas.

## Tu rol
Atiendes a personas interesadas en adquirir lotes o proyectos urbanísticos de Sucol.
Tu objetivo es entender en qué proyecto está interesado el cliente y conectarlo con la
información correcta.

## Proyectos disponibles
{lista_proyectos}

## Cómo actuar
- Saluda de forma cálida y profesional
- Pregunta por cuál de los proyectos le interesa obtener información
- Una vez que identifiques el proyecto, el sistema te dará información detallada
- Si el cliente no está seguro, descríbele brevemente cada proyecto y ayúdalo a elegir
- NUNCA inventes precios ni datos que no tengas — di que lo conectarás con un asesor

## Reglas
- Responde siempre en español
- Sé empática, clara y profesional
- Mantén las respuestas cortas y útiles
- Termina siempre con una pregunta o invitación a continuar"""


def _cargar_config_prompts() -> dict:
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _prompt_base_yaml() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "system_prompt",
        "Eres Sofía, asesora virtual de Sucol Soluciones Urbanísticas. Responde en español.",
    )


def _mensaje_error() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "error_message",
        "Lo siento, estoy teniendo un pequeño inconveniente técnico. Por favor intenta de nuevo en unos minutos.",
    )


def _mensaje_fallback() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "fallback_message",
        "Disculpa, no entendí bien tu mensaje. ¿Puedes contarme en qué te puedo ayudar?",
    )


def _construir_contexto_crm(lead: dict | None, lotes: list[dict]) -> str:
    """Construye el bloque de contexto CRM completo para inyectar al system prompt."""
    partes = []

    if lead:
        partes.append("## Información del cliente en el CRM")
        campos = {
            "nombre_completo": "Nombre",
            "etapa_lead": "Etapa en el CRM",
            "asesor_responsable": "Asesor asignado",
            "proyecto": "Proyecto de interés",
            "fuente": "Fuente del lead",
            "presupuesto": "Presupuesto declarado",
            "notas": "Notas previas",
        }
        for campo, etiqueta in campos.items():
            valor = lead.get(campo)
            if valor:
                partes.append(f"- {etiqueta}: {valor}")
        if lead.get("asesor_responsable"):
            partes.append(
                f"\nSi el cliente necesita hablar con alguien, "
                f"su asesor asignado es {lead['asesor_responsable']}."
            )

    if lotes:
        partes.append("\n## Lotes disponibles en este proyecto")
        partes.append("Usa esta información cuando pregunten por precios, áreas o formas de pago:")
        for lote in lotes:
            linea = f"- Lote {lote.get('codigo', 'S/N')}"
            if lote.get("area_m2"):
                linea += f" | {lote['area_m2']} m²"
            if lote.get("precio_total"):
                linea += f" | Precio: ${lote['precio_total']:,.0f}"
            if lote.get("separacion_inicial"):
                linea += f" | Separación: ${lote['separacion_inicial']:,.0f}"
            if lote.get("cuotas_cantidad") and lote.get("cuota_valor"):
                linea += f" | {lote['cuotas_cantidad']} cuotas de ${lote['cuota_valor']:,.0f}"
            partes.append(linea)
    elif lead and lead.get("proyecto"):
        partes.append("\n## Disponibilidad de lotes")
        partes.append(
            "No hay lotes disponibles en este momento. "
            "Ofrece al cliente hablar con el asesor para revisar opciones."
        )

    return "\n".join(partes)


async def _prompt_bienvenida_con_proyectos() -> str:
    """Genera el prompt genérico listando los proyectos activos del CRM."""
    try:
        from agent.crm import obtener_proyectos_activos
        proyectos = await obtener_proyectos_activos()
        if proyectos:
            lista = "\n".join(f"- {p['nombre']}" for p in proyectos)
        else:
            lista = "- Proyectos urbanísticos Sucol (consulta disponibilidad)"
    except Exception:
        lista = "- Proyectos urbanísticos Sucol (consulta disponibilidad)"
    return _PROMPT_BIENVENIDA.format(lista_proyectos=lista)


_TOOL_ESCALAR_ASESOR = {
    "name": "escalar_a_asesor",
    "description": (
        "Transfiere al cliente con su asesor asignado de forma inmediata cuando el cliente "
        "quiere hablar con una persona ahora, tiene una consulta urgente, o prefiere no agendar "
        "una cita y simplemente ser contactado por un asesor."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "nombre_cliente": {
                "type": "string",
                "description": "Nombre del cliente si se conoce, si no escribir 'Cliente'",
            },
            "motivo": {
                "type": "string",
                "description": "Resumen breve de por qué el cliente quiere hablar con un asesor",
            },
        },
        "required": ["nombre_cliente", "motivo"],
    },
}

_TOOL_CONFIRMAR_CITA = {
    "name": "confirmar_cita",
    "description": (
        "Agenda una cita en el CRM y notifica al asesor por WhatsApp cuando el cliente "
        "confirma fecha, hora y tipo de cita."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "nombre_cliente": {
                "type": "string",
                "description": "Nombre completo del cliente",
            },
            "tipo_cita": {
                "type": "string",
                "description": "Tipo de cita: Cita Virtual, Visita Presencial, Llamada",
            },
            "fecha_cita": {
                "type": "string",
                "description": "Fecha en formato YYYY-MM-DD",
            },
            "hora_cita": {
                "type": "string",
                "description": "Hora en formato HH:MM",
            },
            "resumen": {
                "type": "string",
                "description": "Resumen breve de lo que conversó el cliente con Sofia",
            },
            "video_url": {
                "type": "string",
                "description": "Enlace de videollamada, dejar vacío si no aplica",
            },
        },
        "required": ["nombre_cliente", "tipo_cita", "fecha_cita", "hora_cita", "resumen"],
    },
}


async def generar_respuesta_con_tools(
    mensaje: str,
    historial: list[dict],
    sistema_prompt: str | None = None,
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
    telefono: str = "",
) -> str:
    """
    Como generar_respuesta() pero con soporte de tool_use.
    Cuando Claude invoca confirmar_cita, ejecuta la herramienta real y le devuelve
    el resultado antes de obtener el mensaje final para el cliente.
    Retorna solo el texto de respuesta para el cliente.
    """
    from agent.tools import confirmar_cita, escalar_a_asesor  # import local para evitar ciclos

    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    if sistema_prompt and sistema_prompt.strip():
        prompt_final = sistema_prompt
    else:
        prompt_final = await _prompt_bienvenida_con_proyectos()

    global_prompt = await _obtener_prompt_global()
    if global_prompt:
        prompt_final = global_prompt + "\n\n---\n\n" + prompt_final

    contexto_crm = _construir_contexto_crm(contexto_lead, lotes_disponibles or [])
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    mensajes: list = [{"role": m["role"], "content": m["content"]} for m in historial]
    mensajes.append({"role": "user", "content": mensaje})

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
            tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR],
        )

        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            tool_results = []

            for tu in tool_uses:
                if tu.name == "confirmar_cita":
                    try:
                        resultado_tool = await confirmar_cita(telefono=telefono, **tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando confirmar_cita: {e}")
                        resultado_tool = "Hubo un problema al agendar la cita. Por favor intenta de nuevo."
                elif tu.name == "escalar_a_asesor":
                    try:
                        resultado_tool = await escalar_a_asesor(telefono=telefono, **tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando escalar_a_asesor: {e}")
                        resultado_tool = "Hubo un problema al contactar al asesor. Por favor intenta de nuevo."
                else:
                    resultado_tool = f"Herramienta {tu.name} no reconocida."

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": resultado_tool,
                })

            mensajes_con_resultado = mensajes + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

            response2 = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=prompt_final,
                messages=mensajes_con_resultado,
                tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR],
            )
            respuesta = response2.content[0].text
            logger.info(
                f"Respuesta con tool_use "
                f"({response.usage.input_tokens}+{response2.usage.input_tokens} in / "
                f"{response2.usage.output_tokens} out)"
            )
        else:
            respuesta = response.content[0].text
            logger.info(f"Respuesta generada ({response.usage.input_tokens} in / {response.usage.output_tokens} out)")

        return respuesta

    except Exception as e:
        logger.error(f"Error Claude API (con tools): {e}")
        return _mensaje_error()


async def generar_respuesta(
    mensaje: str,
    historial: list[dict],
    sistema_prompt: str | None = None,
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
) -> str:
    """
    Genera una respuesta usando Claude API (claude-sonnet-4-6).

    Args:
        mensaje: El mensaje nuevo del cliente
        historial: Mensajes anteriores [{"role": "...", "content": "..."}]
        sistema_prompt: System prompt del proyecto desde el CRM. Si es None, usa bienvenida genérica.
        contexto_lead: Datos del lead para personalizar la respuesta.
        lotes_disponibles: Lotes del proyecto para responder preguntas de precios/áreas.
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    if sistema_prompt and sistema_prompt.strip():
        prompt_final = sistema_prompt
    else:
        prompt_final = await _prompt_bienvenida_con_proyectos()

    # Inyectar prompt global del admin (si existe) al inicio
    global_prompt = await _obtener_prompt_global()
    if global_prompt:
        prompt_final = global_prompt + "\n\n---\n\n" + prompt_final

    # Inyectar contexto CRM completo (lead + lotes)
    contexto_crm = _construir_contexto_crm(contexto_lead, lotes_disponibles or [])
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    mensajes = [{"role": m["role"], "content": m["content"]} for m in historial]
    mensajes.append({"role": "user", "content": mensaje})

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
        )
        respuesta = response.content[0].text
        logger.info(f"Respuesta generada ({response.usage.input_tokens} in / {response.usage.output_tokens} out)")
        return respuesta

    except Exception as e:
        logger.error(f"Error Claude API: {e}")
        return _mensaje_error()
