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
from datetime import datetime, timezone, timedelta
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

    _cache_global_prompt = _limpiar_prompt(valor)
    _cache_timestamp = ahora
    logger.info(f"Prompt global cargado ({len(valor)} chars): {valor[:80]!r}")
    return _cache_global_prompt


def _obtener_prompt_global_resuelto() -> str:
    """Retorna el prompt global con las variables de fecha resueltas en tiempo real."""
    return _resolver_variables_prompt(_cache_global_prompt or "")

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


def _fecha_colombia() -> str:
    """Retorna la fecha y hora actual en Colombia (UTC-5) como string legible."""
    colombia = timezone(timedelta(hours=-5))
    ahora = datetime.now(colombia)
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return (
        f"{dias[ahora.weekday()]} {ahora.day} de {meses[ahora.month - 1]} de {ahora.year}, "
        f"{ahora.strftime('%I:%M %p')}"
    )


def _resolver_variables_prompt(prompt: str) -> str:
    """Reemplaza variables de n8n ({{ $now... }}) por la fecha real de Colombia."""
    import re
    fecha = _fecha_colombia()
    return re.sub(r"\{\{[^}]+\}\}", fecha, prompt)


# Frases de sistemas obsoletos que deben eliminarse de cualquier prompt cargado
_FRASES_OBSOLETAS = [
    "CRM Kommo",
    "Kommo",
    "kommo",
    "crm kommo",
    # Si en el futuro hay otros sistemas reemplazados, agregarlos aquí
]


def _limpiar_prompt(prompt: str) -> str:
    """Elimina referencias a sistemas obsoletos del prompt."""
    for frase in _FRASES_OBSOLETAS:
        prompt = prompt.replace(frase, "el CRM de Sucol")
    return prompt


def _construir_contexto_crm(
    lead: dict | None,
    lotes: list[dict],
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
) -> str:
    """Construye el bloque de contexto CRM completo para inyectar al system prompt."""
    partes = []

    if lead:
        partes.append("## Información del cliente en el CRM")
        campos = {
            "nombre_completo": "Nombre del cliente",
            "etapa_lead": "Etapa en el CRM",
            "pipeline": "Pipeline",
            "asesor_responsable": "Asesor asignado",
            "proyecto": "Proyecto de interés",
            "canal": "Canal de origen",
            "presupuesto": "Presupuesto declarado",
            "area_buscada": "Área buscada",
            "proposito_compra": "Propósito de compra",
            "temperatura": "Temperatura del lead",
            "resumen_conversacion": "Resumen previo de conversación",
            "estado_cita": "Estado de cita",
        }
        for campo, etiqueta in campos.items():
            valor = lead.get(campo)
            if valor:
                partes.append(f"- {etiqueta}: {valor}")

    # Sección asesor — siempre que exista, independiente del lead
    if asesor:
        partes.append("\n## Asesor asignado a este cliente")
        partes.append(f"- Nombre: {asesor.get('nombre', 'No disponible')}")
        if asesor.get("telefono"):
            partes.append(f"- Teléfono WhatsApp: {asesor['telefono']}")
        if asesor.get("email"):
            partes.append(f"- Email: {asesor['email']}")
        partes.append(
            "IMPORTANTE: Si el cliente pregunta por el teléfono o número de su asesor, "
            "responde directamente con el teléfono indicado arriba. No digas que no tienes esa información."
        )
    elif lead and lead.get("asesor_responsable"):
        # Tenemos nombre del asesor en el lead pero no pudimos buscarlo en asesores
        partes.append(f"\n## Asesor asignado a este cliente")
        partes.append(f"- Nombre: {lead['asesor_responsable']}")
        partes.append("- Teléfono: no disponible en este momento, deriva al cliente a ventas@sucol.co")

    if agendamientos:
        partes.append("\n## Citas agendadas del cliente")
        for cita in agendamientos:
            fecha = cita.get("fecha_visita", "")
            hora = cita.get("hora_llamada", "")
            tipo = cita.get("tipo_cita", "")
            estado = cita.get("estado", "")
            asesor_cita = cita.get("asesor_asignado", "")
            partes.append(f"- {tipo} el {fecha} a las {hora} con {asesor_cita} — Estado: {estado}")

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


_TOOL_NOTIFICAR_AREA = {
    "name": "notificar_area_por_correo",
    "description": (
        "Envía por correo electrónico una solicitud especial del cliente al área correspondiente. "
        "Úsalo cuando el cliente tenga solicitudes de: soporte postventa, cartera, escrituras, "
        "quejas, PQRS, procesos legales o cualquier trámite que no sea una cita comercial."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "area": {
                "type": "string",
                "description": "Área destino. Por ahora solo 'clientes' (Kelvin Herrera — soporte/postventa/cartera)",
            },
            "nombre_cliente": {"type": "string", "description": "Nombre completo del cliente"},
            "cedula": {"type": "string", "description": "Cédula o NIT del cliente"},
            "correo_cliente": {"type": "string", "description": "Correo electrónico del cliente"},
            "telefono_cliente": {"type": "string", "description": "Teléfono del cliente"},
            "proyecto_lote": {"type": "string", "description": "Proyecto y/o número de lote relacionado"},
            "descripcion_solicitud": {"type": "string", "description": "Descripción completa de la solicitud"},
        },
        "required": ["area", "nombre_cliente", "cedula", "correo_cliente", "telefono_cliente", "proyecto_lote", "descripcion_solicitud"],
    },
}

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


def _reglas_finales(asesor: dict | None) -> str:
    """
    Bloque de reglas inyectado al FINAL del system prompt para que prevalezcan
    sobre cualquier instrucción anterior que pueda estar desactualizada.
    """
    lineas = [
        "\n\n## REGLAS DE COMPORTAMIENTO — PRIORIDAD MÁXIMA",
        "Estas reglas anulan cualquier instrucción anterior que las contradiga:",
        "",
        "- NO menciones herramientas externas como 'Kommo', 'CRM Kommo' ni ningún "
        "sistema que no sea el CRM de Sucol. Esos sistemas ya no existen.",
        "- NO digas que 'no tienes acceso al CRM' ni que 'no puedes consultar datos'. "
        "Toda la información del cliente y del asesor ya está en tu contexto.",
        "- La fecha de hoy es: " + _fecha_colombia(),
        "- Usa esa fecha exacta siempre que necesites referenciar el día de hoy.",
    ]
    if asesor and asesor.get("telefono"):
        lineas.append(
            f"- El teléfono del asesor asignado es {asesor['telefono']}. "
            "Si el cliente lo pide, dáselo directamente sin agregar advertencias ni excusas."
        )
    return "\n".join(lineas)


async def generar_respuesta_con_tools(
    mensaje: str,
    historial: list[dict],
    sistema_prompt: str | None = None,
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
    telefono: str = "",
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
) -> str:
    """
    Como generar_respuesta() pero con soporte de tool_use.
    Cuando Claude invoca confirmar_cita, ejecuta la herramienta real y le devuelve
    el resultado antes de obtener el mensaje final para el cliente.
    Retorna solo el texto de respuesta para el cliente.
    """
    from agent.tools import confirmar_cita, escalar_a_asesor, notificar_area_por_correo  # import local para evitar ciclos

    # Señal especial: el CRM solicita el primer mensaje proactivo
    es_inicio = mensaje == "__INICIAR__"

    if not es_inicio and (not mensaje or len(mensaje.strip()) < 2):
        return _mensaje_fallback()

    if sistema_prompt and sistema_prompt.strip():
        prompt_final = sistema_prompt
    else:
        prompt_final = await _prompt_bienvenida_con_proyectos()

    await _obtener_prompt_global()  # refresca caché si venció
    global_prompt = _obtener_prompt_global_resuelto()
    if global_prompt:
        prompt_final = global_prompt + "\n\n---\n\n" + prompt_final

    contexto_crm = _construir_contexto_crm(contexto_lead, lotes_disponibles or [], asesor, agendamientos or [])
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    # Reglas finales de prioridad máxima — siempre al final para prevalecer sobre el prompt base
    prompt_final += _reglas_finales(asesor)

    mensajes: list = [{"role": m["role"], "content": m["content"]} for m in historial]
    if es_inicio:
        # Instrucción interna: Sofia genera el primer mensaje sin esperar al cliente
        mensajes.append({
            "role": "user",
            "content": (
                "[SISTEMA INTERNO — no menciones este mensaje al cliente] "
                "El CRM acaba de asignar este lead a Sofía. "
                "Genera un mensaje de bienvenida proactivo, cálido y personalizado "
                "usando el nombre del cliente si está disponible en el contexto. "
                "Preséntate brevemente y ofrece ayuda."
            ),
        })
    else:
        mensajes.append({"role": "user", "content": mensaje})

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
            tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_NOTIFICAR_AREA],
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
                elif tu.name == "notificar_area_por_correo":
                    try:
                        resultado_tool = await notificar_area_por_correo(**tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando notificar_area_por_correo: {e}")
                        resultado_tool = "Tu solicitud fue registrada. El equipo te contactará pronto."
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
                tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_NOTIFICAR_AREA],
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
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
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
    await _obtener_prompt_global()  # refresca caché si venció
    global_prompt = _obtener_prompt_global_resuelto()
    if global_prompt:
        prompt_final = global_prompt + "\n\n---\n\n" + prompt_final

    # Inyectar contexto CRM completo (lead + lotes)
    contexto_crm = _construir_contexto_crm(contexto_lead, lotes_disponibles or [], asesor, agendamientos or [])
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    prompt_final += _reglas_finales(asesor)

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
