# agent/brain.py — Cerebro de Sofía: conexión con Gemini API
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Lógica de IA de Sofía. Soporta prompts dinámicos por proyecto desde el CRM
y un prompt genérico de bienvenida cuando el cliente aún no tiene proyecto asignado.
"""

import os
import yaml
import logging
import httpx
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")


class _GeminiCompatMessages:
    async def create(self, model: str, max_tokens: int, system: str, messages: list, tools: list | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no configurada")

        contents = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                contents.append({
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                })
                continue
            if role == "assistant" and isinstance(content, list):
                parts = []
                for block in content:
                    if getattr(block, "type", None) == "text":
                        parts.append({"text": getattr(block, "text", "")})
                    elif getattr(block, "type", None) == "tool_use":
                        parts.append({
                            "functionCall": {
                                "name": block.name,
                                "args": block.input,
                            }
                        })
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
                continue
            if role == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        contents.append({
                            "role": "user",
                            "parts": [{
                                "functionResponse": {
                                    "name": block["tool_use_id"].split(":", 1)[0],
                                    "response": {"result": block["content"]},
                                }
                            }],
                        })

        gemini_tools = None
        if tools:
            gemini_tools = [{
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    }
                    for tool in tools
                ]
            }]

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if gemini_tools:
            payload["tools"] = gemini_tools

        url = f"{_GEMINI_BASE_URL}/models/{_GEMINI_MODEL}:generateContent"
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(url, params={"key": api_key}, json=payload)
            response.raise_for_status()
            data = response.json()

        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        usage_data = data.get("usageMetadata", {})
        usage = SimpleNamespace(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0),
        )

        function_calls = [part["functionCall"] for part in parts if "functionCall" in part]
        if function_calls:
            content = []
            for part in parts:
                if "text" in part:
                    content.append(SimpleNamespace(type="text", text=part["text"]))
            for idx, function_call in enumerate(function_calls):
                name = function_call.get("name", "")
                content.append(SimpleNamespace(
                    type="tool_use",
                    id=f"{name}:{idx}",
                    name=name,
                    input=function_call.get("args", {}),
                ))
            return SimpleNamespace(stop_reason="tool_use", content=content, usage=usage)

        text = "".join(part.get("text", "") for part in parts)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
            usage=usage,
        )


class _GeminiCompatClient:
    def __init__(self):
        self.messages = _GeminiCompatMessages()


client = _GeminiCompatClient()

_KNOWLEDGE_DIR = "knowledge"


def _cargar_knowledge(proyecto_slug: str | None = None) -> str:
    """
    Carga el conocimiento comercial en dos capas:
    1. knowledge/global.md            — siempre (empresa, directorio, directrices)
    2. knowledge/proyectos/[slug].md  — solo cuando se conoce el proyecto del lead

    El slug se normaliza a minúsculas con guion_bajo para buscar el archivo.
    Si no se encuentra el archivo del proyecto, solo se usa global.md.
    """
    partes = []

    # Capa 1: conocimiento global (siempre)
    ruta_global = os.path.join(_KNOWLEDGE_DIR, "global.md")
    if os.path.isfile(ruta_global):
        try:
            with open(ruta_global, "r", encoding="utf-8") as f:
                contenido = f.read().strip()
                if contenido:
                    partes.append(contenido)
        except (OSError, IOError) as e:
            logger.warning(f"No se pudo leer global.md: {e}")
    else:
        logger.warning("knowledge/global.md no encontrado — prompts sin conocimiento global")

    # Capa 2: ficha del proyecto específico (solo si se conoce)
    if proyecto_slug:
        normalizado = proyecto_slug.lower().replace("-", "_").replace(" ", "_")
        ruta_proyecto = os.path.join(_KNOWLEDGE_DIR, "proyectos", f"{normalizado}.md")
        if os.path.isfile(ruta_proyecto):
            try:
                with open(ruta_proyecto, "r", encoding="utf-8") as f:
                    contenido = f.read().strip()
                    if contenido:
                        partes.append(contenido)
                logger.debug(f"Knowledge: global + {normalizado}.md ({len(partes[1]) if len(partes) > 1 else 0} chars proyecto)")
            except (OSError, IOError) as e:
                logger.warning(f"No se pudo leer {ruta_proyecto}: {e}")
        else:
            logger.warning(f"Sin ficha para proyecto '{proyecto_slug}' — usando solo global.md")
    else:
        logger.debug("Knowledge: solo global.md (proyecto no detectado aún)")

    return "\n\n---\n\n".join(partes)


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
        "Eres Sofía, Asesora Digital de SUCOL Soluciones Urbanísticas. Responde en español.",
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


def _construir_prompt_base(proyecto_slug: str | None = None) -> str:
    """
    Construye el system prompt base combinando:
    1. config/prompts.yaml           — identidad y reglas de comportamiento de Sofía
    2. knowledge/global.md           — empresa, directorio, directrices maestras
    3. knowledge/proyectos/[slug].md — ficha del proyecto (si se conoce)
    """
    persona = _prompt_base_yaml()
    knowledge = _cargar_knowledge(proyecto_slug)
    if knowledge:
        return persona + "\n\n---\n\n" + knowledge
    return persona


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
        partes.append("\n## Asesor asignado a este cliente (uso interno)")
        partes.append(f"- Nombre: {asesor.get('nombre', 'No disponible')}")
        if asesor.get("telefono"):
            partes.append(f"- Teléfono WhatsApp: {asesor['telefono']}")
        if asesor.get("email"):
            partes.append(f"- Email: {asesor['email']}")
        partes.append(
            "INSTRUCCIÓN: NO menciones espontáneamente al asesor ni su teléfono en tu respuesta. "
            "Solo proporciona estos datos si el cliente los pide explícitamente. "
            "Tu función es resolver las preguntas del cliente y agendar citas — no derivarlo al asesor."
        )
    elif lead and lead.get("asesor_responsable"):
        partes.append(f"\n## Asesor asignado a este cliente (uso interno)")
        partes.append(f"- Nombre: {lead['asesor_responsable']}")
        partes.append(
            "INSTRUCCIÓN: NO menciones al asesor ni lo ofrezcas proactivamente. "
            "Solo si el cliente pide contactar a una persona humana, indícale que su asesor se pondrá en contacto."
        )

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
        partes.append("\n## Lotes disponibles en este proyecto (datos exactos del CRM)")
        partes.append("Usa esta información para precios y áreas específicas. Si hay contradicción con la ficha del proyecto, prevalece esta tabla:")
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
    # Si el CRM no tiene lotes en tabla, Sofía usa la ficha del proyecto en knowledge/ como fuente de verdad.
    # NO se agrega ningún mensaje de "no hay unidades" — ese dato viene de la ficha.

    return "\n".join(partes)



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
        "Notifica al asesor humano para que contacte al cliente. "
        "ÚSALO SOLO en estos casos específicos: "
        "(1) El cliente pide hablar con una persona humana de forma explícita. "
        "(2) El cliente quiere negociar precio o un descuento mayor al 3%. "
        "(3) El cliente tiene una queja, problema legal, tema de cartera o escrituras. "
        "NO usar para preguntas sobre proyectos, disponibilidad, precios o para agendar citas — "
        "esas las gestiona Sofía directamente con confirmar_cita."
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

_TOOL_CONSULTAR_ASESOR = {
    "name": "consultar_asesor_por_nombre",
    "description": (
        "Consulta en el CRM el teléfono y email de un asesor activo por nombre parcial. "
        "Úsalo cuando el cliente pregunte por el número, WhatsApp, teléfono, correo o contacto "
        "de un asesor específico, por ejemplo 'Fabio', 'Luz Aide' o 'Kelvin'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "nombre_asesor": {
                "type": "string",
                "description": "Nombre o parte del nombre del asesor que el cliente está consultando",
            },
        },
        "required": ["nombre_asesor"],
    },
}

_TOOL_CALIFICAR_SIN_VISITA = {
    "name": "calificar_lead_sin_visita",
    "description": (
        "Registra al lead como calificado cuando ha mostrado interés claro pero "
        "no quiere agendar una cita todavía. Un asesor de Sucol lo contactará directamente. "
        "Usar cuando el lead perfiló sus necesidades, presupuesto o proyecto de interés "
        "pero prefiere que lo llamen, necesita tiempo para decidir, o no está listo para agendar."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "nombre_cliente": {
                "type": "string",
                "description": "Nombre del cliente si se conoce, si no escribir 'Cliente'",
            },
            "resumen": {
                "type": "string",
                "description": (
                    "Resumen de la conversación: proyecto de interés, presupuesto, "
                    "propósito de compra, dudas principales y por qué no agendó visita"
                ),
            },
        },
        "required": ["nombre_cliente", "resumen"],
    },
}

_TOOL_CONFIRMAR_CITA = {
    "name": "confirmar_cita",
    "description": (
        "Agenda una cita en el CRM y notifica al asesor por WhatsApp cuando el cliente "
        "confirma fecha, hora y tipo de cita. La respuesta de esta herramienta contiene "
        "los datos completos de la cita y del asesor; al responder al cliente, conserva "
        "esos datos sin omitir el teléfono del asesor."
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
        "- LONGITUD: máximo 3 oraciones por mensaje. Esto es WhatsApp, no un email. "
        "Si tu respuesta tiene más de 4 líneas, córtala.",
        "- NO escales al asesor humano solo porque el cliente hizo una pregunta informativa. "
        "Respóndela tú directamente con la información de tu ficha.",
        "- NO menciones herramientas externas como 'Kommo', 'CRM Kommo' ni ningún "
        "sistema que no sea el CRM de Sucol. Esos sistemas ya no existen.",
        "- NO digas que 'no tienes acceso al CRM' ni que 'no puedes consultar datos'. "
        "Toda la información del cliente y del asesor ya está en tu contexto.",
        "- Si el cliente pide el telefono, WhatsApp, correo o contacto de un asesor por nombre "
        "especifico, usa la herramienta consultar_asesor_por_nombre antes de responder.",
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
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
    telefono: str = "",
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
    proyecto_slug: str | None = None,
) -> str:
    """
    Genera respuesta con soporte de tool_use.
    El system prompt se construye desde config/prompts.yaml + knowledge/global.md
    + knowledge/proyectos/[proyecto_slug].md (si se conoce el proyecto).
    Retorna solo el texto de respuesta para el cliente.
    """
    from agent.tools import (
        confirmar_cita,
        escalar_a_asesor,
        notificar_area_por_correo,
        calificar_lead_sin_visita,
        consultar_asesor_por_nombre,
    )  # import local para evitar ciclos

    # Señal especial: el CRM solicita el primer mensaje proactivo
    es_inicio = mensaje == "__INICIAR__"

    if not es_inicio and (not mensaje or len(mensaje.strip()) < 2):
        return _mensaje_fallback()

    prompt_final = _construir_prompt_base(proyecto_slug)

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
            model=_GEMINI_MODEL,
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
            tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_CONSULTAR_ASESOR, _TOOL_NOTIFICAR_AREA, _TOOL_CALIFICAR_SIN_VISITA],
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
                elif tu.name == "consultar_asesor_por_nombre":
                    try:
                        resultado_tool = await consultar_asesor_por_nombre(**tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando consultar_asesor_por_nombre: {e}")
                        resultado_tool = "No pude consultar ese asesor en este momento."
                elif tu.name == "notificar_area_por_correo":
                    try:
                        resultado_tool = await notificar_area_por_correo(**tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando notificar_area_por_correo: {e}")
                        resultado_tool = "Tu solicitud fue registrada. El equipo te contactará pronto."
                elif tu.name == "calificar_lead_sin_visita":
                    try:
                        resultado_tool = await calificar_lead_sin_visita(telefono=telefono, **tu.input)
                    except Exception as e:
                        logger.error(f"Error ejecutando calificar_lead_sin_visita: {e}")
                        resultado_tool = "Tus datos quedaron registrados. Un asesor te contactará pronto."
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
                model=_GEMINI_MODEL,
                max_tokens=1024,
                system=prompt_final,
                messages=mensajes_con_resultado,
                tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_CONSULTAR_ASESOR, _TOOL_NOTIFICAR_AREA, _TOOL_CALIFICAR_SIN_VISITA],
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
        logger.error(f"Error Gemini API (con tools): {e}")
        return _mensaje_error()


async def generar_respuesta(
    mensaje: str,
    historial: list[dict],
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
    proyecto_slug: str | None = None,
) -> str:
    """
    Genera una respuesta usando Gemini API.
    El system prompt se construye desde config/prompts.yaml + knowledge/global.md
    + knowledge/proyectos/[proyecto_slug].md (si se conoce el proyecto).
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    prompt_final = _construir_prompt_base(proyecto_slug)

    contexto_crm = _construir_contexto_crm(contexto_lead, lotes_disponibles or [], asesor, agendamientos or [])
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    prompt_final += _reglas_finales(asesor)

    mensajes = [{"role": m["role"], "content": m["content"]} for m in historial]
    mensajes.append({"role": "user", "content": mensaje})

    try:
        response = await client.messages.create(
            model=_GEMINI_MODEL,
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
        )
        respuesta = response.content[0].text
        logger.info(f"Respuesta generada ({response.usage.input_tokens} in / {response.usage.output_tokens} out)")
        return respuesta

    except Exception as e:
        logger.error(f"Error Gemini API: {e}")
        return _mensaje_error()
