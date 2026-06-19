# agent/brain.py — Cerebro de Sofía: conexión con Gemini API
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Lógica de IA de Sofía. Soporta prompts dinámicos por proyecto desde el CRM
y un prompt genérico de bienvenida cuando el cliente aún no tiene proyecto asignado.
"""

import os
import re
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


def _cargar_knowledge(
    proyecto_slug: str | None = None,
    proyecto: dict | None = None,
) -> str:
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
        elif proyecto and proyecto.get("system_prompt"):
            partes.append(str(proyecto["system_prompt"]).strip())
            logger.warning(
                f"Sin ficha local para '{proyecto_slug}' — usando system_prompt del CRM como respaldo"
            )
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


def _construir_prompt_base(
    proyecto_slug: str | None = None,
    proyecto: dict | None = None,
) -> str:
    """
    Construye el system prompt base combinando:
    1. config/prompts.yaml           — identidad y reglas de comportamiento de Sofía
    2. knowledge/global.md           — empresa, directorio, directrices maestras
    3. knowledge/proyectos/[slug].md — ficha del proyecto (si se conoce)
    """
    persona = _prompt_base_yaml()
    knowledge = _cargar_knowledge(proyecto_slug, proyecto)
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
    proyecto: dict | None = None,
) -> str:
    """Construye el bloque de contexto CRM completo para inyectar al system prompt."""
    partes = []

    if proyecto:
        partes.append("## DATOS OFICIALES ACTUALES DEL PROYECTO — PRIORIDAD MÁXIMA")
        partes.append(f"- Proyecto: {proyecto.get('nombre') or proyecto.get('slug')}")
        if proyecto.get("ubicacion"):
            partes.append(f"- Ubicación oficial: {proyecto['ubicacion']}")
        if proyecto.get("direccion_visita"):
            partes.append(f"- Dirección oficial para visitas: {proyecto['direccion_visita']}")
        else:
            partes.append("- Dirección oficial para visitas: NO REGISTRADA")
        if proyecto.get("google_maps_url"):
            partes.append(f"- Google Maps oficial: {proyecto['google_maps_url']}")
        if proyecto.get("indicaciones_visita"):
            partes.append(f"- Indicaciones oficiales: {proyecto['indicaciones_visita']}")
        partes.append(
            "REGLA: estos datos operativos del CRM prevalecen sobre la ficha, el historial "
            "y cualquier conocimiento general. Nunca completes ni deduzcas una dirección."
        )

    if lead:
        partes.append("## Información del cliente en el CRM")
        campos = {
            "nombre_completo": "Nombre del cliente",
            "etapa_lead": "Etapa en el CRM",
            "pipeline": "Pipeline",
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
        partes.append(
            "Esta información es dinámica y prevalece sobre cantidades, áreas, precios "
            "y disponibilidad escritos en cualquier ficha estática."
        )
        partes.append(f"- Total de registros disponibles: {len(lotes)}")

        resumen_areas: dict[str, dict] = {}
        for lote in lotes:
            area = lote.get("area_m2")
            clave = str(area) if area is not None else "Área no registrada"
            grupo = resumen_areas.setdefault(
                clave,
                {"cantidad": 0, "precios": []},
            )
            grupo["cantidad"] += 1
            if lote.get("precio_total") is not None:
                grupo["precios"].append(lote["precio_total"])

        def _orden_area(item):
            try:
                return (0, float(item[0]))
            except (TypeError, ValueError):
                return (1, item[0])

        grupos_ordenados = sorted(resumen_areas.items(), key=_orden_area)
        for area, grupo in grupos_ordenados[:30]:
            linea = f"- {area} m²: {grupo['cantidad']} disponible(s)"
            precios = grupo["precios"]
            if precios:
                minimo = min(precios)
                maximo = max(precios)
                if minimo == maximo:
                    linea += f" | Precio CRM: ${minimo:,.0f}"
                else:
                    linea += f" | Rango CRM: ${minimo:,.0f} a ${maximo:,.0f}"
            partes.append(linea)
        if len(grupos_ordenados) > 30:
            partes.append(
                f"- Hay {len(grupos_ordenados) - 30} áreas adicionales no incluidas por brevedad. "
                "No inventes sus datos."
            )
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
        "la confirmación de la cita. Al responder al cliente, confirma tipo, fecha y hora, "
        "pero NO menciones el nombre, teléfono ni correo del asesor salvo que el cliente "
        "haya pedido explícitamente esos datos."
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


def _reglas_finales(asesor: dict | None, proyecto: dict | None = None) -> str:
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
        "- PROACTIVIDAD: después de responder, invita de forma natural a agendar una visita, "
        "una llamada o una cita virtual según el proyecto. Sofía conduce el proceso.",
        "- ASESOR: no menciones que el cliente tiene asesor asignado, ni su nombre, teléfono "
        "o correo. No ofrezcas 'conectarlo con su asesor'. Solo puedes hacerlo si el cliente "
        "pide explícitamente hablar con una persona o solicita el contacto de un asesor.",
        "- NO menciones herramientas externas como 'Kommo', 'CRM Kommo' ni ningún "
        "sistema que no sea el CRM de Sucol. Esos sistemas ya no existen.",
        "- NO digas que 'no tienes acceso al CRM' ni que 'no puedes consultar datos'. "
        "Toda la información del cliente y del asesor ya está en tu contexto.",
        "- FUENTES: usa únicamente los DATOS OFICIALES ACTUALES DEL PROYECTO, los lotes "
        "del CRM y la ficha de conocimiento cargada. El historial sirve para entender la "
        "conversación, pero NUNCA es una fuente para confirmar hechos.",
        "- PROHIBIDO INVENTAR: no deduzcas ni completes direcciones, ciudades, teléfonos, "
        "precios, áreas, disponibilidad, fechas, amenidades, enlaces o condiciones de pago.",
        "- Si un dato exacto no aparece en las fuentes oficiales, responde: "
        "\"No tengo ese dato exacto registrado. Puedo confirmarlo con el equipo de SUCOL.\"",
        "- Si una respuesta anterior del historial contradice los datos oficiales actuales, "
        "corrígela de forma explícita y usa solamente el dato oficial.",
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
    if proyecto:
        direccion = proyecto.get("direccion_visita")
        maps_url = proyecto.get("google_maps_url")
        if direccion:
            lineas.append(
                f"- La ÚNICA dirección autorizada para visitas de este proyecto es: {direccion}."
            )
        else:
            lineas.append(
                "- Este proyecto NO tiene una dirección de visita registrada. No proporciones ninguna."
            )
        if maps_url:
            lineas.append(f"- El ÚNICO enlace autorizado de ubicación es: {maps_url}.")
    return "\n".join(lineas)


_PATRON_DIRECCION = re.compile(
    r"\b(carrera|cra\.?|calle|cl\.?|transversal|diagonal|avenida|av\.?|"
    r"oficina|local)\b|#",
    re.IGNORECASE,
)


def _numeros_direccion(texto: str) -> set[str]:
    return set(re.findall(r"\d+", texto or ""))


def _direccion_coincide_con_oficial(texto: str, proyecto: dict | None) -> bool:
    """
    Valida direcciones urbanas exactas.

    Las ubicaciones generales por vía o kilómetro pertenecen a la ficha comercial y
    no se comparan contra la dirección del punto de atención.
    """
    if not _PATRON_DIRECCION.search(texto or ""):
        return True
    direccion = (proyecto or {}).get("direccion_visita") or ""
    if not direccion:
        return False
    numeros_oficiales = _numeros_direccion(direccion)
    numeros_respuesta = _numeros_direccion(texto)
    return bool(numeros_oficiales) and numeros_oficiales.issubset(numeros_respuesta)


def _sanitizar_historial(
    historial: list[dict],
    proyecto: dict | None,
) -> list[dict]:
    """
    Elimina respuestas previas del asistente que contienen direcciones no respaldadas
    por el CRM. Evita que una alucinación guardada se convierta en falsa fuente.
    """
    limpio = []
    for mensaje in historial:
        contenido = mensaje.get("content", "")
        if (
            mensaje.get("role") == "assistant"
            and _PATRON_DIRECCION.search(contenido)
            and not _direccion_coincide_con_oficial(contenido, proyecto)
        ):
            logger.warning("Historial: dirección no oficial descartada del contexto")
            continue
        limpio.append(mensaje)
    return limpio


def _validar_respuesta_oficial(
    respuesta: str,
    proyecto: dict | None,
) -> str:
    """Impide enviar una dirección que no coincida con el dato oficial del CRM."""
    if _direccion_coincide_con_oficial(respuesta, proyecto):
        return respuesta

    nombre = (proyecto or {}).get("nombre") or "este proyecto"
    direccion = (proyecto or {}).get("direccion_visita")
    maps_url = (proyecto or {}).get("google_maps_url")
    logger.error(f"Respuesta bloqueada por dirección no oficial: {respuesta[:180]}")

    if not direccion:
        return (
            f"No tengo una dirección de visita oficial registrada para {nombre}. "
            "Puedo confirmarla con el equipo de SUCOL antes de que te desplaces."
        )

    respuesta_segura = f"La dirección oficial registrada para visitar {nombre} es {direccion}."
    if maps_url:
        respuesta_segura += f" Puedes guiarte con Google Maps: {maps_url}"
    return respuesta_segura


_PATRON_PREGUNTA_VISITA = re.compile(
    r"\b(d[oó]nde|direcci[oó]n|ubicaci[oó]n|c[oó]mo\s+llegar|"
    r"c[oó]mo\s+voy|a\s+d[oó]nde|para\s+una\s+visita|ir\s+a\s+ver|"
    r"visitar|visita\s+presencial)\b",
    re.IGNORECASE,
)

_PATRON_SOLICITUD_ASESOR = re.compile(
    r"\b(asesor(?:a)?|persona\s+real|persona\s+humana|humano|ejecutiv[oa]|"
    r"hablar\s+con\s+alguien|contacto\s+humano|tel[eé]fono\s+de|"
    r"n[uú]mero\s+de|whatsapp\s+de)\b",
    re.IGNORECASE,
)

_PATRON_MENCION_ASESOR = re.compile(
    r"\b(tu|el|la|un|una)\s+asesor(?:a)?\b|"
    r"\basesor(?:a)?\s+asignad[oa]\b|"
    r"\bte\s+conect[oa]\s+con\b",
    re.IGNORECASE,
)

_PATRON_SIN_DISPONIBILIDAD = re.compile(
    r"\b(no\s+(?:contamos|tenemos|hay)\s+(?:con\s+)?(?:unidades|lotes|"
    r"eco-h[aá]bitats?)\s+disponibles|sin\s+unidades\s+disponibles|"
    r"agotad[oa]s?)\b",
    re.IGNORECASE,
)


def _cliente_pide_asesor(mensaje: str) -> bool:
    return bool(_PATRON_SOLICITUD_ASESOR.search(mensaje or ""))


def _quitar_mencion_asesor_no_solicitada(
    respuesta: str,
    mensaje_cliente: str,
    asesor: dict | None,
) -> str:
    """Elimina párrafos que exponen o promocionan al asesor sin solicitud expresa."""
    if _cliente_pide_asesor(mensaje_cliente):
        return respuesta

    nombre = str((asesor or {}).get("nombre") or "").strip()
    telefono = re.sub(r"\D", "", str((asesor or {}).get("telefono") or ""))
    bloques = re.split(r"\n\s*\n", respuesta or "")
    limpios = []

    for bloque in bloques:
        bloque_digitos = re.sub(r"\D", "", bloque)
        menciona_datos = bool(
            (nombre and nombre.lower() in bloque.lower())
            or (telefono and telefono in bloque_digitos)
            or _PATRON_MENCION_ASESOR.search(bloque)
        )
        if menciona_datos:
            logger.warning("Respuesta: mención proactiva del asesor eliminada")
            continue
        limpios.append(bloque.strip())

    return "\n\n".join(b for b in limpios if b).strip()


def _corregir_disponibilidad(
    respuesta: str,
    lotes: list[dict],
    proyecto: dict | None,
) -> str:
    """Bloquea afirmaciones de inventario agotado cuando el CRM tiene disponibles."""
    if not lotes or not _PATRON_SIN_DISPONIBILIDAD.search(respuesta or ""):
        return respuesta

    nombre = (proyecto or {}).get("nombre") or "este proyecto"
    areas = sorted(
        {str(lote.get("area_m2")) for lote in lotes if lote.get("area_m2") is not None},
        key=lambda valor: float(valor),
    )
    detalle = f" en áreas de {', '.join(areas)} m²" if areas else ""
    logger.error("Respuesta: falsa indisponibilidad reemplazada con inventario CRM")
    return (
        f"Sí tenemos opciones disponibles actualmente en {nombre}{detalle}. "
        "Puedo mostrarte precios y alternativas o agendar una visita, llamada o cita virtual. "
        "¿Cuál opción prefieres?"
    )


_PATRON_RESPUESTA_INCOMPLETA = re.compile(
    r"\b(para|sobre|del|de|el|la|los|las|proyecto|informaci[oó]n)\s*$",
    re.IGNORECASE,
)


def _respuesta_resumen_crm(
    proyecto: dict | None,
    lotes: list[dict],
) -> str:
    """Construye un resumen factual cuando la salida del modelo queda incompleta."""
    nombre = (proyecto or {}).get("nombre") or "este proyecto"

    if lotes:
        areas_numericas = sorted(
            {
                float(lote["area_m2"])
                for lote in lotes
                if lote.get("area_m2") is not None
            }
        )
        precios = [
            float(lote["precio_total"])
            for lote in lotes
            if lote.get("precio_total") is not None
        ]

        detalles = [f"{len(lotes)} opciones disponibles registradas"]
        if areas_numericas:
            area_min = f"{areas_numericas[0]:g}"
            area_max = f"{areas_numericas[-1]:g}"
            if area_min == area_max:
                detalles.append(f"de {area_min} m²")
            else:
                detalles.append(f"con áreas entre {area_min} y {area_max} m²")
        if precios:
            detalles.append(f"con precios desde ${min(precios):,.0f}")

        return (
            f"{nombre} tiene {', '.join(detalles)} según el inventario actual. "
            "Puedo ampliarte características y financiación, o agendar una visita, "
            "llamada o cita virtual. ¿Qué deseas conocer primero?"
        )

    return (
        f"Puedo darte la información oficial disponible de {nombre} y ayudarte a agendar "
        "una visita, llamada o cita virtual. ¿Qué aspecto quieres conocer primero?"
    )


def _respuesta_es_incompleta(respuesta: str) -> bool:
    texto = re.sub(r"\s+", " ", respuesta or "").strip()
    if len(texto) < 45:
        return True
    if _PATRON_RESPUESTA_INCOMPLETA.search(texto):
        return True
    palabras = re.findall(r"\w+", texto, re.UNICODE)
    return len(palabras) < 8


def _procesar_respuesta_cliente(
    respuesta: str,
    mensaje_cliente: str,
    proyecto: dict | None,
    lotes: list[dict],
    asesor: dict | None,
) -> str:
    respuesta = _validar_respuesta_oficial(respuesta, proyecto)
    respuesta = _corregir_disponibilidad(respuesta, lotes, proyecto)
    respuesta = _quitar_mencion_asesor_no_solicitada(
        respuesta,
        mensaje_cliente,
        asesor,
    )
    if respuesta and not _respuesta_es_incompleta(respuesta):
        return respuesta

    logger.error(f"Respuesta incompleta reemplazada: {respuesta!r}")
    return _respuesta_resumen_crm(proyecto, lotes)


def _respuesta_operativa_visita(
    mensaje: str,
    proyecto: dict | None,
) -> str | None:
    """
    Responde preguntas operativas de ubicación sin generación libre.

    Direcciones y protocolos de acceso son datos de alto riesgo: se copian del
    CRM o de una política explícita del proyecto, nunca se delegan al modelo.
    """
    if not proyecto or not _PATRON_PREGUNTA_VISITA.search(mensaje or ""):
        return None

    slug = str(proyecto.get("slug") or "").lower()
    nombre = proyecto.get("nombre") or "el proyecto"

    if slug == "cascata":
        return (
            "Para conocer Cascata, el primer paso es hacer el recorrido virtual 360° en "
            "https://cascata360.sucol.co. La visita presencial se agenda después del "
            "recorrido virtual y de la autorización de la Dirección Comercial. "
            "¿Quieres que agendemos primero la sesión virtual?"
        )

    direccion = proyecto.get("direccion_visita")
    maps_url = proyecto.get("google_maps_url")
    if direccion:
        respuesta = f"Para visitar {nombre}, la dirección oficial es {direccion}."
        if maps_url:
            respuesta += f" Puedes guiarte aquí: {maps_url}"
        respuesta += " ¿Qué día y hora deseas agendar?"
        return respuesta

    return (
        f"No hay una dirección oficial de visita registrada para {nombre}. "
        "Antes de que te desplaces, puedo solicitar al equipo de SUCOL que confirme "
        "el punto autorizado."
    )


async def generar_respuesta_con_tools(
    mensaje: str,
    historial: list[dict],
    contexto_lead: dict | None = None,
    lotes_disponibles: list[dict] | None = None,
    telefono: str = "",
    asesor: dict | None = None,
    agendamientos: list[dict] | None = None,
    proyecto_slug: str | None = None,
    proyecto: dict | None = None,
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

    respuesta_operativa = _respuesta_operativa_visita(mensaje, proyecto)
    if respuesta_operativa:
        return respuesta_operativa

    prompt_final = _construir_prompt_base(proyecto_slug, proyecto)

    asesor_para_contexto = asesor if _cliente_pide_asesor(mensaje) else None
    contexto_crm = _construir_contexto_crm(
        contexto_lead,
        lotes_disponibles or [],
        asesor_para_contexto,
        agendamientos or [],
        proyecto,
    )
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    # Reglas finales de prioridad máxima — siempre al final para prevalecer sobre el prompt base
    prompt_final += _reglas_finales(asesor_para_contexto, proyecto)

    historial_limpio = _sanitizar_historial(historial, proyecto)
    mensajes: list = [
        {"role": m["role"], "content": m["content"]}
        for m in historial_limpio
    ]
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

        return _procesar_respuesta_cliente(
            respuesta,
            mensaje,
            proyecto,
            lotes_disponibles or [],
            asesor,
        )

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
    proyecto: dict | None = None,
) -> str:
    """
    Genera una respuesta usando Gemini API.
    El system prompt se construye desde config/prompts.yaml + knowledge/global.md
    + knowledge/proyectos/[proyecto_slug].md (si se conoce el proyecto).
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    respuesta_operativa = _respuesta_operativa_visita(mensaje, proyecto)
    if respuesta_operativa:
        return respuesta_operativa

    prompt_final = _construir_prompt_base(proyecto_slug, proyecto)

    asesor_para_contexto = asesor if _cliente_pide_asesor(mensaje) else None
    contexto_crm = _construir_contexto_crm(
        contexto_lead,
        lotes_disponibles or [],
        asesor_para_contexto,
        agendamientos or [],
        proyecto,
    )
    if contexto_crm:
        prompt_final += "\n\n" + contexto_crm

    prompt_final += _reglas_finales(asesor_para_contexto, proyecto)

    historial_limpio = _sanitizar_historial(historial, proyecto)
    mensajes = [
        {"role": m["role"], "content": m["content"]}
        for m in historial_limpio
    ]
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
        return _procesar_respuesta_cliente(
            respuesta,
            mensaje,
            proyecto,
            lotes_disponibles or [],
            asesor,
        )

    except Exception as e:
        logger.error(f"Error Gemini API: {e}")
        return _mensaje_error()
