# agent/brain.py — Cerebro de Sofía: conexión con la Anthropic Messages API
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
import holidays
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

# claude-haiku-4-5: el modelo más económico y rápido de la familia actual
# ($1.00 / $5.00 por millón de tokens input/output), suficiente para las
# respuestas cortas (max_tokens=1024) de un bot de atención por WhatsApp.
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
_ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
_ANTHROPIC_VERSION = "2023-06-01"


def _bloque_a_json(block) -> dict:
    """Convierte un bloque de contenido (dict ya serializable o SimpleNamespace
    devuelto por una respuesta previa) al formato JSON que espera la API."""
    if isinstance(block, dict):
        return block
    tipo = getattr(block, "type", None)
    if tipo == "text":
        return {"type": "text", "text": block.text}
    if tipo == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": tipo}


def _serializar_mensaje(mensaje: dict) -> dict:
    content = mensaje.get("content")
    if isinstance(content, list):
        return {"role": mensaje["role"], "content": [_bloque_a_json(b) for b in content]}
    return {"role": mensaje["role"], "content": content}


def _bloque_desde_json(block: dict) -> SimpleNamespace:
    tipo = block.get("type")
    if tipo == "text":
        return SimpleNamespace(type="text", text=block.get("text", ""))
    if tipo == "tool_use":
        return SimpleNamespace(
            type="tool_use",
            id=block.get("id"),
            name=block.get("name"),
            input=block.get("input", {}),
        )
    return SimpleNamespace(type=tipo)


class _AnthropicMessages:
    async def create(self, model: str, max_tokens: int, system: str, messages: list, tools: list | None = None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no configurada")

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [_serializar_mensaje(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(
                f"{_ANTHROPIC_BASE_URL}/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        usage_data = data.get("usage", {})
        usage = SimpleNamespace(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )
        content = [_bloque_desde_json(b) for b in data.get("content", [])]
        return SimpleNamespace(
            stop_reason=data.get("stop_reason"),
            content=content,
            usage=usage,
        )


class _AnthropicClient:
    def __init__(self):
        self.messages = _AnthropicMessages()


client = _AnthropicClient()

_BASE_DIR = Path(__file__).resolve().parent.parent
_KNOWLEDGE_DIR = _BASE_DIR / "knowledge"


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
    ruta_global = _KNOWLEDGE_DIR / "global.md"
    if ruta_global.is_file():
        try:
            with ruta_global.open("r", encoding="utf-8") as f:
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
        candidatos = [
            normalizado,
            normalizado.replace("_de_", "_"),
            normalizado.replace("_del_", "_"),
        ]
        ruta_proyecto = next(
            (
                _KNOWLEDGE_DIR / "proyectos" / f"{candidato}.md"
                for candidato in dict.fromkeys(candidatos)
                if (_KNOWLEDGE_DIR / "proyectos" / f"{candidato}.md").is_file()
            ),
            _KNOWLEDGE_DIR / "proyectos" / f"{normalizado}.md",
        )
        if ruta_proyecto.is_file():
            try:
                with ruta_proyecto.open("r", encoding="utf-8") as f:
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


def _dato_ficha_proyecto(proyecto: dict | None, campo: str) -> str:
    """Lee un dato de la tabla Markdown de la ficha oficial del proyecto."""
    slug = str((proyecto or {}).get("slug") or "").strip()
    if not slug:
        return ""
    contenido = _cargar_knowledge(slug, proyecto)
    patron = re.compile(
        rf"^\|\s*\*\*{re.escape(campo)}\*\*\s*\|\s*(.+?)\s*\|\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    coincidencia = patron.search(contenido)
    return coincidencia.group(1).strip() if coincidencia else ""


def _ubicacion_oficial_proyecto(proyecto: dict | None) -> str:
    return str((proyecto or {}).get("ubicacion") or "").strip() or _dato_ficha_proyecto(
        proyecto,
        "Ubicación",
    )


def _cargar_config_prompts() -> dict:
    try:
        with (_BASE_DIR / "config" / "prompts.yaml").open("r", encoding="utf-8") as f:
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


# ── Disponibilidad de visitas presenciales por proyecto ──────────────────────
# Grupo A (sala de ventas, Jamundí): lunes a sábado, excepto festivos.
# Grupo B (visita en el terreno): martes a domingo, excepto festivos. Si el
# lunes de esa semana es festivo, ese lunes se habilita y el martes se cierra
# (el descanso semanal se corre al martes).
_CO_FESTIVOS = holidays.country_holidays("CO")

_SLUGS_GRUPO_B_VISITA = {"vientos_ginebra", "reservas_ilama"}

_DIAS_SEMANA_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _hoy_colombia():
    """Fecha (date) actual en Colombia, UTC-5."""
    return datetime.now(timezone(timedelta(hours=-5))).date()


def _grupo_visita_proyecto(proyecto: dict | None) -> str:
    """'B' para proyectos con visita en el terreno propio, 'A' para el resto (sala de ventas)."""
    slug = str((proyecto or {}).get("slug") or "").strip().lower()
    return "B" if slug in _SLUGS_GRUPO_B_VISITA else "A"


def _dia_habil_visita(fecha, grupo: str) -> bool:
    """Determina si `fecha` (date) admite visitas presenciales para el grupo dado."""
    es_festivo = fecha in _CO_FESTIVOS
    dow = fecha.weekday()  # 0 = lunes ... 6 = domingo

    if grupo == "A":
        if dow == 6:  # domingo, siempre cerrado
            return False
        return not es_festivo

    # Grupo B: martes(1) a domingo(6) habitual, lunes(0) cerrado salvo festivo.
    lunes_de_la_semana = fecha - timedelta(days=dow)
    lunes_es_festivo = lunes_de_la_semana in _CO_FESTIVOS

    if dow == 0:  # lunes
        return es_festivo  # solo abre si ESE lunes es festivo
    if dow == 1:  # martes
        if lunes_es_festivo:
            return False  # el descanso se corrió al martes
        return not es_festivo
    return not es_festivo  # miércoles a domingo


# Horas reales de atención de asesores (24h), por día de la semana.
# Grupo A (Jamundí, sala de ventas): lunes a viernes 8am-5pm, sábado 8am-12pm, domingo cerrado.
# Grupo B (Vientos de Ginebra, Reservas de Ilama): martes a viernes 8am-5pm, sábado y domingo 8am-1pm.
_HORAS_GRUPO_A = {0: (8, 17), 1: (8, 17), 2: (8, 17), 3: (8, 17), 4: (8, 17), 5: (8, 12)}
_HORAS_GRUPO_B = {1: (8, 17), 2: (8, 17), 3: (8, 17), 4: (8, 17), 5: (8, 13), 6: (8, 13)}


def _horario_dia(fecha, grupo: str) -> tuple[int, int] | None:
    """Rango de horas (inicio, fin) en 24h en que el asesor atiende `fecha`, o None si no atiende."""
    if not _dia_habil_visita(fecha, grupo):
        return None
    dow = fecha.weekday()
    if grupo == "A":
        return _HORAS_GRUPO_A.get(dow)
    if dow == 0:
        # Grupo B: el lunes solo abre cuando ESE lunes es festivo (descanso corrido a martes),
        # y en ese caso opera con el horario normal de entre semana.
        return _HORAS_GRUPO_A[0]
    return _HORAS_GRUPO_B.get(dow)


def _formatear_hora_legible(hora24: int) -> str:
    if hora24 == 0:
        return "12am"
    if hora24 == 12:
        return "12pm"
    if hora24 < 12:
        return f"{hora24}am"
    return f"{hora24 - 12}pm"


def _proximos_dias_visita(proyecto: dict | None, cantidad: int = 5) -> list:
    """Retorna las próximas `cantidad` fechas (date) hábiles para visita presencial."""
    grupo = _grupo_visita_proyecto(proyecto)
    fecha = _hoy_colombia() + timedelta(days=1)  # se agenda desde mañana
    resultado = []
    intentos = 0
    while len(resultado) < cantidad and intentos < 30:
        if _dia_habil_visita(fecha, grupo):
            resultado.append(fecha)
        fecha += timedelta(days=1)
        intentos += 1
    return resultado


def _formatear_fecha_es(fecha) -> str:
    return f"{_DIAS_SEMANA_ES[fecha.weekday()]} {fecha.day} de {_MESES_ES[fecha.month - 1]}"


def _validar_slot_cita(fecha_cita: str, hora_cita: str, proyecto: dict | None) -> str | None:
    """
    Valida fecha y hora de una cita contra el horario real de asesores del proyecto
    (día de la semana + rango de horas, festivos y descansos ya considerados).

    Retorna None si el slot es válido, o un mensaje con alternativas reales si no lo es.
    Esto evita que el modelo agende una fecha pasada o una hora fuera de horario.
    """
    try:
        fecha = date.fromisoformat(str(fecha_cita))
        hora = int(str(hora_cita).split(":")[0])
    except (ValueError, IndexError, TypeError):
        return (
            "No logré interpretar esa fecha o esa hora. ¿Me confirmas el día y la hora "
            "exacta, por ejemplo 'sábado 25 de julio a las 10am'?"
        )

    hoy = _hoy_colombia()
    grupo = _grupo_visita_proyecto(proyecto)
    rango = _horario_dia(fecha, grupo) if fecha > hoy else None

    if rango and rango[0] <= hora < rango[1]:
        return None

    alternativas = _proximos_dias_visita(proyecto, 3)
    opciones = []
    for d in alternativas:
        r = _horario_dia(d, grupo)
        if r:
            opciones.append(
                f"{_formatear_fecha_es(d)} entre {_formatear_hora_legible(r[0])} "
                f"y {_formatear_hora_legible(r[1])}"
            )
    texto_opciones = "; ".join(opciones) if opciones else "los próximos días hábiles del asesor"

    motivo = "esa fecha ya pasó" if fecha <= hoy else "el asesor no atiende en ese horario"
    return f"Esa fecha u hora no está disponible porque {motivo}. Opciones reales: {texto_opciones}."


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


# Estructura estándar SUCOL (5% sep + 25% CI a 6 meses + 70% a 60 meses).
# Excepciones documentadas en knowledge/global.md — mismo slug que en proyectos.slug.
_FINANCIACION_ESTANDAR = {
    "separacion_pct": 0.05, "cuota_inicial_pct": 0.25, "cuota_inicial_meses": 6, "saldo_meses": 60,
}
_FINANCIACION_PROYECTOS = {
    "reservas_ilama": {"separacion_pct": 0.05, "cuota_inicial_pct": 0.15, "cuota_inicial_meses": 6, "saldo_meses": 24},
    "maloka_mallki": {"separacion_pct": 0.05, "cuota_inicial_pct": 0.15, "cuota_inicial_meses": 2, "saldo_meses": 18},
    "cascata": {"separacion_pct": 0.05, "cuota_inicial_pct": 0.10, "cuota_inicial_meses": 2, "saldo_meses": 36},
    # 5% sep + 20% CI al mes siguiente (pago único) + 75% a 36 meses directo con la
    # promotora (Promotora Vientos de Ginebra S.A.S.) — NO es SERCAPITAL.
    "vientos_ginebra": {"separacion_pct": 0.05, "cuota_inicial_pct": 0.20, "cuota_inicial_meses": 1, "saldo_meses": 36},
}


def _estructura_financiacion(proyecto: dict | None) -> dict:
    slug = str((proyecto or {}).get("slug") or "").lower()
    return _FINANCIACION_PROYECTOS.get(slug, _FINANCIACION_ESTANDAR)


def _simular_financiacion(precio: float, estructura: dict) -> dict:
    """Calcula separación/cuota inicial/saldo en Python — nunca se le pide a Claude que haga esta aritmética."""
    precio = float(precio)
    separacion = precio * estructura["separacion_pct"]
    cuota_inicial_total = precio * estructura["cuota_inicial_pct"]
    cuota_inicial_mensual = cuota_inicial_total / estructura["cuota_inicial_meses"]
    saldo_pct = 1 - estructura["separacion_pct"] - estructura["cuota_inicial_pct"]
    saldo_mensual = (precio * saldo_pct) / estructura["saldo_meses"]
    return {
        "separacion": separacion,
        "cuota_inicial_mensual": cuota_inicial_mensual,
        "cuota_inicial_meses": estructura["cuota_inicial_meses"],
        "saldo_mensual": saldo_mensual,
        "saldo_meses": estructura["saldo_meses"],
    }


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
        ubicacion = _ubicacion_oficial_proyecto(proyecto)
        if ubicacion:
            partes.append(f"- Ubicación oficial: {ubicacion}")
        if proyecto.get("direccion_visita"):
            partes.append(f"- Punto de atención o visita autorizado: {proyecto['direccion_visita']}")
        else:
            partes.append("- Punto de atención o visita autorizado: NO REGISTRADO")
        if proyecto.get("google_maps_url"):
            partes.append(f"- Google Maps del punto de atención o visita: {proyecto['google_maps_url']}")
        if proyecto.get("indicaciones_visita"):
            partes.append(f"- Indicaciones oficiales: {proyecto['indicaciones_visita']}")
        partes.append(
            "REGLA: estos datos operativos del CRM prevalecen sobre la ficha, el historial "
            "y cualquier conocimiento general. No confundas el punto de atención con la "
            "ubicación física del proyecto. Nunca completes ni deduzcas una dirección."
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

        estructura_fin = _estructura_financiacion(proyecto)
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
                sim = _simular_financiacion(minimo, estructura_fin)
                linea += (
                    f" | Simulación estándar sobre ${float(minimo):,.0f}: "
                    f"separación ${sim['separacion']:,.0f}, "
                    f"cuota inicial ≈ ${sim['cuota_inicial_mensual']:,.0f}/mes "
                    f"x {sim['cuota_inicial_meses']} meses, "
                    f"saldo ≈ ${sim['saldo_mensual']:,.0f}/mes x {sim['saldo_meses']} meses"
                )
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
        "- PROACTIVIDAD: después de responder, invita de forma natural a un siguiente paso, "
        "pero respeta la intención del cliente. Si pide información por este medio, brochure, "
        "PDF, fotos, algo digital o dice que debe revisarlo con su pareja/familia, primero "
        "entrégale material o un resumen concreto y NO lo empujes de inmediato a visita o llamada.",
        "- ASESOR: no menciones que el cliente tiene asesor asignado, ni su nombre, teléfono "
        "o correo. No ofrezcas 'conectarlo con su asesor'. Solo puedes hacerlo si el cliente "
        "pide explícitamente hablar con una persona o solicita el contacto de un asesor.",
        "- NO menciones herramientas externas como 'Kommo', 'CRM Kommo' ni ningún "
        "sistema que no sea el CRM de Sucol. Esos sistemas ya no existen.",
        "- NO digas que 'no tienes acceso al CRM' ni que 'no puedes consultar datos'. "
        "Toda la información del cliente y del asesor ya está en tu contexto.",
        "- FUENTES: usa únicamente los DATOS OFICIALES ACTUALES DEL PROYECTO, los lotes "
        "del CRM, la ficha de conocimiento cargada y el DIRECTORIO DE PROYECTOS ACTIVOS "
        "del conocimiento global. El historial sirve para entender la conversación, pero "
        "NUNCA es una fuente para confirmar hechos.",
        "- PROHIBIDO INVENTAR: no deduzcas ni completes direcciones, ciudades, teléfonos, "
        "precios, áreas, disponibilidad, fechas, amenidades, enlaces o condiciones de pago.",
        "- Si un dato exacto no aparece en las fuentes oficiales, responde: "
        "\"No tengo ese dato exacto registrado. Puedo confirmarlo con el equipo de SUCOL.\"",
        "- Si una respuesta anterior del historial contradice los datos oficiales actuales, "
        "corrígela de forma explícita y usa solamente el dato oficial.",
        "- Si el cliente pide el telefono, WhatsApp, correo o contacto de un asesor por nombre "
        "especifico, usa la herramienta consultar_asesor_por_nombre antes de responder.",
        "- CAMBIO DE INTERES: si el cliente menciona otra zona, barrio, ciudad o municipio "
        "distinto al proyecto actual, revisa primero el DIRECTORIO DE PROYECTOS ACTIVOS. "
        "Si allí hay uno o más proyectos en esa zona, menciónalos por nombre en tu "
        "respuesta y no insistas en el proyecto actual.",
        "- Solo si NINGÚN proyecto del directorio está en esa zona, di: "
        "\"No tengo registrado en este chat un proyecto exactamente en esa zona\" y pregunta "
        "si busca lote urbano pequeno, lote campestre o si considera otro municipio cercano.",
        "- No digas \"no tenemos proyectos disponibles en X\" a menos que una fuente oficial "
        "lo confirme explicitamente.",
        "- Si el cliente dice \"no\", \"no me interesa\" o \"busco otra zona\", no vuelvas "
        "a ofrecer el mismo proyecto en la siguiente respuesta.",
        "- CIERRE: si el cliente dice \"gracias\", \"muchas gracias\", \"con la informacion "
        "actual esta bien\", \"no muchas gracias\" o \"por ahora no\", responde breve y "
        "amable, sin ofrecer visita, llamada, cita virtual ni mas informacion.",
        "- Si el cliente pregunta por medidas, areas o tamanos de lotes, responde directamente "
        "con el rango o areas oficiales disponibles. No cambies a una invitacion generica.",
        "- Si el cliente pregunta por financiacion, responde condiciones registradas y pide el "
        "dato minimo para simular cuotas (lote de interes, cuota inicial o presupuesto). No "
        "reemplaces la respuesta por una llamada si todavia quiere informacion por WhatsApp.",
        "- CALCULO DE CUOTAS: nunca calcules separacion, cuota inicial ni saldo mensual de "
        "memoria. Usa EXCLUSIVAMENTE las cifras ya calculadas en 'Simulación estándar' dentro "
        "de la seccion de lotes del CRM. Si el lote o precio exacto que pide el cliente no "
        "tiene una simulacion en el contexto, di que no tienes esa cifra exacta y que la "
        "confirmas con el equipo de SUCOL — no inventes ni estimes el numero tu misma.",
        "- TAMAÑOS Y PRECIOS GENERALES: si el cliente pregunta por tamaños o precios sin pedir "
        "un lote exacto, y el CRM no trae lotes en tabla para este proyecto, usa los campos "
        "'Áreas' y 'Precios' de la ficha del proyecto (ej. 'desde 168 m² hasta 607 m²', "
        "'desde $103.000.000') — son datos oficiales de la ficha, no una invención tuya. La "
        "regla de 'no inventar cifras' aplica a simulaciones de cuota y a precios de lotes "
        "específicos que no aparecen en ninguna fuente, no a los rangos generales ya escritos "
        "en la ficha del proyecto.",
        "- CIERRE FALSO: frases como 'quedo atento', 'quedo pendiente', 'estoy pendiente', "
        "'ok', 'dale' o 'listo' indican que el cliente SIGUE interesado y espera tu siguiente "
        "mensaje — NO son un cierre de conversacion. No respondas con una despedida tipo "
        "'quedo atenta si mas adelante...'. Continua la conversacion con normalidad: confirma "
        "brevemente y haz la siguiente pregunta o propuesta relevante.",
        "- AGENDAMIENTO SIN FECHA: si el cliente ya confirmo que quiere agendar (ej. 'si', "
        "'sii', 'dale', 'listo') pero no dio dia ni hora, no repitas la misma pregunta abierta "
        "una segunda vez. Propon 2 franjas concretas de dia y hora dentro del horario de "
        "asesores para que el cliente solo tenga que elegir una.",
        "- CIERRE DE AGENDAMIENTO: si ya tienes en la conversacion el nombre completo, "
        "telefono, dia y hora del cliente para una cita, llama la herramienta confirmar_cita "
        "en este mismo turno. No vuelvas a preguntar por datos que el cliente ya te dio antes "
        "en el historial.",
        "- TELEFONO DISTINTO: si el numero que el cliente te da ahora es diferente al que dio "
        "antes en esta misma conversacion, preguntale cual es el correcto antes de llamar "
        "confirmar_cita.",
        "- LENGUAJE DE CONFIRMACION: nunca digas 'listo', 'agendado', 'quedó registrado' ni "
        "nada que suene a que la cita YA está guardada mientras todavía te falte un dato "
        "(nombre completo, dia u hora) o no hayas llamado la herramienta confirmar_cita. "
        "Mientras falte info usa lenguaje condicional: 'Perfecto, para confirmar tu visita "
        "del [dia] a las [hora] necesito tu nombre completo'. Solo usa 'listo'/'agendado' "
        "DESPUES de que confirmar_cita se haya ejecutado.",
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

        grupo_horario = _grupo_visita_proyecto(proyecto)
        dias_disponibles = _proximos_dias_visita(proyecto)
        if dias_disponibles:
            partes_dias = []
            for d in dias_disponibles:
                rango = _horario_dia(d, grupo_horario)
                if rango:
                    partes_dias.append(
                        f"{_formatear_fecha_es(d)} de {_formatear_hora_legible(rango[0])} "
                        f"a {_formatear_hora_legible(rango[1])}"
                    )
            dias_legibles = "; ".join(partes_dias)
            lineas.append(
                f"- DÍAS Y HORARIOS REALES del asesor para cita, llamada o visita de este "
                f"proyecto (ya excluye festivos y descansos según la política de Sucol): "
                f"{dias_legibles}. SOLO puedes proponer o aceptar día y hora dentro de estos "
                "rangos exactos — nunca fuera de esas horas ni en una fecha que ya pasó. Si el "
                "cliente pide un día u hora que no está en este rango, dile sin dar "
                "explicaciones internas que ahí no hay disponibilidad y ofrécele directamente "
                "la opción válida más cercana de esta lista."
            )
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


_PATRON_NEGACION_ZONA = re.compile(
    r"no tengo registrado en este chat un proyecto exactamente en ([^.!?\n]+)",
    re.IGNORECASE,
)


def _sanitizar_historial(
    historial: list[dict],
    proyecto: dict | None,
) -> list[dict]:
    """
    Elimina respuestas previas del asistente que contienen direcciones no respaldadas
    por el CRM, o negaciones de cobertura de zona que ya quedaron obsoletas (por
    ejemplo, si el directorio de proyectos activos ahora sí cubre esa zona). Evita
    que una alucinación o un error guardado se convierta en falsa fuente repetida.
    """
    historial = _recortar_historial_desde_inicio_proyecto(historial, proyecto)
    slug_actual = str((proyecto or {}).get("slug") or "").strip()
    limpio = []
    for mensaje in historial:
        contenido = mensaje.get("content", "")
        if mensaje.get("role") != "assistant":
            limpio.append(mensaje)
            continue
        if _PATRON_DIRECCION.search(contenido) and not _direccion_coincide_con_oficial(contenido, proyecto):
            logger.warning("Historial: dirección no oficial descartada del contexto")
            continue
        negacion = _PATRON_NEGACION_ZONA.search(contenido)
        if negacion and _zona_cubierta_por_otro_proyecto(negacion.group(1).strip(), slug_actual):
            logger.warning("Historial: negación de zona obsoleta descartada del contexto")
            continue
        limpio.append(mensaje)
    return limpio


def _normalizar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    return re.sub(r"\s+", " ", texto).strip()


def _recortar_historial_desde_inicio_proyecto(
    historial: list[dict],
    proyecto: dict | None,
) -> list[dict]:
    """
    Si el mismo telefono tuvo conversaciones de otros proyectos, usa solo el tramo
    posterior a la ultima plantilla inicial del proyecto actual.
    """
    if not historial or not proyecto:
        return historial

    nombre = _normalizar_texto(str(proyecto.get("nombre") or ""))
    slug = _normalizar_texto(str(proyecto.get("slug") or "").replace("_", " ").replace("-", " "))
    candidatos = [valor for valor in (nombre, slug) if valor]
    if not candidatos:
        return historial

    inicio = None
    for i, mensaje in enumerate(historial):
        if mensaje.get("role") != "assistant":
            continue
        contenido = _normalizar_texto(mensaje.get("content", ""))
        if "asistente virtual" in contenido and any(valor in contenido for valor in candidatos):
            inicio = i

    if inicio is not None and inicio > 0:
        logger.info(
            "Historial recortado desde plantilla del proyecto actual: "
            f"{len(historial)} -> {len(historial[inicio:])}"
        )
        return historial[inicio:]

    return historial


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
    r"\b(direcci[oó]n\s+(?:para|de)\s+(?:la\s+)?visita|c[oó]mo\s+llegar\s+(?:a\s+)?(?:la\s+)?visita|"
    r"c[oó]mo\s+voy|para\s+una\s+visita|ir\s+a\s+ver|visitar|visita\s+presencial|"
    r"direcci[oó]n\s+de\s+atenci[oó]n|"
    r"(?:me\s+(?:das|puedes\s+dar|compartes|env[ií]as|mandas|pasas)|cu[aá]l\s+es|"
    r"env[ií]ame|m[aá]ndame|pas[aá]me)\s+la\s+direcci[oó]n\b)\b",
    re.IGNORECASE,
)

_PATRON_UBICACION_PROYECTO = re.compile(
    r"\b(?:en\s+)?d[oó]nde\s+(?:se\s+)?(?:queda|est[aá]|est[aá]n|estan|encuentra|encuentran|ubica|ubican)\b|"
    r"\bubicaci[oó]n(?:\s+exacta)?\b|"
    r"\bdirecci[oó]n\s+del\s+(?:proyecto|lote|terreno)\b|"
    r"\blugar\s+exacto\b|"
    r"\best[aá]n\s+ubicad[oa]s?\b|"
    r"\bestan\s+ubicad[oa]s?\b",
    re.IGNORECASE,
)

_PATRON_IMAGENES_PROYECTO = re.compile(
    r"\b(im[aá]gen(?:es)?|fotos?|galer[ií]a|ver\s+(?:el\s+)?(?:lugar|proyecto))\b",
    re.IGNORECASE,
)

_PATRON_BROCHURE_PROYECTO = re.compile(
    r"\b(brochure|brochur|broshur|cat[aá]logo|catalogo|pdf|folleto)\b",
    re.IGNORECASE,
)

_PATRON_INFO_DIGITAL_PROYECTO = re.compile(
    r"\b(por\s+(?:este\s+)?medio|por\s+aqu[ií]|algo\s+digital|material\s+digital|"
    r"para\s+(?:yo\s+)?(?:guiarme|revisar|ver)|mientras\s+(?:se\s+)?(?:cuenta|coordina)|"
    r"lo\s+(?:hablo|reviso|miro)\s+con\s+mi\s+(?:esposa|esposo|pareja|familia))\b",
    re.IGNORECASE,
)

_PATRON_OTRO_TEMA_COMERCIAL = re.compile(
    r"\b(tama[ñn]os?|[aá]reas?|metraje|m2|m²|precios?|valor(?:es)?|"
    r"cu[aá]nto\s+(?:cuesta|vale)|costo|financiaci[oó]n|cuota)\b",
    re.IGNORECASE,
)

_BROCHURES_POR_SLUG = {
    "buenavista": "https://drive.google.com/file/d/1K-tjU8z0iJuFr-e9hCSPhcoJxcE5xmt0/view?usp=share_link",
    "vientos_de_ginebra": "https://drive.google.com/file/d/19OKqC1txQhXk3BEanYG3HjJ_LC-GUL9K/view?usp=share_link",
    "vientos-de-ginebra": "https://drive.google.com/file/d/19OKqC1txQhXk3BEanYG3HjJ_LC-GUL9K/view?usp=share_link",
    "bora": "https://drive.google.com/file/d/12PU5oGhKcxSK1rVag4CrMbQs34cAxW0C/view?usp=share_link",
    "reservas_de_ilama": "https://drive.google.com/file/d/1VQgZ5CDTt4zsonxhsl_hqI063dwsZP4k/view?usp=share_link",
    "reservas-de-ilama": "https://drive.google.com/file/d/1VQgZ5CDTt4zsonxhsl_hqI063dwsZP4k/view?usp=share_link",
    "maloka_mallki": "https://drive.google.com/file/d/1giN-M_VfjKi64U1XJbaqtk3I8HdCs_IB/view?usp=share_link",
    "maloka-mallki": "https://drive.google.com/file/d/1giN-M_VfjKi64U1XJbaqtk3I8HdCs_IB/view?usp=share_link",
    "solares_del_cabuyal": "https://drive.google.com/file/d/1vr6BoXABZgt2IWXHSkHb-rvRwMmUZ3eA/view?usp=share_link",
    "solares-del-cabuyal": "https://drive.google.com/file/d/1vr6BoXABZgt2IWXHSkHb-rvRwMmUZ3eA/view?usp=share_link",
    "santa_elena": "https://drive.google.com/file/d/1poayGpz8G9jpnul0625nCjE2OUTHXVN1/view?usp=share_link",
    "santa-elena": "https://drive.google.com/file/d/1poayGpz8G9jpnul0625nCjE2OUTHXVN1/view?usp=share_link",
    "forestal_garden": "https://drive.google.com/file/d/1-gPUj_xkOud7f9It9BaQHAjDMJPUklwz/view?usp=share_link",
    "forestal-garden": "https://drive.google.com/file/d/1-gPUj_xkOud7f9It9BaQHAjDMJPUklwz/view?usp=share_link",
    "praderas_de_guachinte": "https://drive.google.com/file/d/1qNoawlq0wE3vSPJZmEbNtpQEU-gsKjaI/view?usp=share_link",
    "praderas-de-guachinte": "https://drive.google.com/file/d/1qNoawlq0wE3vSPJZmEbNtpQEU-gsKjaI/view?usp=share_link",
}

_PATRON_VIO_PUBLICIDAD = re.compile(
    r"^\s*(?:yo\s+)?vi\s+(?:la\s+)?(?:publicidad|pauta|anuncio)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_PATRON_REFERENCIA_ENLACE = re.compile(
    r"^\s*en\s+(?:el\s+)?(?:enlace|link|anuncio|publicidad)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_PATRON_SOLICITA_INFO_PROYECTO = re.compile(
    r"^\s*(?:s[ií]|si|claro|dale|ok|bueno|perfecto)[,.\s]*(?:"
    r"cu[eé]ntame(?:\s+m[aá]s)?|"
    r"quiero\s+(?:m[aá]s\s+)?informaci[oó]n|"
    r"env[ií]ame\s+(?:m[aá]s\s+)?informaci[oó]n|"
    r"me\s+interesa|"
    r"informaci[oó]n"
    r")\s*[.!?]*$|"
    r"^\s*(?:cu[eé]ntame(?:\s+m[aá]s)?|"
    r"quiero\s+(?:m[aá]s\s+)?informaci[oó]n|"
    r"env[ií]ame\s+(?:m[aá]s\s+)?informaci[oó]n)\s*[.!?]*$",
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


_PATRON_FUERA_HORARIO_SOFIA = re.compile(
    r"\b(?:estamos|est[aá]n|sucol|sof[ií]a|el\s+chat)\s+"
    r"(?:fuera\s+de\s+horario|cerrad[oa]s?)\b|"
    r"\bfuera\s+de\s+horario\b",
    re.IGNORECASE,
)


def _horario_legible_proyecto(proyecto: dict | None) -> str:
    """Descripción legible del horario real de asesores para este proyecto."""
    if _grupo_visita_proyecto(proyecto) == "B":
        return "martes a viernes de 8am a 5pm, y sábados y domingos de 8am a 1pm"
    return "lunes a viernes de 8am a 5pm, y sábados de 8am a 12pm (domingos no hay atención)"


def _corregir_fuera_horario_sofia(
    respuesta: str,
    mensaje_cliente: str,
    proyecto: dict | None = None,
) -> str:
    """Sofia atiende siempre; el horario solo aplica a asesores fisicos."""
    if not _PATRON_FUERA_HORARIO_SOFIA.search(respuesta or ""):
        return respuesta

    logger.error("Respuesta corregida: Sofía no debe decir que está fuera de horario")
    horario = _horario_legible_proyecto(proyecto)
    if _PATRON_SOLICITUD_ASESOR.search(mensaje_cliente or "") or _PATRON_INTENCION_CITA.search(mensaje_cliente or ""):
        return (
            "Sofía puede ayudarte por WhatsApp en este momento. "
            f"Las llamadas, visitas y atención con asesores físicos se programan en horario de "
            f"asesores: {horario}. ¿Quieres que dejemos una llamada programada?"
        )

    return (
        "Sofía puede ayudarte por WhatsApp en este momento. "
        f"El horario aplica solo para llamadas, visitas o atención con asesores físicos: "
        f"{horario}. ¿Qué información necesitas?"
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
        "Puedo mostrarte precios, alternativas o financiación por este medio. "
        "¿Qué quieres revisar primero?"
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
            "Puedo ampliarte características, financiación o comparar opciones por este medio. "
            "¿Qué deseas conocer primero?"
        )

    return (
        f"Puedo darte la información oficial disponible de {nombre} por este medio. "
        "¿Qué aspecto quieres conocer primero?"
    )


def _respuesta_solicitud_info_proyecto(
    mensaje: str,
    proyecto: dict | None,
    lotes: list[dict],
) -> str | None:
    """Responde intenciones simples de 'si, cuentame mas' sin depender del modelo."""
    if not proyecto or not _PATRON_SOLICITA_INFO_PROYECTO.search(mensaje or ""):
        return None
    return _respuesta_resumen_crm(proyecto, lotes)


_PATRON_PREGUNTA_AGENDAMIENTO = re.compile(
    r"(d[ií]a y (?:la )?hora|hora exacta|qu[eé] d[ií]a|\d{1,2}\s*(?:am|pm)|te sirve|"
    r"prefieres.*hora|agendar|confirmar (?:tu|la) (?:visita|cita|llamada)|"
    r"nombre (?:completo )?y tel[eé]fono|nombre y (?:tu )?n[uú]mero)",
    re.IGNORECASE,
)


def _en_medio_de_agendamiento(historial: list[dict] | None) -> bool:
    """Detecta si el último mensaje de Sofía dejó una pregunta de día/hora sin resolver."""
    if not historial:
        return False
    for msg in reversed(historial):
        if msg.get("role") != "assistant":
            continue
        return bool(_PATRON_PREGUNTA_AGENDAMIENTO.search(msg.get("content", "") or ""))
    return False


_PATRON_PUNTUACION_FINAL = re.compile(r"[.!?…]['\"”)]*$")


def _respuesta_es_incompleta(respuesta: str) -> bool:
    """Detecta respuestas cortadas a mitad de frase — no respuestas simplemente cortas.

    El prompt le exige a Sofía ser breve (máximo 3 oraciones), así que una respuesta
    corta pero bien puntuada es válida. Solo se marca como incompleta si termina en
    una palabra colgante (preposición/artículo) o, siendo muy corta, no cierra con
    puntuación final (señal de truncamiento real).
    """
    texto = re.sub(r"\s+", " ", respuesta or "").strip()
    if not texto:
        return True
    if _PATRON_RESPUESTA_INCOMPLETA.search(texto):
        return True
    if len(texto) < 45 and not _PATRON_PUNTUACION_FINAL.search(texto):
        return True
    return False


def _procesar_respuesta_cliente(
    respuesta: str,
    mensaje_cliente: str,
    proyecto: dict | None,
    lotes: list[dict],
    asesor: dict | None,
    historial: list[dict] | None = None,
    resultado_cita_confirmada: str | None = None,
    ultimo_resultado_tool: str | None = None,
) -> str:
    if _menciona_proyecto_distinto(respuesta, proyecto):
        nombre = (proyecto or {}).get("nombre") or "el proyecto registrado"
        logger.error(
            "Respuesta bloqueada por nombre de proyecto no oficial: %s",
            (respuesta or "")[:180],
        )
        return (
            f"La información oficial que tengo asociada a este chat corresponde a {nombre}. "
            "¿Qué deseas conocer sobre este proyecto?"
        )
    respuesta = _corregir_fuera_horario_sofia(respuesta, mensaje_cliente, proyecto)
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
    if resultado_cita_confirmada:
        # La cita ya se guardó exitosamente en este turno (tool confirmar_cita).
        # Nunca reemplaces esa confirmación real por un mensaje que sugiera
        # que falta información — el cliente vería un dato falso.
        return resultado_cita_confirmada
    if _en_medio_de_agendamiento(historial):
        return (
            "Disculpa, se me cortó la respuesta. ¿Me confirmas de nuevo el día, "
            "la hora y tu nombre completo?"
        )
    if ultimo_resultado_tool:
        # Ya se ejecutó una herramienta en este turno (p.ej. confirmar_cita rechazando
        # un horario inválido con alternativas reales). Usar ese resultado en vez de
        # un resumen genérico que ignoraría todo lo que el cliente ya conversó.
        return ultimo_resultado_tool
    return _respuesta_resumen_crm(proyecto, lotes)


def separar_mensajes_whatsapp(
    respuesta: str,
    proyecto: dict | None = None,
) -> list[str]:
    """
    Separa la información y la invitación final en dos mensajes de WhatsApp.

    Si el modelo no generó una pregunta final, agrega una CTA segura de acuerdo
    con el protocolo del proyecto.
    """
    texto = (respuesta or "").strip()
    if not texto:
        return []

    if texto in (_mensaje_error(), _mensaje_fallback()):
        return [texto]

    inicio_pregunta = texto.rfind("¿")
    if inicio_pregunta > 40:
        informacion = texto[:inicio_pregunta].strip()
        pregunta = texto[inicio_pregunta:].strip()
        if informacion and pregunta:
            return [informacion, pregunta]

    slug = str((proyecto or {}).get("slug") or "").lower()
    if slug == "cascata":
        cta = "¿Quieres recibir más información o agendar primero el recorrido virtual 360°?"
    else:
        cta = (
            "¿Qué te gustaría revisar ahora: áreas, precios o financiación?"
        )
    return [texto, cta]


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
    if (
        _PATRON_BROCHURE_PROYECTO.search(mensaje or "")
        or _PATRON_INFO_DIGITAL_PROYECTO.search(mensaje or "")
    ):
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
        f"No hay un punto autorizado de visita registrado para {nombre}. "
        "Antes de que te desplaces, puedo solicitar al equipo de SUCOL que confirme "
        "el punto autorizado."
    )


def _urls_recursos_proyecto(proyecto: dict | None) -> list[str]:
    """Obtiene únicamente enlaces de material comercial registrados en el CRM."""
    urls: list[str] = []
    for campo in ("galeria_url", "imagenes_url", "brochure_url", "sitio_web_url"):
        valor = (proyecto or {}).get(campo)
        valores = valor if isinstance(valor, list) else [valor]
        for item in valores:
            item = str(item or "").strip()
            if item.startswith(("https://", "http://")) and item not in urls:
                urls.append(item)
    return urls[:3]


def _url_brochure_proyecto(proyecto: dict | None) -> str:
    """Obtiene el brochure oficial desde CRM o desde el mapa local autorizado."""
    brochure_crm = str((proyecto or {}).get("brochure_url") or "").strip()
    if brochure_crm.startswith(("https://", "http://")):
        return brochure_crm

    slug = str((proyecto or {}).get("slug") or "").strip().lower()
    slug_normalizado = slug.replace(" ", "_").replace("-", "_")
    return (
        _BROCHURES_POR_SLUG.get(slug)
        or _BROCHURES_POR_SLUG.get(slug_normalizado)
        or ""
    )


def _respuesta_recursos_proyecto(
    mensaje: str,
    proyecto: dict | None,
) -> str | None:
    """Responde imágenes y ubicación sin generación libre ni direcciones inferidas."""
    pide_imagenes = bool(_PATRON_IMAGENES_PROYECTO.search(mensaje or ""))
    pide_ubicacion = bool(_PATRON_UBICACION_PROYECTO.search(mensaje or ""))
    pide_brochure = bool(_PATRON_BROCHURE_PROYECTO.search(mensaje or ""))
    pide_info_digital = bool(_PATRON_INFO_DIGITAL_PROYECTO.search(mensaje or ""))
    if not proyecto or not (pide_imagenes or pide_ubicacion or pide_brochure or pide_info_digital):
        return None

    # Si el cliente pidió esto junto con tamaños, precios o financiación, no lo
    # respondamos a medias: dejamos que el modelo conteste todo junto con el
    # contexto completo del CRM en vez de contestar solo ubicación/imágenes/brochure.
    if _PATRON_OTRO_TEMA_COMERCIAL.search(mensaje or ""):
        return None

    nombre = proyecto.get("nombre") or "este proyecto"
    partes = []
    if pide_brochure or pide_info_digital:
        brochure_url = _url_brochure_proyecto(proyecto)
        if brochure_url:
            partes.append(f"Claro, te comparto el brochure de {nombre}: {brochure_url}")
        else:
            partes.append(
                f"Aún no tengo un brochure oficial de {nombre} registrado para enviarte por este chat."
            )

    if pide_imagenes:
        urls = _urls_recursos_proyecto(proyecto)
        if urls:
            partes.append(f"Puedes ver el material oficial de {nombre} aquí: {' '.join(urls)}")
        else:
            brochure_url = _url_brochure_proyecto(proyecto)
            if brochure_url:
                partes.append(f"No tengo galería aparte, pero te comparto el brochure de {nombre}: {brochure_url}")
            else:
                partes.append(
                    f"Aún no tengo una galería oficial de {nombre} registrada para enviarte por este chat."
                )

    if pide_ubicacion:
        ubicacion = _ubicacion_oficial_proyecto(proyecto)
        if ubicacion:
            partes.append(f"La ubicación general oficial es: {ubicacion}.")
        else:
            partes.append("No tengo registrada la ubicación exacta del proyecto.")

        mapa_proyecto = str(
            proyecto.get("google_maps_proyecto_url")
            or proyecto.get("ubicacion_maps_url")
            or ""
        ).strip()
        if mapa_proyecto.startswith(("https://", "http://")):
            partes.append(f"Mapa oficial del proyecto: {mapa_proyecto}")
        else:
            partes.append(
                "Todavía no tengo registrado el enlace oficial del terreno para compartirte la ubicación exacta."
            )

    if pide_info_digital and not (pide_ubicacion or pide_imagenes):
        partes.append("Si quieres, por aquí mismo te ayudo a revisar áreas, precios o financiación.")

    return " ".join(partes)


def _respuesta_referencia_publicidad(
    mensaje: str,
    proyecto: dict | None,
) -> str | None:
    """Aclara referencias breves al anuncio sin asumir que preguntan una dirección."""
    nombre = (proyecto or {}).get("nombre") or "el proyecto"
    if _PATRON_VIO_PUBLICIDAD.search(mensaje or ""):
        return (
            f"Entiendo, viste la publicidad de {nombre}. "
            "¿Quieres confirmar precios, áreas disponibles o ubicación?"
        )
    if _PATRON_REFERENCIA_ENLACE.search(mensaje or ""):
        return (
            f"Entiendo. ¿Qué información del enlace de {nombre} quieres confirmar: "
            "precios, áreas, imágenes o ubicación?"
        )
    return None


_PATRON_ZONA_EXPLICITA = re.compile(
    r"\b(poblado\s+campestre|cali|jamund[ií]|ginebra|potrerito|rozo|palmira|"
    r"candelaria|yumbo|valle\s+del\s+cauca)\b",
    re.IGNORECASE,
)

_PATRON_RECHAZO_CORTO = re.compile(
    r"^\s*(no|nop|no gracias|no muchas gracias|no me interesa|busco otra zona|otra zona)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_PATRON_CIERRE_CONVERSACION = re.compile(
    r"^\s*(gracias|muchas gracias|mil gracias|listo gracias|ok gracias|"
    r"con la informaci[oó]n( actual)? (est[aá] )?bien|"
    r"con eso (est[aá] )?bien|por ahora no|no por ahora|ahora no|"
    r"no en este momento|en este momento no|m[aá]s adelante|"
    r"lo reviso despu[eé]s|no muchas gracias|no gracias)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_RAZONES_DESCARTE = {
    5: {
        "categoria_id": "sin-capacidad",
        "categoria_label": "Sin capacidad",
        "razon_id": 5,
        "razon_label": "No tiene capacidad de cuota actual",
        "reactivable": "si",
    },
    6: {
        "categoria_id": "sin-capacidad",
        "categoria_label": "Sin capacidad",
        "razon_id": 6,
        "razon_label": "No tiene cuota inicial disponible",
        "reactivable": "si",
    },
    7: {
        "categoria_id": "sin-capacidad",
        "categoria_label": "Sin capacidad",
        "razon_id": 7,
        "razon_label": "Reportó deudas activas o mora",
        "reactivable": "si",
    },
    8: {
        "categoria_id": "sin-capacidad",
        "categoria_label": "Sin capacidad",
        "razon_id": 8,
        "razon_label": "Perfil no califica para financiamiento",
        "reactivable": "si",
    },
    9: {
        "categoria_id": "sin-interes",
        "categoria_label": "Sin interés",
        "razon_id": 9,
        "razon_label": "Perdió el interés sin razón clara",
        "reactivable": "si",
    },
    10: {
        "categoria_id": "sin-interes",
        "categoria_label": "Sin interés",
        "razon_id": 10,
        "razon_label": "Dijo explícitamente que no le interesa",
        "reactivable": "depende",
    },
    11: {
        "categoria_id": "sin-interes",
        "categoria_label": "Sin interés",
        "razon_id": 11,
        "razon_label": "Cambió de objetivo: no quiere lote",
        "reactivable": "si",
    },
    13: {
        "categoria_id": "sin-interes",
        "categoria_label": "Sin interés",
        "razon_id": 13,
        "razon_label": "Le gustó pero no quiere comprar ahora",
        "reactivable": "si",
    },
    14: {
        "categoria_id": "competencia",
        "categoria_label": "Competencia",
        "razon_id": 14,
        "razon_label": "Compró en otro proyecto de SUCOL",
        "reactivable": "na",
    },
    15: {
        "categoria_id": "competencia",
        "categoria_label": "Competencia",
        "razon_id": 15,
        "razon_label": "Compró en proyecto de la competencia",
        "reactivable": "si",
    },
    16: {
        "categoria_id": "competencia",
        "categoria_label": "Competencia",
        "razon_id": 16,
        "razon_label": "Prefiere invertir en otro activo",
        "reactivable": "si",
    },
    17: {
        "categoria_id": "timing",
        "categoria_label": "Timing",
        "razon_id": 17,
        "razon_label": "Compra en más de 12 meses",
        "reactivable": "si",
    },
    18: {
        "categoria_id": "timing",
        "categoria_label": "Timing",
        "razon_id": 18,
        "razon_label": "Está esperando vender otro inmueble",
        "reactivable": "si",
    },
    19: {
        "categoria_id": "timing",
        "categoria_label": "Timing",
        "razon_id": 19,
        "razon_label": "Situación personal o familiar en pausa",
        "reactivable": "si",
    },
    20: {
        "categoria_id": "timing",
        "categoria_label": "Timing",
        "razon_id": 20,
        "razon_label": "Esperando bono o ingreso extraordinario",
        "reactivable": "si",
    },
}

_PATRON_DESCARTE_EXPLICITO = re.compile(
    r"\b(no me interesa|no estoy interesado|no estoy interesad[ao]|"
    r"no quiero|ya no me interesa|no gracias|no muchas gracias)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_POR_AHORA = re.compile(
    r"\b(por ahora no|ahora no|todav[ií]a no|m[aá]s adelante|"
    r"en este momento no|no por ahora)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_NO_LOTE = re.compile(
    r"\b(no quiero lote|ya no quiero lote|busco casa hecha|busco apartamento|"
    r"quiero apartamento|quiero casa construida)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_CUOTA = re.compile(
    r"\b(no puedo pagar la cuota|cuota muy alta|no me da la cuota|"
    r"no tengo capacidad de cuota)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_INICIAL = re.compile(
    r"\b(no tengo (la )?(cuota inicial|inicial)|sin cuota inicial|"
    r"no cuento con (la )?inicial)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_MORA = re.compile(
    r"\b(estoy reportad[ao]|tengo deudas|estoy en mora|centrales de riesgo|datacr[eé]dito)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_FINANCIAMIENTO = re.compile(
    r"\b(no me aprobaron|no califico|no califico para financiaci[oó]n|"
    r"no soy sujeto de cr[eé]dito)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_COMPETENCIA_SUCOL = re.compile(
    r"\b(compr[eé]|separ[eé]|ya compr[eé]|ya separ[eé])\b.*\b(sucol)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_COMPETENCIA = re.compile(
    r"\b(compr[eé]|separ[eé]|ya compr[eé]|ya separ[eé])\b.*\b(otro proyecto|otra constructora|competencia)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_OTRO_ACTIVO = re.compile(
    r"\b(prefiero invertir|voy a invertir)\b.*\b(apartamento|casa|local|negocio|acciones|cdt|otro activo)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_MAS_12_MESES = re.compile(
    r"\b(en|dentro de|para)\s+(1\s*a[nñ]o|un\s*a[nñ]o|12\s*meses|"
    r"m[aá]s de\s+(1\s*a[nñ]o|un\s*a[nñ]o|12\s*meses)|"
    r"el pr[oó]ximo a[nñ]o|el otro a[nñ]o)\b|"
    r"\b(pr[oó]ximo a[nñ]o|otro a[nñ]o|m[aá]s de 12 meses|m[aá]s de un a[nñ]o)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_VENDER_INMUEBLE = re.compile(
    r"\b(vender|venda|vendamos|venda mi|vender mi|vender otro|"
    r"venta de)\b.*\b(casa|apartamento|apto|inmueble|propiedad|lote)\b|"
    r"\b(cuando venda|hasta vender)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_PAUSA_PERSONAL = re.compile(
    r"\b(situaci[oó]n personal|tema personal|problema familiar|tema familiar|"
    r"asunto familiar|estoy en pausa|dej[eé]moslo en pausa|pausar|"
    r"no es buen momento)\b",
    re.IGNORECASE,
)

_PATRON_DESCARTE_BONO_INGRESO = re.compile(
    r"\b(bono|prima|cesant[ií]as|liquidaci[oó]n|ingreso extraordinario|"
    r"pago extraordinario|cuando me paguen|cuando reciba)\b",
    re.IGNORECASE,
)

_PATRON_INTENCION_CITA = re.compile(
    r"\b(agendar|programar|reservar|coordinar)\b.*\b(llamada|cita|visita|videollamada)\b|"
    r"\b(llamada|cita|visita|videollamada)\b.*\b(agendar|programar|reservar|coordinar)\b",
    re.IGNORECASE,
)

_PATRON_RANGO_HORARIO = re.compile(
    r"\bentre\s+(?:las\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?"
    r"\s+(?:y|a|hasta)\s+(?:las\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)?\b",
    re.IGNORECASE,
)

_PATRON_NOMBRE_DESPUES_DE_PROYECTO = re.compile(
    r"\bproyecto[ \t]+(?:llamado[ \t]+)?"
    r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÜÑáéíóúüñ-]*"
    r"(?:[ \t]+(?:de|del|la|las|los|y|[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÜÑáéíóúüñ-]*)){0,4})"
)


def _respuesta_sin_proyecto(mensaje: str) -> str:
    """Evita que el modelo complete un proyecto cuando el CRM no identificó uno."""
    if _PATRON_INTENCION_CITA.search(mensaje or ""):
        return (
            "Claro. Antes de programarla necesito saber sobre cuál proyecto de SUCOL "
            "quieres hablar. ¿Cuál proyecto o zona te interesa?"
        )
    return (
        "Hola, soy Sofía de SUCOL. Para darte información correcta, "
        "¿qué proyecto o zona te interesa?"
    )


def _respuesta_rango_horario(mensaje: str) -> str | None:
    """Solicita una hora concreta para no elegir arbitrariamente dentro de un rango."""
    if not _PATRON_RANGO_HORARIO.search(mensaje or ""):
        return None
    return (
        "Perfecto. Para registrar la llamada necesito una hora exacta dentro de ese rango. "
        "¿La programamos a las 8:00, 8:30 o 9:00?"
    )


def clasificar_descarte_sofia(mensaje: str) -> dict | None:
    """Clasifica cierres descartables segun el catalogo oficial del CRM."""
    texto = mensaje or ""
    razon_id = None

    if _PATRON_DESCARTE_COMPETENCIA_SUCOL.search(texto):
        razon_id = 14
    elif _PATRON_DESCARTE_COMPETENCIA.search(texto):
        razon_id = 15
    elif _PATRON_DESCARTE_OTRO_ACTIVO.search(texto):
        razon_id = 16
    elif _PATRON_DESCARTE_VENDER_INMUEBLE.search(texto):
        razon_id = 18
    elif _PATRON_DESCARTE_BONO_INGRESO.search(texto):
        razon_id = 20
    elif _PATRON_DESCARTE_PAUSA_PERSONAL.search(texto):
        razon_id = 19
    elif _PATRON_DESCARTE_MAS_12_MESES.search(texto):
        razon_id = 17
    elif _PATRON_DESCARTE_NO_LOTE.search(texto):
        razon_id = 11
    elif _PATRON_DESCARTE_INICIAL.search(texto):
        razon_id = 6
    elif _PATRON_DESCARTE_CUOTA.search(texto):
        razon_id = 5
    elif _PATRON_DESCARTE_MORA.search(texto):
        razon_id = 7
    elif _PATRON_DESCARTE_FINANCIAMIENTO.search(texto):
        razon_id = 8
    elif _PATRON_DESCARTE_POR_AHORA.search(texto):
        razon_id = 13
    elif _PATRON_DESCARTE_EXPLICITO.search(texto):
        razon_id = 10

    if not razon_id:
        return None

    clasificacion = dict(_RAZONES_DESCARTE[razon_id])
    clasificacion["etapa_lead"] = "DESCARTADO"
    clasificacion["motivo"] = (
        f"Sofía descartó el lead por mensaje del cliente: {texto.strip()[:240]}"
    )
    return clasificacion


def _menciona_proyecto_distinto(respuesta: str, proyecto: dict | None) -> bool:
    """Detecta nombres presentados como proyecto que no coinciden con el CRM."""
    nombre_oficial = _normalizar_texto(str((proyecto or {}).get("nombre") or ""))
    if not nombre_oficial:
        return bool(_PATRON_NOMBRE_DESPUES_DE_PROYECTO.search(respuesta or ""))

    for coincidencia in _PATRON_NOMBRE_DESPUES_DE_PROYECTO.finditer(respuesta or ""):
        candidato = _normalizar_texto(coincidencia.group(1))
        if candidato in {"este", "el", "nuestro"}:
            continue
        if candidato not in nombre_oficial and nombre_oficial not in candidato:
            return True
    return False


def _resumen_cita_oficial(
    tipo_cita: str,
    proyecto: dict | None,
) -> str:
    """Genera el resumen operativo sin aceptar hechos redactados por el modelo."""
    nombre = str((proyecto or {}).get("nombre") or "").strip()
    tipo = str(tipo_cita or "cita").strip().lower()
    if nombre:
        return f"El cliente solicitó programar una {tipo} sobre el proyecto {nombre}."
    return f"El cliente solicitó programar una {tipo} con el equipo comercial de SUCOL."


def _texto_proyecto_para_matching(proyecto: dict | None) -> str:
    if not proyecto:
        return ""
    partes = [
        proyecto.get("nombre"),
        proyecto.get("slug"),
        proyecto.get("ubicacion"),
        proyecto.get("direccion_visita"),
        proyecto.get("indicaciones_visita"),
        _ubicacion_oficial_proyecto(proyecto),
    ]
    return _normalizar_texto(" ".join(str(p) for p in partes if p))


_UBICACION_FICHA_PATRON = re.compile(
    r"^\|\s*\*\*Ubicaci[oó]n\*\*\s*\|\s*(.+?)\s*\|\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _zona_cubierta_por_otro_proyecto(zona: str, slug_actual: str) -> bool:
    """
    Revisa las fichas de knowledge/proyectos/*.md (excluyendo la del proyecto
    activo) para saber si SUCOL ya tiene otro proyecto registrado en esa zona.

    Evita que Sofía niegue cobertura en una zona donde sí hay proyectos, solo
    porque no coincide con el proyecto que el CRM tiene asignado a este chat.
    """
    zona_normalizada = _normalizar_texto(zona)
    carpeta = _KNOWLEDGE_DIR / "proyectos"
    if not carpeta.is_dir():
        return False
    for ruta in carpeta.glob("*.md"):
        if slug_actual and ruta.stem == slug_actual:
            continue
        try:
            contenido = ruta.read_text(encoding="utf-8")
        except (OSError, IOError):
            continue
        coincidencia = _UBICACION_FICHA_PATRON.search(contenido)
        if coincidencia and zona_normalizada in _normalizar_texto(coincidencia.group(1)):
            return True
    return False


def _ultimo_asistente(historial: list[dict]) -> str:
    for mensaje in reversed(historial or []):
        if mensaje.get("role") == "assistant":
            return mensaje.get("content", "")
    return ""


def _respuesta_cambio_interes(
    mensaje: str,
    historial: list[dict],
    proyecto: dict | None,
) -> str | None:
    """Responde sin IA cuando el cliente se sale claramente del proyecto activo."""
    texto = _normalizar_texto(mensaje or "")
    if not texto:
        return None

    nombre_actual = _normalizar_texto(str((proyecto or {}).get("nombre") or ""))
    slug_actual = _normalizar_texto(
        str((proyecto or {}).get("slug") or "").replace("_", " ").replace("-", " ")
    )
    if any(valor and valor in texto for valor in (nombre_actual, slug_actual)):
        return None

    if _PATRON_RECHAZO_CORTO.search(mensaje or ""):
        ultimo = _normalizar_texto(_ultimo_asistente(historial))
        nombre = _normalizar_texto(str((proyecto or {}).get("nombre") or ""))
        if nombre and nombre in ultimo:
            return (
                "Entiendo. Para no enviarte información que no te sirve, "
                "¿lo buscas solo en esa zona o te sirve una alternativa cercana?"
            )
        return None

    zonas = [m.group(0) for m in _PATRON_ZONA_EXPLICITA.finditer(mensaje or "")]
    if not zonas or not proyecto:
        return None

    proyecto_texto = _texto_proyecto_para_matching(proyecto)
    zonas_fuera = [
        zona for zona in zonas
        if _normalizar_texto(zona) not in proyecto_texto
    ]
    if not zonas_fuera:
        return None

    zona_legible = zonas_fuera[0].strip()
    slug_actual = str((proyecto or {}).get("slug") or "").strip()
    if _zona_cubierta_por_otro_proyecto(zona_legible, slug_actual):
        return None
    return (
        f"No tengo registrado en este chat un proyecto exactamente en {zona_legible}. "
        "¿Buscas un lote urbano pequeño o también considerarías un lote campestre "
        "en otro municipio cercano?"
    )


def _respuesta_cierre_conversacion(
    mensaje: str,
    historial: list[dict],
    contexto_lead: dict | None = None,
) -> str | None:
    """Cierra sin CTA cuando el cliente agradece o dice que no necesita mas."""
    if not _PATRON_CIERRE_CONVERSACION.search(mensaje or ""):
        return None

    nombre = ""
    if contexto_lead:
        nombre = (
            contexto_lead.get("nombre_completo")
            or contexto_lead.get("nombre")
            or contexto_lead.get("primer_nombre")
            or ""
        )
        nombre = str(nombre).strip().split()[0] if nombre else ""

    cierres_previos = 0
    for msg in historial or []:
        if msg.get("role") != "user":
            continue
        if _PATRON_CIERRE_CONVERSACION.search(msg.get("content", "")):
            cierres_previos += 1

    saludo = f", {nombre}" if nombre else ""
    if cierres_previos:
        return f"Con gusto{saludo}. Quedo atenta si más adelante quieres retomar."
    return f"Con gusto{saludo}. Quedo atenta si necesitas algo más adelante."


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

    if not proyecto:
        return _respuesta_sin_proyecto(mensaje)

    respuesta_info = _respuesta_solicitud_info_proyecto(
        mensaje,
        proyecto,
        lotes_disponibles or [],
    )
    if respuesta_info:
        return respuesta_info

    respuesta_rango = _respuesta_rango_horario(mensaje)
    if respuesta_rango:
        return respuesta_rango

    respuesta_cierre = _respuesta_cierre_conversacion(mensaje, historial, contexto_lead)
    if respuesta_cierre:
        return respuesta_cierre

    respuesta_cambio = _respuesta_cambio_interes(mensaje, historial, proyecto)
    if respuesta_cambio:
        return respuesta_cambio

    respuesta_referencia = _respuesta_referencia_publicidad(mensaje, proyecto)
    if respuesta_referencia:
        return respuesta_referencia

    respuesta_recursos = _respuesta_recursos_proyecto(mensaje, proyecto)
    if respuesta_recursos:
        return respuesta_recursos

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
            model=_ANTHROPIC_MODEL,
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
            tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_CONSULTAR_ASESOR, _TOOL_NOTIFICAR_AREA, _TOOL_CALIFICAR_SIN_VISITA],
        )

        resultado_cita_confirmada = None
        ultimo_resultado_tool = None
        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            tool_results = []

            for tu in tool_uses:
                if tu.name == "confirmar_cita":
                    try:
                        datos_cita = dict(tu.input)
                        correccion = _validar_slot_cita(
                            datos_cita.get("fecha_cita", ""),
                            datos_cita.get("hora_cita", ""),
                            proyecto,
                        )
                        if correccion:
                            resultado_tool = correccion
                        else:
                            datos_cita["resumen"] = _resumen_cita_oficial(
                                datos_cita.get("tipo_cita", "cita"),
                                proyecto,
                            )
                            resultado_tool = await confirmar_cita(telefono=telefono, **datos_cita)
                            if resultado_tool.startswith("Cita agendada"):
                                resultado_cita_confirmada = resultado_tool
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

                ultimo_resultado_tool = resultado_tool
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
                model=_ANTHROPIC_MODEL,
                max_tokens=1024,
                system=prompt_final,
                messages=mensajes_con_resultado,
                tools=[_TOOL_CONFIRMAR_CITA, _TOOL_ESCALAR_ASESOR, _TOOL_CONSULTAR_ASESOR, _TOOL_NOTIFICAR_AREA, _TOOL_CALIFICAR_SIN_VISITA],
            )
            texto_response2 = next(
                (b.text for b in response2.content if getattr(b, "type", None) == "text"),
                None,
            )
            if texto_response2:
                respuesta = texto_response2
            else:
                # Claude no devolvió texto en la segunda vuelta (p.ej. intentó otra
                # tool_use encadenada, que no soportamos). No devolvemos un error
                # genérico: si una herramienta como confirmar_cita ya tuvo éxito,
                # el cliente debe ver esa confirmación real en vez de un mensaje falso.
                logger.warning(
                    f"response2 sin bloque de texto (stop_reason={response2.stop_reason}); "
                    "usando resultado_tool como respuesta"
                )
                respuesta = tool_results[-1]["content"] if tool_results else _mensaje_error()
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
            historial,
            resultado_cita_confirmada,
            ultimo_resultado_tool,
        )

    except Exception:
        logger.exception("Error Anthropic API (con tools)")
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
    Genera una respuesta usando la Anthropic API.
    El system prompt se construye desde config/prompts.yaml + knowledge/global.md
    + knowledge/proyectos/[proyecto_slug].md (si se conoce el proyecto).
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    if not proyecto:
        return _respuesta_sin_proyecto(mensaje)

    respuesta_info = _respuesta_solicitud_info_proyecto(
        mensaje,
        proyecto,
        lotes_disponibles or [],
    )
    if respuesta_info:
        return respuesta_info

    respuesta_rango = _respuesta_rango_horario(mensaje)
    if respuesta_rango:
        return respuesta_rango

    respuesta_cierre = _respuesta_cierre_conversacion(mensaje, historial, contexto_lead)
    if respuesta_cierre:
        return respuesta_cierre

    respuesta_cambio = _respuesta_cambio_interes(mensaje, historial, proyecto)
    if respuesta_cambio:
        return respuesta_cambio

    respuesta_referencia = _respuesta_referencia_publicidad(mensaje, proyecto)
    if respuesta_referencia:
        return respuesta_referencia

    respuesta_recursos = _respuesta_recursos_proyecto(mensaje, proyecto)
    if respuesta_recursos:
        return respuesta_recursos

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
            model=_ANTHROPIC_MODEL,
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
            historial,
        )

    except Exception:
        logger.exception("Error Anthropic API")
        return _mensaje_error()
