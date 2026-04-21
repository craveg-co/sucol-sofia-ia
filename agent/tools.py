# agent/tools.py — Herramientas de Sofía para Sucol Soluciones Urbanísticas
# Generado por AgentKit

"""
Herramientas específicas del negocio de Sucol.
Cubren los 3 casos de uso: FAQ, agendamiento de citas y calificación de leads.
"""

import os
import yaml
import logging
from datetime import datetime

from agent.crm import crear_agendamiento, obtener_lead
from agent.providers.meta import ProveedorMeta

logger = logging.getLogger("agentkit")

_meta_proveedor: ProveedorMeta | None = None


def _obtener_meta() -> ProveedorMeta:
    global _meta_proveedor
    if _meta_proveedor is None:
        _meta_proveedor = ProveedorMeta()
    return _meta_proveedor


def cargar_info_negocio() -> dict:
    """Carga la información del negocio desde config/business.yaml."""
    try:
        with open("config/business.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config/business.yaml no encontrado")
        return {}


def obtener_horario() -> dict:
    """Retorna el horario de atención de Sucol."""
    info = cargar_info_negocio()
    horario = info.get("negocio", {}).get("horario", "Lunes a Viernes 8am-6pm, Sábados 8am-5pm, Domingos 8am-4pm")

    # Verificar si estamos en horario de atención (hora Colombia UTC-5)
    from datetime import timezone, timedelta
    colombia_tz = timezone(timedelta(hours=-5))
    ahora = datetime.now(colombia_tz)
    hora_actual = ahora.hour
    dia_semana = ahora.weekday()  # 0=Lunes, 6=Domingo

    if dia_semana <= 4:  # Lunes a Viernes
        esta_abierto = 8 <= hora_actual < 18
    elif dia_semana == 5:  # Sábado
        esta_abierto = 8 <= hora_actual < 17
    else:  # Domingo
        esta_abierto = 8 <= hora_actual < 16

    return {
        "horario": horario,
        "esta_abierto": esta_abierto,
        "hora_actual_colombia": ahora.strftime("%H:%M"),
    }


def buscar_en_knowledge(consulta: str) -> str:
    """
    Busca información relevante en los archivos de /knowledge.
    Retorna el contenido más relevante encontrado.
    """
    resultados = []
    knowledge_dir = "knowledge"

    if not os.path.exists(knowledge_dir):
        return "No hay archivos de conocimiento disponibles."

    for archivo in os.listdir(knowledge_dir):
        ruta = os.path.join(knowledge_dir, archivo)
        if archivo.startswith(".") or not os.path.isfile(ruta):
            continue
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
                if consulta.lower() in contenido.lower():
                    resultados.append(f"[{archivo}]: {contenido[:500]}")
        except (UnicodeDecodeError, IOError):
            continue

    if resultados:
        return "\n---\n".join(resultados)
    return "No encontré información específica sobre eso en mis archivos."


# ════════════════════════════════════════
# HERRAMIENTA: Agendamiento de citas
# ════════════════════════════════════════

# Almacenamiento simple en memoria (en producción se usaría una DB)
_citas_pendientes: dict[str, dict] = {}


def iniciar_agendamiento(telefono: str, nombre: str) -> str:
    """
    Inicia el proceso de agendamiento de cita para un cliente.

    Args:
        telefono: Número del cliente
        nombre: Nombre completo del cliente

    Returns:
        Mensaje de confirmación
    """
    _citas_pendientes[telefono] = {
        "nombre": nombre,
        "estado": "pendiente_disponibilidad",
        "creada": datetime.utcnow().isoformat(),
    }
    logger.info(f"Agendamiento iniciado para {nombre} ({telefono})")
    return f"Cita iniciada para {nombre}"


async def confirmar_cita(
    telefono: str,
    tipo_cita: str,
    fecha: str,
    hora: str,
    resumen_conversacion: str,
    nombre_cliente: str,
    video_url: str = "",
) -> str:
    """
    Confirma y persiste una cita en Supabase y notifica al asesor via Meta plantilla.

    Args:
        telefono: Número del cliente (para buscar su lead)
        tipo_cita: "Cita Virtual" u otro tipo
        fecha: Fecha en formato YYYY-MM-DD
        hora: Hora en formato HH:MM
        resumen_conversacion: Resumen generado por Sofía
        nombre_cliente: Nombre del cliente para la notificación
        video_url: Enlace de videollamada (opcional)

    Returns:
        Mensaje indicando el resultado
    """
    # ── Buscar el lead para obtener IDs del asesor ────────────────────────────
    lead = await obtener_lead(telefono)
    if not lead:
        logger.warning(f"confirmar_cita: no se encontró lead para {telefono} — cita no guardada")
        return "No se pudo guardar la cita: lead no encontrado."

    lead_id = lead.get("id")
    asesor_id = lead.get("asesor_id")
    asesor_nombre = lead.get("asesor_responsable") or lead.get("asesor_asignado") or "Asesor Sucol"
    asesor_telefono = lead.get("asesor_telefono") or lead.get("telefono_asesor") or ""

    # ── Guardar la cita en Supabase ───────────────────────────────────────────
    datos_agendamiento = {
        "lead_id": lead_id,
        "tipo_cita": tipo_cita,
        "fecha_visita": fecha,
        "hora_llamada": hora,
        "resumen_conversacion": resumen_conversacion,
        "estado": "PENDIENTE",
        "asesor_id": asesor_id,
        "asesor_asignado": asesor_nombre,
        "video_url": video_url or "",
    }
    agendamiento = await crear_agendamiento(datos_agendamiento)
    if not agendamiento:
        logger.error(f"confirmar_cita: falló el INSERT para lead {lead_id} ({telefono})")
        return "No se pudo registrar la cita en el sistema."

    logger.info(f"Cita creada: id={agendamiento.get('id')} lead={lead_id} fecha={fecha} {hora}")

    # ── Notificar al asesor por WhatsApp (plantilla Meta) ────────────────────
    if not asesor_telefono:
        logger.warning(f"confirmar_cita: asesor sin teléfono para lead {lead_id} — notificación omitida")
    else:
        try:
            meta = _obtener_meta()
            enviado = await meta.enviar_plantilla_cita_asesor(
                telefono_asesor=asesor_telefono,
                asesor_nombre=asesor_nombre,
                tipo_cita=tipo_cita,
                fecha_cita=fecha,
                hora_cita=hora,
                nombre_cliente=nombre_cliente,
                telefono_cliente=telefono,
                resumen_conversacion=resumen_conversacion,
                video_url=video_url or "",
            )
            if enviado:
                logger.info(f"Notificación enviada a asesor {asesor_nombre} ({asesor_telefono})")
            else:
                logger.warning(f"No se pudo enviar notificación al asesor {asesor_telefono}")
        except Exception as e:
            logger.error(f"confirmar_cita: error enviando plantilla al asesor: {e}")

    return f"Cita registrada: {tipo_cita} el {fecha} a las {hora}."


# ════════════════════════════════════════
# HERRAMIENTA: Calificación de leads
# ════════════════════════════════════════

# Perfiles de leads detectados durante la conversación
_perfiles_leads: dict[str, dict] = {}


def registrar_perfil_lead(
    telefono: str,
    proposito: str = None,      # "vivir" | "invertir"
    ubicacion: str = None,      # "colombia" | "exterior"
    necesita_financiacion: bool = None,
    primer_contacto: bool = True,
) -> str:
    """
    Registra o actualiza el perfil de un lead.

    Args:
        telefono: Número del cliente
        proposito: Para qué quiere el lote (vivir o invertir)
        ubicacion: Si vive en Colombia o en el exterior
        necesita_financiacion: Si necesita financiación con Sercapital
        primer_contacto: Si es la primera vez que contacta a Sucol

    Returns:
        Resumen del perfil del lead
    """
    if telefono not in _perfiles_leads:
        _perfiles_leads[telefono] = {}

    perfil = _perfiles_leads[telefono]
    if proposito:
        perfil["proposito"] = proposito
    if ubicacion:
        perfil["ubicacion"] = ubicacion
    if necesita_financiacion is not None:
        perfil["necesita_financiacion"] = necesita_financiacion
    perfil["primer_contacto"] = primer_contacto
    perfil["ultima_interaccion"] = datetime.utcnow().isoformat()

    logger.info(f"Lead registrado/actualizado: {telefono} — {perfil}")
    return f"Perfil actualizado: {perfil}"


def obtener_perfil_lead(telefono: str) -> dict:
    """
    Retorna el perfil acumulado de un lead.

    Args:
        telefono: Número del cliente

    Returns:
        Diccionario con el perfil del lead
    """
    return _perfiles_leads.get(telefono, {})


def calificar_lead(telefono: str) -> str:
    """
    Califica la prioridad de un lead según su perfil.

    Returns:
        "alta" | "media" | "baja"
    """
    perfil = obtener_perfil_lead(telefono)
    if not perfil:
        return "sin_calificar"

    # Lead de alta prioridad: quiere invertir o tiene presupuesto definido
    if perfil.get("proposito") == "invertir":
        return "alta"
    if perfil.get("necesita_financiacion") is False:
        return "alta"

    # Lead de media prioridad: quiere vivir y necesita financiación
    if perfil.get("proposito") == "vivir":
        return "media"

    return "baja"
