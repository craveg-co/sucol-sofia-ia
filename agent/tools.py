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

logger = logging.getLogger("agentkit")


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


def confirmar_cita(telefono: str, disponibilidad: str) -> str:
    """
    Confirma una cita con la disponibilidad del cliente.

    Args:
        telefono: Número del cliente
        disponibilidad: Texto con día y hora preferida

    Returns:
        Mensaje de confirmación de la cita
    """
    if telefono in _citas_pendientes:
        _citas_pendientes[telefono]["disponibilidad"] = disponibilidad
        _citas_pendientes[telefono]["estado"] = "confirmada"
        nombre = _citas_pendientes[telefono]["nombre"]
        logger.info(f"Cita confirmada: {nombre} ({telefono}) — {disponibilidad}")
        return f"Cita confirmada para {nombre} en: {disponibilidad}"
    return "No se encontró una cita pendiente para este número."


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
