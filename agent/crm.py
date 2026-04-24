# agent/crm.py — Conexión de Sofía al CRM de Sucol

import os
import ssl
import logging
from datetime import date, time
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

# ── Conexión ───────────────────────────────────────────────────────────────────

_CRM_URL = os.getenv("CRM_DATABASE_URL") or os.getenv("DATABASE_URL", "")

if _CRM_URL.startswith("postgresql://"):
    _CRM_URL = _CRM_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _CRM_URL.startswith("postgres://"):
    _CRM_URL = _CRM_URL.replace("postgres://", "postgresql+asyncpg://", 1)

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

_crm_engine = None
_crm_session = None

if _CRM_URL:
    _crm_engine = create_async_engine(_CRM_URL, echo=False, connect_args={"ssl": _ssl_ctx})
    _crm_session = async_sessionmaker(_crm_engine, class_=AsyncSession, expire_on_commit=False)


def _crm_disponible() -> bool:
    return _crm_session is not None


# ── Proyectos ──────────────────────────────────────────────────────────────────

async def obtener_proyecto_por_slug(slug: str) -> dict | None:
    """Lee un proyecto completo por su slug."""
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT * FROM proyectos WHERE slug = :slug AND activo = true LIMIT 1"),
                {"slug": slug},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM obtener_proyecto_por_slug: {e}")
        return None


async def obtener_proyecto_por_telefono(telefono: str) -> dict | None:
    """
    Detecta el proyecto asignado a un teléfono con esta prioridad:
    1. contactos_whatsapp (fuente de verdad del chat)
    2. leads del CRM
    Si lo encuentra en leads pero no en contactos_whatsapp, lo registra automáticamente.
    """
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            # 1. Buscar en contactos_whatsapp
            result = await session.execute(
                text("""
                    SELECT p.*
                    FROM proyectos p
                    INNER JOIN contactos_whatsapp cw ON cw.proyecto_slug = p.slug
                    WHERE cw.telefono = :telefono AND p.activo = true
                    LIMIT 1
                """),
                {"telefono": telefono},
            )
            row = result.mappings().first()
            if row:
                return dict(row)

            # 2. Buscar en leads
            result = await session.execute(
                text("""
                    SELECT p.*
                    FROM proyectos p
                    INNER JOIN leads l ON l.proyecto = p.slug
                    WHERE l.telefono_principal = :telefono AND p.activo = true
                    LIMIT 1
                """),
                {"telefono": telefono},
            )
            row = result.mappings().first()
            if not row:
                return None

            proyecto = dict(row)

            # 3. Registrar en contactos_whatsapp para las próximas consultas
            try:
                await session.execute(
                    text("""
                        INSERT INTO contactos_whatsapp (telefono, proyecto_slug)
                        VALUES (:telefono, :slug)
                        ON CONFLICT (telefono) DO UPDATE SET proyecto_slug = EXCLUDED.proyecto_slug
                    """),
                    {"telefono": telefono, "slug": proyecto["slug"]},
                )
                await session.commit()
            except Exception as e:
                logger.warning(f"CRM auto-registro contacto desde lead: {e}")

            return proyecto

    except Exception as e:
        logger.error(f"CRM obtener_proyecto_por_telefono: {e}")
        return None


async def obtener_proyectos_activos() -> list[dict]:
    """Retorna slug y nombre de todos los proyectos activos."""
    if not _crm_disponible():
        return []
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT slug, nombre FROM proyectos WHERE activo = true ORDER BY nombre")
            )
            return [dict(row) for row in result.mappings().all()]
    except Exception as e:
        logger.error(f"CRM obtener_proyectos_activos: {e}")
        return []


async def detectar_proyecto_en_mensaje(mensaje: str) -> dict | None:
    """
    Busca si el mensaje menciona el nombre o slug de algún proyecto activo.
    Retorna el proyecto completo si encuentra coincidencia, None si no.
    """
    if not _crm_disponible():
        return None
    try:
        proyectos = await obtener_proyectos_activos()
        mensaje_lower = mensaje.lower()
        logger.info(f"CRM detección: {len(proyectos)} proyectos activos | mensaje='{mensaje_lower[:60]}'")
        for p in proyectos:
            slug_legible = p["slug"].replace("_", " ").replace("-", " ")
            nombre_lower = p["nombre"].lower()
            logger.info(f"  → comparando slug='{slug_legible}' nombre='{nombre_lower}'")
            if slug_legible in mensaje_lower or nombre_lower in mensaje_lower:
                logger.info(f"  ✓ coincidencia encontrada: {p['slug']}")
                return await obtener_proyecto_por_slug(p["slug"])
        logger.info("  ✗ ningún proyecto coincide con el mensaje")
        return None
    except Exception as e:
        logger.error(f"CRM detectar_proyecto_en_mensaje: {e}")
        return None


# ── Leads ──────────────────────────────────────────────────────────────────────

async def obtener_lead(telefono: str) -> dict | None:
    """Retorna todos los datos del lead desde la tabla leads."""
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT * FROM leads WHERE telefono_principal = :telefono LIMIT 1"),
                {"telefono": telefono},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM obtener_lead: {e}")
        return None


async def actualizar_lead_crm(telefono: str, datos: dict):
    """Actualiza campos del lead en la tabla leads del CRM."""
    if not _crm_disponible() or not datos:
        return
    try:
        sets = ", ".join(f"{k} = :{k}" for k in datos)
        params = {"telefono": telefono, **datos}
        async with _crm_session() as session:
            await session.execute(
                text(f"UPDATE leads SET {sets} WHERE telefono_principal = :telefono"),
                params,
            )
            await session.commit()
    except Exception as e:
        logger.error(f"CRM actualizar_lead_crm: {e}")


# ── Contactos WhatsApp ─────────────────────────────────────────────────────────

async def crear_o_actualizar_contacto_whatsapp(telefono: str, datos: dict):
    """Upsert en contactos_whatsapp usando telefono como clave única."""
    if not _crm_disponible() or not datos:
        return
    try:
        sets = ", ".join(f"{k} = EXCLUDED.{k}" for k in datos if k != "telefono")
        cols = ", ".join(["telefono"] + list(datos.keys()))
        vals = ", ".join([":telefono"] + [f":{k}" for k in datos])
        params = {"telefono": telefono, **datos}
        async with _crm_session() as session:
            await session.execute(
                text(f"""
                    INSERT INTO contactos_whatsapp ({cols})
                    VALUES ({vals})
                    ON CONFLICT (telefono) DO UPDATE SET {sets}
                """),
                params,
            )
            await session.commit()
    except Exception as e:
        logger.error(f"CRM crear_o_actualizar_contacto_whatsapp: {e}")


# ── Lotes ──────────────────────────────────────────────────────────────────────

async def crear_agendamiento(datos: dict) -> dict | None:
    """
    Inserta una cita en la tabla agendamientos de Supabase.

    Campos esperados en datos:
        lead_id, tipo_cita, fecha_visita, hora_llamada, resumen_conversacion,
        estado, asesor_id, asesor_asignado, video_url
    Retorna el registro creado con su id, o None si falla.
    """
    if not _crm_disponible():
        return None
    # Excluir columnas que no existen en la tabla
    datos_insert = {k: v for k, v in datos.items() if k != "video_url"}
    # Si asesor_id es None, excluirlo para no violar el FK
    if datos_insert.get("asesor_id") is None:
        datos_insert.pop("asesor_id", None)

    # fecha_visita requiere date nativo; hora_llamada es text en la tabla
    if isinstance(datos_insert.get("fecha_visita"), str):
        datos_insert["fecha_visita"] = date.fromisoformat(datos_insert["fecha_visita"])
    if not isinstance(datos_insert.get("hora_llamada"), str):
        t = datos_insert["hora_llamada"]
        datos_insert["hora_llamada"] = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)
    try:
        async with _crm_session() as session:
            cols = ", ".join(datos_insert.keys())
            vals = ", ".join(f":{k}" for k in datos_insert.keys())
            result = await session.execute(
                text(f"""
                    INSERT INTO agendamientos ({cols})
                    VALUES ({vals})
                    RETURNING *
                """),
                datos_insert,
            )
            await session.commit()
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM crear_agendamiento: {e}")
        return None


async def obtener_lotes_disponibles(proyecto_slug: str) -> list[dict]:
    """Retorna lotes con estado='disponible' de un proyecto."""
    if not _crm_disponible():
        return []
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("""
                    SELECT l.codigo, l.area_m2, l.precio_total,
                           l.separacion_inicial, l.cuotas_cantidad, l.cuota_valor, l.estado
                    FROM lotes l
                    INNER JOIN proyectos p ON p.id = l.proyecto_id
                    WHERE p.slug = :slug AND l.estado = 'disponible'
                    ORDER BY l.codigo
                """),
                {"slug": proyecto_slug},
            )
            return [dict(row) for row in result.mappings().all()]
    except Exception as e:
        logger.error(f"CRM obtener_lotes_disponibles: {e}")
        return []


# ── Asesores ───────────────────────────────────────────────────────────────────

async def obtener_asesor_por_nombre(nombre: str) -> dict | None:
    """
    Busca un asesor activo por nombre con matching flexible:
    verifica que todas las palabras del nombre del asesor aparezcan en el input.
    Ej: "Fabio Cardona" coincide con "Fabio Alonso Cardona".
    """
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT id, user_id, nombre, email, telefono FROM asesores WHERE activo = true")
            )
            asesores = [dict(row) for row in result.mappings().all()]

        nombre_lower = nombre.lower()
        for asesor in asesores:
            palabras = asesor["nombre"].lower().split()
            if all(p in nombre_lower for p in palabras):
                logger.info(f"CRM asesor encontrado: {asesor['nombre']} ({asesor['telefono']})")
                return asesor

        logger.info(f"CRM asesor no encontrado para nombre='{nombre}'")
        return None
    except Exception as e:
        logger.error(f"CRM obtener_asesor_por_nombre: {e}")
        return None


async def obtener_asesor_de_lead(telefono_cliente: str) -> dict | None:
    """
    Retorna el asesor asignado al lead del cliente dado.
    Flujo: lead(telefono) → asesor_responsable → asesores(nombre) → dict asesor.
    """
    lead = await obtener_lead(telefono_cliente)
    if not lead:
        logger.info(f"CRM obtener_asesor_de_lead: lead no encontrado para {telefono_cliente}")
        return None

    nombre_asesor: str = lead.get("asesor_responsable") or ""
    if not nombre_asesor:
        logger.info(f"CRM obtener_asesor_de_lead: lead {telefono_cliente} sin asesor_responsable")
        return None

    return await obtener_asesor_por_nombre(nombre_asesor)
