# agent/crm.py — Conexión de Sofía al CRM de Sucol

import os
import ssl
import logging
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
        for p in proyectos:
            slug_legible = p["slug"].replace("_", " ").replace("-", " ")
            nombre_lower = p["nombre"].lower()
            if slug_legible in mensaje_lower or nombre_lower in mensaje_lower:
                return await obtener_proyecto_por_slug(p["slug"])
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
