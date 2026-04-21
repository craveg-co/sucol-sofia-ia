# agent/crm.py — Conexión de Sofía al CRM de Sucol
# Lee proyectos, leads y lotes desde el Supabase del CRM (solo lectura excepto contactos y leads)

import os
import ssl
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

# ── Conexión al CRM ────────────────────────────────────────────────────────────

_CRM_URL = os.getenv("CRM_DATABASE_URL", "")

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
    _crm_engine = create_async_engine(
        _CRM_URL,
        echo=False,
        connect_args={"ssl": _ssl_ctx},
    )
    _crm_session = async_sessionmaker(_crm_engine, class_=AsyncSession, expire_on_commit=False)


def _crm_disponible() -> bool:
    return _crm_session is not None


# ── Proyectos ──────────────────────────────────────────────────────────────────

async def obtener_proyecto_por_slug(slug: str) -> dict | None:
    """Lee un proyecto y su system_prompt de la tabla proyectos."""
    if not _crm_disponible():
        logger.warning("CRM_DATABASE_URL no configurada")
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
    """Busca el lead por telefono_principal y retorna el proyecto asociado con su system_prompt."""
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("""
                    SELECT p.*
                    FROM proyectos p
                    INNER JOIN leads l ON l.proyecto = p.slug
                    WHERE l.telefono_principal = :telefono
                      AND p.activo = true
                    LIMIT 1
                """),
                {"telefono": telefono},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM obtener_proyecto_por_telefono: {e}")
        return None


async def obtener_proyectos_activos() -> list[dict]:
    """Retorna todos los proyectos activos (para el prompt genérico de bienvenida)."""
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
    """Upsert en tabla contactos_whatsapp con telefono como clave."""
    if not _crm_disponible():
        return
    try:
        sets = ", ".join(f"{k} = EXCLUDED.{k}" for k in datos if k != "telefono")
        cols = ", ".join(["telefono"] + [k for k in datos])
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
