# agent/crm.py — Conexión de Sofía al CRM de Sucol

import os
import ssl
import logging
import httpx
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

def _variantes_telefono(telefono: str) -> list[str]:
    """
    Genera variantes del número para cubrir distintos formatos en el CRM.
    Ej: '573217512428' → también prueba '+573217512428' y '3217512428'
    """
    t = telefono.strip().replace(" ", "").replace("-", "")
    variantes = {t}
    if not t.startswith("+"):
        variantes.add("+" + t)
    # Colombia: quitar código de país 57 si el número tiene 12 dígitos
    if t.startswith("57") and len(t) == 12:
        variantes.add(t[2:])
    if t.startswith("+57") and len(t) == 13:
        variantes.add(t[3:])
    return list(variantes)


async def obtener_lead_por_id(lead_id: str) -> dict | None:
    """Retorna todos los datos del lead por su ID (UUID o entero)."""
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT * FROM leads WHERE id = :lead_id LIMIT 1"),
                {"lead_id": lead_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM obtener_lead_por_id: {e}")
        return None


async def obtener_lead(telefono: str) -> dict | None:
    """Retorna todos los datos del lead desde la tabla leads.
    Intenta múltiples formatos del teléfono para cubrir variantes del CRM."""
    if not _crm_disponible():
        return None
    variantes = _variantes_telefono(telefono)
    try:
        async with _crm_session() as session:
            for variante in variantes:
                result = await session.execute(
                    text("SELECT * FROM leads WHERE telefono_principal = :tel LIMIT 1"),
                    {"tel": variante},
                )
                row = result.mappings().first()
                if row:
                    logger.info(f"CRM obtener_lead: encontrado con variante '{variante}'")
                    return dict(row)
        logger.info(f"CRM obtener_lead: sin resultado para variantes {variantes}")
        return None
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


async def actualizar_lead_por_id(lead_id: str, datos: dict):
    """Actualiza campos del lead por su PK. Más preciso que por teléfono."""
    if not _crm_disponible() or not datos:
        return
    try:
        sets = ", ".join(f"{k} = :{k}" for k in datos)
        params = {"lead_id": lead_id, **datos}
        async with _crm_session() as session:
            await session.execute(
                text(f"UPDATE leads SET {sets} WHERE id = :lead_id"),
                params,
            )
            await session.commit()
        logger.info(f"CRM actualizar_lead_por_id {lead_id}: {list(datos.keys())}")
    except Exception as e:
        logger.error(f"CRM actualizar_lead_por_id: {e}")


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


async def crear_lead(datos: dict) -> dict | None:
    """
    Inserta un nuevo lead en la tabla leads con etapa_lead='calificado'.
    El trigger trigger_asignar_asesor_calificado asigna el asesor automáticamente.
    Retorna el row completo (con asesor ya asignado) o None si falla.
    """
    if not _crm_disponible():
        return None
    campos_base = {
        "etapa_lead": "calificado",
        "canal": "WhatsApp Sofia",
        "origen_creacion": "sofia",
    }
    datos_insert = {**campos_base, **datos}
    cols = ", ".join(datos_insert.keys())
    vals = ", ".join(f":{k}" for k in datos_insert.keys())
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text(f"""
                    INSERT INTO leads ({cols})
                    VALUES ({vals})
                    RETURNING *
                """),
                datos_insert,
            )
            await session.commit()
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM crear_lead: {e}")
        return None


async def crear_o_actualizar_sofia_lead(lead_id: str, proyecto_id: str | None, etapa: str) -> dict | None:
    """
    Inserta o actualiza el registro en sofia_leads.
    Al insertar con etapa='calificado', el trigger asigna el asesor automáticamente.
    """
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO sofia_leads (lead_id, proyecto_id, etapa)
                    VALUES (:lead_id, :proyecto_id, :etapa)
                    ON CONFLICT (lead_id)
                    DO UPDATE SET etapa = EXCLUDED.etapa, updated_at = now()
                    RETURNING *
                """),
                {"lead_id": lead_id, "proyecto_id": proyecto_id, "etapa": etapa},
            )
            await session.commit()
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM crear_o_actualizar_sofia_lead: {e}")
        return None


async def obtener_asesor_por_user_id(user_id: str) -> dict | None:
    """Busca un asesor activo por su user_id (FK a profiles, usado en sofia_leads.asesor_asignado_id)."""
    if not _crm_disponible():
        return None
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("SELECT id, user_id, nombre, email, telefono FROM asesores WHERE user_id = :uid AND activo = true LIMIT 1"),
                {"uid": user_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"CRM obtener_asesor_por_user_id: {e}")
        return None


async def obtener_agendamientos_lead(telefono: str) -> list[dict]:
    """Retorna las últimas 5 citas agendadas para el lead de este teléfono."""
    lead = await obtener_lead(telefono)
    if not lead:
        return []
    lead_id = lead.get("id")
    if not lead_id:
        return []
    try:
        async with _crm_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, tipo_cita, fecha_visita, hora_llamada, estado, asesor_asignado
                    FROM agendamientos
                    WHERE lead_id = :lead_id
                    ORDER BY fecha_visita DESC, hora_llamada DESC
                    LIMIT 5
                """),
                {"lead_id": lead_id},
            )
            return [dict(row) for row in result.mappings().all()]
    except Exception as e:
        logger.error(f"CRM obtener_agendamientos_lead: {e}")
        return []


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
                    WHERE p.slug = :slug AND LOWER(l.estado) = 'disponible'
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
    Retorna el asesor comercial asignado al lead del cliente.
    Prioridad:
    1. leads.asesor_responsable (asesor comercial real, asignado manualmente)
    2. sofia_leads.asesor_asignado_id como fallback (asignado por trigger)
    """
    lead = await obtener_lead(telefono_cliente)
    if not lead:
        logger.info(f"CRM obtener_asesor_de_lead: lead no encontrado para {telefono_cliente}")
        return None

    # 1. Asesor comercial del lead (fuente de verdad para la conversación)
    nombre_asesor: str = lead.get("asesor_responsable") or ""
    if nombre_asesor:
        asesor = await obtener_asesor_por_nombre(nombre_asesor)
        if asesor:
            logger.info(f"CRM asesor encontrado via leads.asesor_responsable: {asesor['nombre']}")
            return asesor

    # 2. Fallback: FK directo en sofia_leads (asignado por trigger)
    lead_id = lead.get("id")
    if lead_id and _crm_disponible():
        try:
            async with _crm_session() as session:
                result = await session.execute(
                    text("""
                        SELECT asesor_asignado_id
                        FROM sofia_leads
                        WHERE lead_id = :lead_id
                        LIMIT 1
                    """),
                    {"lead_id": str(lead_id)},
                )
                row = result.mappings().first()
                if row and row.get("asesor_asignado_id"):
                    asesor = await obtener_asesor_por_user_id(str(row["asesor_asignado_id"]))
                    if asesor:
                        logger.info(f"CRM asesor encontrado via sofia_leads FK: {asesor['nombre']}")
                        return asesor
        except Exception as e:
            logger.warning(f"CRM obtener_asesor_de_lead via sofia_leads: {e}")

    logger.info(f"CRM obtener_asesor_de_lead: sin asesor para {telefono_cliente}")
    return None


# ── Caso B: incontactable ──────────────────────────────────────────────────────

async def pick_asesor(lead_id: str, proyecto: str | None) -> dict:
    """
    Llama al edge function de leads-sucol para obtener el asesor disponible.
    Retorna { asesor_id: str|None, asesor_nombre: str|None }.
    """
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not supabase_key:
        logger.warning("pick_asesor: SUPABASE_URL / SUPABASE_SERVICE_KEY no configurados")
        return {"asesor_id": None, "asesor_nombre": None}
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(
                f"{supabase_url}/functions/v1/pick-asesor",
                headers={
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                },
                json={"lead_id": lead_id, "proyecto": proyecto},
            )
            r.raise_for_status()
            data = r.json()
            logger.info(f"pick_asesor lead={lead_id}: {data}")
            return data
    except Exception as e:
        logger.error(f"pick_asesor: {e}")
        return {"asesor_id": None, "asesor_nombre": None}


async def marcar_incontactable(lead_id: str, proyecto: str | None) -> bool:
    """
    Caso B del contrato Sofia ↔ leads-sucol: 48h sin respuesta o 3 intentos fallidos.

    1. Llama a pick-asesor para obtener el asesor disponible.
    2a. Si hay asesor → UPDATE leads (asesor_id + etapa_lead) → dispara trg_enqueue_e1b.
    2b. Si no hay asesor → no toca leads, queda para asignación manual en el kanban.
    3. Siempre → UPDATE sofia_leads SET etapa='incontactable'.

    Retorna True si el UPDATE de leads se realizó, False si quedó pendiente de asignación manual.
    """
    if not _crm_disponible():
        return False

    data = await pick_asesor(lead_id, proyecto)
    asesor_id = data.get("asesor_id")
    asesor_nombre = data.get("asesor_nombre")

    asignado = False
    if asesor_id:
        await actualizar_lead_por_id(lead_id, {
            "asesor_id": asesor_id,
            "asesor_responsable": asesor_nombre,
            "etapa_lead": "LEAD NUEVO ASIGNADO",
        })
        logger.info(f"marcar_incontactable lead={lead_id}: asesor asignado={asesor_nombre} → trg_enqueue_e1b disparará")
        asignado = True
    else:
        logger.info(f"marcar_incontactable lead={lead_id}: sin asesores disponibles — pendiente asignación manual")

    # Siempre actualizar sofia_leads sin importar si se asignó asesor
    await _actualizar_sofia_leads_etapa(lead_id, "incontactable")
    return asignado


async def marcar_sofia_lead_respondio(lead_id: str):
    """
    Marca en sofia_leads que el lead respondió por primera vez.
    Solo actualiza si la etapa es 'primer_contacto' o 'segundo_contacto'
    para no sobreescribir estados más avanzados.
    """
    await _actualizar_sofia_leads_etapa(
        lead_id, "respondio",
        solo_si_etapa_en=("primer_contacto", "segundo_contacto"),
        campos_extra={"fecha_respuesta": "NOW()"},
    )


async def _actualizar_sofia_leads_etapa(
    lead_id: str,
    etapa: str,
    solo_si_etapa_en: tuple | None = None,
    campos_extra: dict | None = None,
):
    """Helper interno: UPDATE sofia_leads SET etapa + campos opcionales."""
    if not _crm_disponible():
        return
    try:
        where = "lead_id = :lead_id"
        params: dict = {"lead_id": lead_id, "etapa": etapa}

        sets = "etapa = :etapa, updated_at = now()"
        if campos_extra:
            for k, v in campos_extra.items():
                if v == "NOW()":
                    sets += f", {k} = now()"
                else:
                    sets += f", {k} = :{k}"
                    params[k] = v

        if solo_si_etapa_en:
            placeholders = ", ".join(f":e{i}" for i in range(len(solo_si_etapa_en)))
            where += f" AND etapa IN ({placeholders})"
            for i, e in enumerate(solo_si_etapa_en):
                params[f"e{i}"] = e

        async with _crm_session() as session:
            await session.execute(
                text(f"UPDATE sofia_leads SET {sets} WHERE {where}"),
                params,
            )
            await session.commit()
        logger.info(f"sofia_leads lead={lead_id} → etapa={etapa}")
    except Exception as e:
        logger.error(f"_actualizar_sofia_leads_etapa lead={lead_id} etapa={etapa}: {e}")
