# agent/main.py — Servidor FastAPI + Webhook de WhatsApp
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Servidor principal de Sofía.
Detecta el proyecto del cliente en 4 pasos: contactos_whatsapp → leads → mensaje → genérico.
"""

import asyncio
import os
import time
import logging
import unicodedata
from datetime import datetime, timezone
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agent.brain import (
    clasificar_descarte_sofia,
    generar_respuesta_con_tools,
    separar_mensajes_whatsapp,
    _mensaje_error,
)
from agent.memory import inicializar_db, guardar_mensaje, obtener_historial
from agent.providers import obtener_proveedor
from agent.crm import (
    obtener_proyecto_por_telefono,
    obtener_proyecto_desde_lead,
    obtener_contacto_whatsapp,
    detectar_proyecto_en_mensaje,
    obtener_lead,
    obtener_lead_por_id,
    obtener_lotes_disponibles,
    obtener_agendamientos_lead,
    obtener_asesor_de_lead,
    crear_o_actualizar_contacto_whatsapp,
    marcar_incontactable,
    marcar_sofia_lead_respondio,
    marcar_sofia_lead_segundo_contacto,
    descartar_lead_sofia,
)

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
log_level = logging.DEBUG if ENVIRONMENT == "development" else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger("agentkit")

proveedor = obtener_proveedor()
PORT = int(os.getenv("PORT", 8000))

# ── Deduplicación de mensajes ──────────────────────────────────────────────────
# Evita procesar el mismo mensaje_id dos veces (reintentos de Meta, message echo)
_ids_procesados: OrderedDict = OrderedDict()
_DEDUP_MAX = 1000   # máximo de IDs en memoria
_DEDUP_TTL = 300    # 5 minutos — alineado con la ventana de antigüedad

# Máximo de segundos que puede tener un mensaje para ser procesado.
# Meta reintenta webhooks hasta 7 días; ignoramos todo lo que tenga más de 5 minutos.
_MSG_MAX_EDAD_SEG = 300


def _normalizar_texto_formulario(texto: str) -> str:
    texto = (texto or "").lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return " ".join(texto.split())


def _es_mensaje_formulario_meta(texto: str) -> bool:
    """
    Detecta el texto automatico que Meta envia al crear un lead desde publicidad.
    Ese mensaje no es una respuesta conversacional del cliente; el CRM ya dispara
    la plantilla de apertura por /iniciar.
    """
    texto_limpio = (texto or "").strip()
    if not texto_limpio:
        return False

    texto_norm = _normalizar_texto_formulario(texto_limpio)
    tiene_intro_formulario = (
        "complete el formulario" in texto_norm
        or "completado el formulario" in texto_norm
        or "completado este formulario" in texto_norm
        or "completed the form" in texto_norm
    )
    campos_formulario = [
        "full name:",
        "phone number:",
        "email:",
    ]
    campos_detectados = sum(campo in texto_norm for campo in campos_formulario)
    if tiene_intro_formulario and campos_detectados >= 2:
        return True

    return bool(
        tiene_intro_formulario
        and texto_norm.startswith(("hola! ", "¡hola! ", "hola "))
        and (
            "mas informacion sobre tu negocio" in texto_norm
            or "mas informacion sobre vuestro negocio" in texto_norm
            or "more information about your business" in texto_norm
        )
    )


def _es_autorespuesta_no_disponible(texto: str) -> bool:
    """
    Detecta contestadores/autorespuestas gen?ricas de WhatsApp.

    No todos los mensajes con "no puedo responder" son autom?ticos. Por eso este
    detector exige se?ales combinadas de autorespuesta y descarta primero si hay
    intenci?n comercial expl?cita.
    """
    texto_norm = _normalizar_texto_formulario(texto)
    if not texto_norm:
        return False

    intenciones_comerciales = (
        "precio",
        "precios",
        "cuanto vale",
        "cu?nto vale",
        "valor",
        "cotizacion",
        "cotizaci?n",
        "lote",
        "lotes",
        "proyecto",
        "financiacion",
        "financiaci?n",
        "cuotas",
        "visita",
        "ubicacion",
        "ubicaci?n",
        "area",
        "?rea",
        "metros",
        "bora",
        "buenavista",
        "vientos",
        "ginebra",
        "santa elena",
        "cascata",
    )
    if any(intencion in texto_norm for intencion in intenciones_comerciales):
        return False

    grupos_senales = [
        (
            "no_disponible",
            (
                "no puedo responder",
                "no puedo contestar",
                "no puedo atender",
                "no estoy disponible",
                "estoy ocupado",
                "estoy ocupada",
                "fuera de horario",
                "ausente",
            ),
        ),
        (
            "dejar_mensaje",
            (
                "dejame tu mensaje",
                "deja tu mensaje",
                "dejanos tu mensaje",
                "deje su mensaje",
                "puedes dejarme un mensaje",
                "escribeme tu mensaje",
            ),
        ),
        (
            "contacto_posterior",
            (
                "me estare comunicando",
                "me comunicare",
                "te respondere",
                "te contestare",
                "lo mas pronto posible",
                "tan pronto pueda",
                "en cuanto pueda",
            ),
        ),
        (
            "saludo_automatico",
            (
                "gracias por comunicarte",
                "gracias por contactarte",
                "gracias por escribir",
                "gracias por llamar",
                "bienvenido",
                "bienvenida",
            ),
        ),
    ]

    puntaje = 0
    for _, patrones in grupos_senales:
        if any(patron in texto_norm for patron in patrones):
            puntaje += 1

    cierres_genericos = (
        "dios te bendiga",
        "bendiciones",
        "te mando un abrazo",
        "feliz dia",
        "feliz tarde",
        "feliz noche",
    )
    if any(patron in texto_norm for patron in cierres_genericos):
        puntaje += 1

    return puntaje >= 2 and len(texto_norm) >= 45

def _bloquea_primer_contacto_por_contacto_activo(contacto: dict | None, proyecto_slug: str | None) -> bool:
    """
    Evita enviar una plantilla inicial si el telefono ya tiene otro proyecto activo.
    El cliente puede cambiar de proyecto explicitamente escribiendo el nombre del proyecto.
    """
    proyecto_activo = str((contacto or {}).get("proyecto_slug") or "").strip()
    return bool(proyecto_activo and proyecto_slug and proyecto_activo != proyecto_slug)


def _bloquea_segundo_contacto_por_contacto_activo(contacto: dict | None, proyecto_slug: str | None) -> bool:
    """El segundo contacto solo debe continuar sobre el mismo proyecto activo del chat."""
    proyecto_activo = str((contacto or {}).get("proyecto_slug") or "").strip()
    return bool(proyecto_activo and proyecto_slug and proyecto_activo != proyecto_slug)


def _ya_procesado(mensaje_id: str) -> bool:
    """True si este mensaje_id ya fue procesado en los últimos 5 minutos. Registra si es nuevo."""
    if not mensaje_id:
        return False
    ahora = time.monotonic()
    if mensaje_id in _ids_procesados:
        if ahora - _ids_procesados[mensaje_id] < _DEDUP_TTL:
            return True
        del _ids_procesados[mensaje_id]
    while len(_ids_procesados) >= _DEDUP_MAX:
        _ids_procesados.popitem(last=False)
    _ids_procesados[mensaje_id] = ahora
    return False


def _mensaje_muy_antiguo(timestamp: int) -> bool:
    """True si el mensaje tiene más de 5 minutos. Descarta reintentos de Meta tras reinicio."""
    if not timestamp:
        return False  # sin timestamp no podemos saber, dejamos pasar
    edad = datetime.now(timezone.utc).timestamp() - timestamp
    return edad > _MSG_MAX_EDAD_SEG


@asynccontextmanager
async def lifespan(app: FastAPI):
    await inicializar_db()
    logger.info(f"Sofía lista en puerto {PORT} — proveedor: {proveedor.__class__.__name__}")
    yield


app = FastAPI(
    title="Sofía — Agente WhatsApp de Sucol Soluciones Urbanísticas",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check():
    return {"status": "ok", "agente": "Sofía", "negocio": "Sucol Soluciones Urbanísticas"}


@app.get("/debug/{telefono}")
async def debug_contexto(telefono: str):
    """
    Diagnóstico completo: muestra exactamente qué ve Sofía para un número.
    Usar con el número tal como llega de WhatsApp (sin +).
    Ejemplo: GET /debug/573217512428
    """
    from agent.crm import (
        obtener_lead, obtener_asesor_de_lead, obtener_lotes_disponibles,
        obtener_agendamientos_lead, _variantes_telefono,
    )
    from agent.crm import _crm_session
    from sqlalchemy import text as _text

    # 1. Normalizar teléfono (igual que el webhook)
    tel_norm = "+" + telefono if not telefono.startswith("+") else telefono
    variantes = _variantes_telefono(tel_norm)

    # 2. Lead
    lead = await obtener_lead(tel_norm)

    # 3. Asesor
    asesor = await obtener_asesor_de_lead(tel_norm) if lead else None

    # 4. Proyecto y lotes
    proyecto = await _detectar_proyecto(tel_norm, "")
    proyecto_slug = proyecto.get("slug") if proyecto else None
    lotes = await obtener_lotes_disponibles(proyecto_slug) if proyecto_slug else []

    # 5. Agendamientos
    agendamientos = await obtener_agendamientos_lead(tel_norm) if lead else []

    # 6. Verificar lotes directamente en la tabla (sin filtro de estado)
    lotes_raw = []
    if proyecto_slug and _crm_session:
        try:
            async with _crm_session() as s:
                r = await s.execute(
                    _text("""
                        SELECT l.codigo, l.estado
                        FROM lotes l
                        JOIN proyectos p ON p.id = l.proyecto_id
                        WHERE p.slug = :slug
                        LIMIT 10
                    """),
                    {"slug": proyecto_slug},
                )
                lotes_raw = [dict(row) for row in r.mappings().all()]
        except Exception as e:
            lotes_raw = [{"error": str(e)}]

    return {
        "1_telefono_recibido": telefono,
        "2_telefono_normalizado": tel_norm,
        "3_variantes_buscadas": variantes,
        "4_lead": {
            "encontrado": bool(lead),
            "nombre": lead.get("nombre_completo") if lead else None,
            "asesor_responsable": lead.get("asesor_responsable") if lead else None,
            "proyecto": lead.get("proyecto") if lead else None,
            "etapa": lead.get("etapa_lead") if lead else None,
        },
        "5_asesor": {
            "encontrado": bool(asesor),
            "nombre": asesor.get("nombre") if asesor else None,
            "telefono": asesor.get("telefono") if asesor else None,
            "email": asesor.get("email") if asesor else None,
        },
        "6_proyecto": {
            "encontrado": bool(proyecto),
            "slug": proyecto_slug,
            "nombre": proyecto.get("nombre") if proyecto else None,
            "tiene_system_prompt": bool(proyecto.get("system_prompt")) if proyecto else False,
        },
        "7_lotes_disponibles": len(lotes),
        "8_lotes_en_bd_sin_filtro": lotes_raw,
        "9_agendamientos": len(agendamientos),
        "10_agendamientos_detalle": agendamientos,
    }


class LeadIdPayload(BaseModel):
    lead_id: str


@app.post("/iniciar")
async def iniciar_conversacion(payload: LeadIdPayload):
    """
    Llamado desde el CRM cuando se asigna un lead a Sofía.
    Busca el teléfono del lead y dispara el primer mensaje proactivo.
    """
    lead = await obtener_lead_por_id(payload.lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead no encontrado")

    telefono = lead.get("telefono_principal")
    if not telefono:
        raise HTTPException(status_code=422, detail="Lead sin teléfono")

    tel_norm = "+" + telefono if not telefono.startswith("+") else telefono

    proyecto = await obtener_proyecto_desde_lead(lead)
    proyecto_slug = proyecto.get("slug") if proyecto else None
    if not proyecto:
        raise HTTPException(status_code=422, detail="No se pudo detectar el proyecto del lead")

    contacto_activo = await obtener_contacto_whatsapp(tel_norm)
    if _bloquea_primer_contacto_por_contacto_activo(contacto_activo, proyecto_slug):
        logger.warning(
            "Primer contacto bloqueado para %s: proyecto activo=%s, lead=%s",
            tel_norm,
            contacto_activo.get("proyecto_slug"),
            proyecto_slug,
        )
        return {
            "status": "skipped",
            "reason": "telefono_con_proyecto_activo_distinto",
            "telefono": tel_norm,
            "proyecto_activo": contacto_activo.get("proyecto_slug"),
            "proyecto_lead": proyecto_slug,
        }

    try:
        await crear_o_actualizar_contacto_whatsapp(
            tel_norm,
            {"proyecto_slug": proyecto["slug"]},
        )
    except Exception as e:
        logger.warning(f"No se pudo actualizar proyecto activo para {tel_norm}: {e}")
    nombre_proyecto = proyecto.get("nombre") or proyecto_slug
    if not nombre_proyecto:
        raise HTTPException(status_code=422, detail="Proyecto sin nombre para plantilla")
    enviar_plantilla = getattr(proveedor, "enviar_plantilla_primer_contacto_sofia", None)
    if not enviar_plantilla:
        raise HTTPException(status_code=500, detail="Proveedor no soporta plantilla de primer contacto")
    try:
        enviado = await enviar_plantilla(telefono, nombre_proyecto)
    except Exception as e:
        logger.error(f"Error enviando plantilla inicial a {tel_norm}: {e}")
        raise HTTPException(status_code=502, detail="Error enviando plantilla por WhatsApp")
    if not enviado:
        raise HTTPException(status_code=502, detail="Meta rechazo el envio de la plantilla")
    mensaje_plantilla = (
        f"Hola, soy *Sofia*, la asistente virtual de *{nombre_proyecto}* "
        "de SUCOL Soluciones Urbanisticas.\n\n"
        "Vi que tuviste interes en nuestro proyecto y estoy aqui para ayudarte "
        "con toda la informacion.\n\n"
        "¿Te puedo contar mas sobre este proyecto?"
    )
    try:
        await guardar_mensaje(tel_norm, "assistant", mensaje_plantilla)
    except Exception as e:
        logger.warning(f"Error guardando plantilla inicial en memoria: {e}")
    logger.info(f"Plantilla inicial enviada a {tel_norm} - proyecto={proyecto_slug}")
    return {
        "status": "ok",
        "telefono": tel_norm,
        "proyecto": proyecto_slug,
        "plantilla": "sofia_primer_contacto_proyecto",
    }


@app.post("/segundo-contacto")
async def enviar_segundo_contacto(payload: LeadIdPayload):
    """
    Llamado desde el CRM cuando pasan 48h sin respuesta al primer contacto.
    Envia la plantilla aprobada de recordatorio y registra el evento.
    """
    lead = await obtener_lead_por_id(payload.lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead no encontrado")

    telefono = lead.get("telefono_principal")
    if not telefono:
        raise HTTPException(status_code=422, detail="Lead sin telefono")

    tel_norm = "+" + telefono if not telefono.startswith("+") else telefono

    proyecto = await obtener_proyecto_desde_lead(lead)
    proyecto_slug = proyecto.get("slug") if proyecto else None
    if not proyecto:
        raise HTTPException(status_code=422, detail="No se pudo detectar el proyecto del lead")

    contacto_activo = await obtener_contacto_whatsapp(tel_norm)
    if _bloquea_segundo_contacto_por_contacto_activo(contacto_activo, proyecto_slug):
        logger.warning(
            "Segundo contacto bloqueado para %s: proyecto activo=%s, lead=%s",
            tel_norm,
            contacto_activo.get("proyecto_slug"),
            proyecto_slug,
        )
        return {
            "status": "skipped",
            "reason": "telefono_con_proyecto_activo_distinto",
            "telefono": tel_norm,
            "proyecto_activo": contacto_activo.get("proyecto_slug"),
            "proyecto_lead": proyecto_slug,
        }

    try:
        await crear_o_actualizar_contacto_whatsapp(
            tel_norm,
            {"proyecto_slug": proyecto["slug"]},
        )
    except Exception as e:
        logger.warning(f"No se pudo actualizar proyecto activo para {tel_norm}: {e}")

    nombre_proyecto = proyecto.get("nombre") or proyecto_slug
    if not nombre_proyecto:
        raise HTTPException(status_code=422, detail="Proyecto sin nombre para plantilla")

    nombre_cliente = (
        lead.get("nombre_completo")
        or lead.get("nombre")
        or lead.get("primer_nombre")
        or ""
    ).strip()
    nombre_cliente = nombre_cliente.split()[0] if nombre_cliente else "de nuevo"

    enviar_plantilla = getattr(proveedor, "enviar_plantilla_segundo_contacto_sofia", None)
    if not enviar_plantilla:
        raise HTTPException(status_code=500, detail="Proveedor no soporta plantilla de segundo contacto")

    try:
        enviado = await enviar_plantilla(telefono, nombre_cliente, nombre_proyecto)
    except Exception as e:
        logger.error(f"Error enviando plantilla segundo contacto a {tel_norm}: {e}")
        raise HTTPException(status_code=502, detail="Error enviando plantilla por WhatsApp")

    if not enviado:
        raise HTTPException(status_code=502, detail="Meta rechazo el envio de la plantilla")

    mensaje_plantilla = (
        f"Hola {nombre_cliente} 👋 Soy Sofía de {nombre_proyecto}.\n"
        "Hace unos días te compartí información sobre nuestro proyecto y quería retomarte.\n"
        "Tenemos unidades disponibles y te puedo enviar en este momento precios actualizados, "
        "áreas y opciones de financiación directa sin banco.\n"
        "Una sola pregunta para orientarte mejor: ¿buscas algo para vivir, para invertir, o las dos cosas?"
    )
    try:
        await guardar_mensaje(tel_norm, "assistant", mensaje_plantilla)
    except Exception as e:
        logger.warning(f"Error guardando plantilla segundo contacto en memoria: {e}")

    try:
        await marcar_sofia_lead_segundo_contacto(payload.lead_id)
    except Exception as e:
        logger.warning(f"Error marcando segundo contacto para lead={payload.lead_id}: {e}")

    logger.info(f"Plantilla segundo contacto enviada a {tel_norm} - proyecto={proyecto_slug}")
    return {
        "status": "ok",
        "telefono": tel_norm,
        "proyecto": proyecto_slug,
        "plantilla": "sofia_segundo_contacto_proyecto",
    }


@app.post("/incontactable")
async def marcar_lead_incontactable(payload: LeadIdPayload):
    """
    Caso B del contrato Sofia ↔ leads-sucol.
    Llamado desde leads-sucol después de 48h sin respuesta o 3 intentos fallidos.
    Llama a pick-asesor, hace los UPDATEs necesarios en leads + sofia_leads
    y deja el lead listo para que trg_enqueue_e1b dispare (si hay asesor disponible).
    """
    lead = await obtener_lead_por_id(payload.lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead no encontrado")

    proyecto = lead.get("proyecto")  # slug o nombre tal como está en leads
    asignado = await marcar_incontactable(payload.lead_id, proyecto)

    return {
        "status": "ok",
        "lead_id": payload.lead_id,
        "asesor_asignado": asignado,
        "nota": "trg_enqueue_e1b disparará en leads-sucol" if asignado else "sin asesores disponibles — pendiente asignación manual en kanban",
    }


@app.get("/webhook")
async def webhook_verificacion(request: Request):
    resultado = await proveedor.validar_webhook(request)
    if resultado is not None:
        return PlainTextResponse(str(resultado))
    return {"status": "ok"}


@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        mensajes = await proveedor.parsear_webhook(request)

        for msg in mensajes:
            if msg.es_propio or not msg.texto:
                continue

            if _mensaje_muy_antiguo(msg.timestamp):
                edad = int(datetime.now(timezone.utc).timestamp() - msg.timestamp)
                logger.warning(f"Mensaje descartado: tiene {edad}s de antigüedad (id={msg.mensaje_id})")
                continue

            if _ya_procesado(msg.mensaje_id):
                logger.info(f"Mensaje duplicado ignorado (id={msg.mensaje_id})")
                continue

            # Normalizar teléfono: el CRM guarda con '+' pero Meta envía sin '+'
            telefono = "+" + msg.telefono if not msg.telefono.startswith("+") else msg.telefono

            logger.info(f"Mensaje de {telefono}: {msg.texto}")

            if _es_mensaje_formulario_meta(msg.texto):
                logger.info(
                    "Mensaje automatico de formulario Meta ignorado para evitar "
                    f"doble primer contacto ({telefono}, id={msg.mensaje_id})"
                )
                continue

            if _es_autorespuesta_no_disponible(msg.texto):
                logger.info(
                    "Autorespuesta/no disponible ignorada; no se marca como respuesta real "
                    f"({telefono}, id={msg.mensaje_id})"
                )
                continue

            # ── Paso 1, 2, 3 en paralelo: historial + lead + asesor + agendamientos
            historial, lead, asesor, agendamientos = await _gather(
                obtener_historial(telefono),
                obtener_lead(telefono),
                obtener_asesor_de_lead(telefono),
                obtener_agendamientos_lead(telefono),
            )

            # Respuesta del lead -> marcar respondio en sofia_leads si aplica.
            if lead and lead.get("id"):
                asyncio.create_task(marcar_sofia_lead_respondio(str(lead["id"])))

            # ── Paso 3: detectar proyecto (contactos_whatsapp → leads → mensaje)
            texto_deteccion = " ".join(
                parte for parte in (msg.texto, msg.referencia) if parte
            )
            proyecto = await _detectar_proyecto(telefono, texto_deteccion)

            # ── Paso 4: lotes del proyecto (requiere saber el proyecto)
            proyecto_slug = proyecto.get("slug") if proyecto else None
            lotes = await _gather_uno(obtener_lotes_disponibles(proyecto_slug)) if proyecto_slug else []

            logger.info(
                f"Contexto {telefono} → proyecto={proyecto_slug or 'ninguno'} "
                f"| lead={'sí' if lead else 'no'} "
                f"| lotes={len(lotes)}"
            )

            # ── Generar respuesta con soporte de tool_use (confirmar_cita)
            try:
                respuesta = await generar_respuesta_con_tools(
                    mensaje=msg.texto,
                    historial=historial,
                    contexto_lead=lead,
                    lotes_disponibles=lotes,
                    telefono=telefono,
                    asesor=asesor,
                    agendamientos=agendamientos,
                    proyecto_slug=proyecto_slug,
                    proyecto=proyecto,
                )
            except Exception as e:
                logger.error(f"Error generando respuesta para {telefono}: {e}")
                respuesta = _mensaje_error()

            # ── Guardar memoria y enviar (silenciosos si fallan)
            try:
                await guardar_mensaje(telefono, "user", msg.texto)
                await guardar_mensaje(telefono, "assistant", respuesta)
            except Exception as e:
                logger.error(f"Error guardando memoria para {telefono}: {e}")

            mensajes_salida = separar_mensajes_whatsapp(respuesta, proyecto)
            try:
                for indice, mensaje_salida in enumerate(mensajes_salida):
                    if indice:
                        await asyncio.sleep(0.6)
                    await proveedor.enviar_mensaje(msg.telefono, mensaje_salida)
            except Exception as e:
                logger.error(f"Error enviando mensaje a {telefono}: {e}")

            logger.info(
                f"Respuesta a {telefono} [{proyecto_slug or 'sin proyecto'}] "
                f"en {len(mensajes_salida)} mensaje(s): {respuesta[:80]}"
            )

            if lead and lead.get("id"):
                clasificacion_descarte = clasificar_descarte_sofia(msg.texto)
                if clasificacion_descarte:
                    await descartar_lead_sofia(
                        str(lead["id"]),
                        clasificacion_descarte,
                    )

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error en webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _gather(*coros):
    """
    Ejecuta coroutines en paralelo.
    Orden esperado: historial(list), lead(dict|None), asesor(dict|None), agendamientos(list)
    Las excepciones se loguean y se reemplazan por el valor vacío correcto para cada posición.
    """
    defaults = [[], None, None, []]
    resultados = await asyncio.gather(*coros, return_exceptions=True)
    limpios = []
    for i, r in enumerate(resultados):
        if isinstance(r, BaseException):
            default = defaults[i] if i < len(defaults) else None
            logger.error(f"_gather[{i}] falló: {r} — usando default={default!r}")
            limpios.append(default)
        else:
            limpios.append(r)
    return limpios


async def _gather_uno(coro):
    """Ejecuta una sola coroutine silenciando excepciones."""
    try:
        return await coro
    except Exception as e:
        logger.warning(f"Error silencioso en gather_uno: {e}")
        return []


async def _detectar_proyecto(telefono: str, mensaje: str) -> dict | None:
    """
    Detecta el proyecto con esta prioridad:
    1. Mención explícita en el mensaje actual.
    2. contactos_whatsapp → leads (contexto persistido).

    Una mención explícita siempre actualiza el proyecto activo del chat. Esto evita
    responder sobre Santa Elena usando el inventario de Cascata, o viceversa.
    """
    proyecto_mencionado = None
    if mensaje and mensaje.strip():
        try:
            proyecto_mencionado = await detectar_proyecto_en_mensaje(mensaje)
        except Exception as e:
            logger.error(f"Error detectando proyecto en mensaje: {e}")

    if proyecto_mencionado:
        logger.info(
            f"Proyecto explícito detectado para {telefono}: "
            f"{proyecto_mencionado.get('slug')}"
        )
        try:
            await crear_o_actualizar_contacto_whatsapp(
                telefono,
                {"proyecto_slug": proyecto_mencionado["slug"]},
            )
        except Exception as e:
            logger.warning(f"No se pudo actualizar proyecto activo para {telefono}: {e}")
        return proyecto_mencionado

    try:
        return await obtener_proyecto_por_telefono(telefono)
    except Exception as e:
        logger.error(f"Error buscando proyecto por teléfono: {e}")
        return None
