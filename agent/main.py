# agent/main.py — Servidor FastAPI + Webhook de WhatsApp
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Servidor principal de Sofía, la agente de WhatsApp de Sucol.
Enruta cada conversación al system prompt del proyecto correspondiente en el CRM.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from agent.brain import generar_respuesta
from agent.memory import inicializar_db, guardar_mensaje, obtener_historial
from agent.providers import obtener_proveedor
from agent.crm import (
    obtener_proyecto_por_telefono,
    obtener_lead,
    crear_o_actualizar_contacto_whatsapp,
)

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
log_level = logging.DEBUG if ENVIRONMENT == "development" else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger("agentkit")

proveedor = obtener_proveedor()
PORT = int(os.getenv("PORT", 8000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la base de datos de memoria al arrancar el servidor."""
    await inicializar_db()
    logger.info("Base de datos de memoria inicializada")
    logger.info(f"Servidor Sofía (Sucol) corriendo en puerto {PORT}")
    logger.info(f"Proveedor de WhatsApp: {proveedor.__class__.__name__}")
    yield


app = FastAPI(
    title="Sofía — Agente WhatsApp de Sucol Soluciones Urbanísticas",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check():
    return {"status": "ok", "agente": "Sofía", "negocio": "Sucol Soluciones Urbanísticas"}


@app.get("/webhook")
async def webhook_verificacion(request: Request):
    """Verificación GET del webhook — requerido por Meta Cloud API."""
    resultado = await proveedor.validar_webhook(request)
    if resultado is not None:
        return PlainTextResponse(str(resultado))
    return {"status": "ok"}


@app.post("/webhook")
async def webhook_handler(request: Request):
    """
    Recibe mensajes de WhatsApp via Meta Cloud API.
    Consulta el CRM para obtener el proyecto y datos del lead,
    genera respuesta personalizada con Claude y la envía de vuelta.
    """
    try:
        mensajes = await proveedor.parsear_webhook(request)

        for msg in mensajes:
            if msg.es_propio or not msg.texto:
                continue

            logger.info(f"Mensaje de {msg.telefono}: {msg.texto}")

            # Consultar CRM en paralelo para minimizar latencia
            historial, proyecto, lead = await _obtener_contexto(msg.telefono)

            # Extraer system_prompt del proyecto (None si no hay proyecto asignado)
            sistema_prompt = proyecto.get("system_prompt") if proyecto else None
            proyecto_slug = proyecto.get("slug") if proyecto else None

            # Generar respuesta con contexto completo
            respuesta = await generar_respuesta(
                mensaje=msg.texto,
                historial=historial,
                sistema_prompt=sistema_prompt,
                contexto_lead=lead,
            )

            # Guardar en memoria de conversación
            await guardar_mensaje(msg.telefono, "user", msg.texto)
            await guardar_mensaje(msg.telefono, "assistant", respuesta)

            # Registrar/actualizar contacto en CRM (no bloqueante si falla)
            await _registrar_contacto(msg.telefono, proyecto_slug, lead)

            # Enviar respuesta por WhatsApp
            await proveedor.enviar_mensaje(msg.telefono, respuesta)
            logger.info(f"Respuesta a {msg.telefono} [{proyecto_slug or 'sin proyecto'}]: {respuesta[:80]}")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error en webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _obtener_contexto(telefono: str):
    """Obtiene historial, proyecto y lead del CRM para un número de teléfono."""
    import asyncio
    historial, proyecto, lead = await asyncio.gather(
        obtener_historial(telefono),
        obtener_proyecto_por_telefono(telefono),
        obtener_lead(telefono),
        return_exceptions=True,
    )
    # Si alguna falla silenciosamente, retornar valor vacío
    if isinstance(historial, Exception):
        historial = []
    if isinstance(proyecto, Exception):
        proyecto = None
    if isinstance(lead, Exception):
        lead = None
    return historial, proyecto, lead


async def _registrar_contacto(telefono: str, proyecto_slug: str | None, lead: dict | None):
    """Actualiza contactos_whatsapp en el CRM sin bloquear la respuesta al cliente."""
    try:
        datos = {}
        if proyecto_slug:
            datos["proyecto_slug"] = proyecto_slug
        if lead and lead.get("id"):
            datos["lead_id"] = lead["id"]
        await crear_o_actualizar_contacto_whatsapp(telefono, datos)
    except Exception as e:
        logger.warning(f"No se pudo actualizar contacto CRM para {telefono}: {e}")
