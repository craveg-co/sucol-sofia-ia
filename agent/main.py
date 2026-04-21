# agent/main.py — Servidor FastAPI + Webhook de WhatsApp
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Servidor principal de Sofía.
Detecta el proyecto del cliente en 4 pasos: contactos_whatsapp → leads → mensaje → genérico.
"""

import asyncio
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
    detectar_proyecto_en_mensaje,
    obtener_lead,
    obtener_lotes_disponibles,
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

            logger.info(f"Mensaje de {msg.telefono}: {msg.texto}")

            # ── Paso 1 y 2 en paralelo: historial + lead (no dependen del proyecto)
            historial, lead = await _gather(
                obtener_historial(msg.telefono),
                obtener_lead(msg.telefono),
            )

            # ── Paso 3: detectar proyecto (contactos_whatsapp → leads → mensaje)
            proyecto = await _detectar_proyecto(msg.telefono, msg.texto)

            # ── Paso 4: lotes del proyecto (requiere saber el proyecto)
            proyecto_slug = proyecto.get("slug") if proyecto else None
            lotes = await _gather_uno(obtener_lotes_disponibles(proyecto_slug)) if proyecto_slug else []

            sistema_prompt = proyecto.get("system_prompt") if proyecto else None

            # ── Generar respuesta
            try:
                respuesta = await generar_respuesta(
                    mensaje=msg.texto,
                    historial=historial,
                    sistema_prompt=sistema_prompt,
                    contexto_lead=lead,
                    lotes_disponibles=lotes,
                )
            except Exception as e:
                logger.error(f"Error generando respuesta para {msg.telefono}: {e}")
                respuesta = "Hola, estoy teniendo un inconveniente técnico. Por favor intenta en unos minutos."

            # ── Guardar memoria y enviar (silenciosos si fallan)
            try:
                await guardar_mensaje(msg.telefono, "user", msg.texto)
                await guardar_mensaje(msg.telefono, "assistant", respuesta)
            except Exception as e:
                logger.error(f"Error guardando memoria para {msg.telefono}: {e}")

            try:
                await proveedor.enviar_mensaje(msg.telefono, respuesta)
            except Exception as e:
                logger.error(f"Error enviando mensaje a {msg.telefono}: {e}")

            logger.info(f"Respuesta a {msg.telefono} [{proyecto_slug or 'sin proyecto'}]: {respuesta[:80]}")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error en webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _gather(*coros):
    """Ejecuta coroutines en paralelo y convierte excepciones en valores vacíos."""
    resultados = await asyncio.gather(*coros, return_exceptions=True)
    limpios = []
    for r in resultados:
        if isinstance(r, BaseException):
            limpios.append(None if not isinstance(r, list) else [])
        else:
            limpios.append(r)
    # Corregir: historial debe ser lista, lead puede ser None
    if len(limpios) == 2:
        if limpios[0] is None:
            limpios[0] = []
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
    Detecta el proyecto en 3 pasos:
    1. contactos_whatsapp → leads  (obtener_proyecto_por_telefono ya hace ambos)
    2. Mención en el mensaje
    Si detecta por mensaje, lo guarda en contactos_whatsapp para la próxima vez.
    """
    # Pasos 1+2: contactos_whatsapp luego leads (con auto-registro si viene de leads)
    try:
        proyecto = await obtener_proyecto_por_telefono(telefono)
    except Exception as e:
        logger.error(f"Error buscando proyecto por teléfono: {e}")
        proyecto = None

    if proyecto:
        return proyecto

    # Paso 3: detectar por mención en el mensaje
    try:
        proyecto = await detectar_proyecto_en_mensaje(mensaje)
    except Exception as e:
        logger.error(f"Error detectando proyecto en mensaje: {e}")
        proyecto = None

    if proyecto:
        logger.info(f"Proyecto detectado por mensaje para {telefono}: {proyecto.get('slug')}")
        try:
            await crear_o_actualizar_contacto_whatsapp(
                telefono,
                {"proyecto_slug": proyecto["slug"], "etapa_chat": "prospecto"},
            )
        except Exception as e:
            logger.warning(f"No se pudo guardar proyecto detectado para {telefono}: {e}")

    return proyecto
