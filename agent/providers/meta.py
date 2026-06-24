# agent/providers/meta.py — Adaptador para Meta WhatsApp Cloud API
# Generado por AgentKit

import os
import logging
import httpx
import json
from datetime import datetime, timezone
from fastapi import Request
from agent.providers.base import ProveedorWhatsApp, MensajeEntrante

logger = logging.getLogger("agentkit")


class ProveedorMeta(ProveedorWhatsApp):
    """Proveedor de WhatsApp usando la API oficial de Meta (Cloud API)."""

    def __init__(self):
        self.access_token = os.getenv("META_ACCESS_TOKEN")
        self.phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.notify_phone_number_id = os.getenv("META_NOTIFY_PHONE_NUMBER_ID", self.phone_number_id)
        self.verify_token = os.getenv("META_VERIFY_TOKEN", "sucol-sofia")
        self.api_version = "v21.0"
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

    async def _guardar_advisor_survey(self, msg: dict) -> None:
        """Guarda respuestas de WhatsApp Flows en public.advisor_surveys."""
        if not self.supabase_url or not self.supabase_key:
            logger.warning("advisor_surveys: SUPABASE_URL / SUPABASE_SERVICE_KEY no configurados")
            return

        interactive = msg.get("interactive", {})
        nfm_reply = interactive.get("nfm_reply", {})
        response_raw = nfm_reply.get("response_json") or "{}"
        try:
            response_json = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
        except Exception:
            response_json = {"raw": response_raw}

        telefono = "+" + msg.get("from", "") if not msg.get("from", "").startswith("+") else msg.get("from", "")
        flow_token = response_json.get("flow_token") if isinstance(response_json, dict) else None
        flow_name = nfm_reply.get("name") or interactive.get("type") or "nfm_reply"
        answered_at = datetime.now(timezone.utc).isoformat()

        if not flow_token:
            logger.warning(f"advisor_surveys: respuesta Flow sin flow_token para {telefono}")
            return

        payload = {
            "telefono_cliente": telefono,
            "wamid": msg.get("id", ""),
            "flow_name": flow_name,
            "status": "answered",
            "answered_at": answered_at,
            "response_payload": {
                "response": response_json,
                "raw_message": msg,
            },
            "last_error": None,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                }
                resp = await client.patch(
                    f"{self.supabase_url}/rest/v1/advisor_surveys",
                    headers=headers,
                    params={"flow_token": f"eq.{flow_token}", "select": "id"},
                    json={k: v for k, v in payload.items() if v is not None},
                )
                if resp.status_code != 200:
                    logger.error(f"advisor_surveys update HTTP {resp.status_code}: {resp.text[:300]}")
                    return
                rows = resp.json()
                if not rows:
                    logger.warning(f"advisor_surveys: no existe registro previo para flow_token={flow_token}")
                    return
                logger.info(f"advisor_surveys: respuesta Flow guardada para {telefono}")
        except Exception as e:
            logger.error(f"advisor_surveys: error guardando respuesta Flow: {e}")

    async def validar_webhook(self, request: Request) -> dict | int | None:
        """Meta requiere verificación GET con hub.verify_token."""
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")
        if mode == "subscribe" and token == self.verify_token:
            # Meta espera el challenge como respuesta en texto plano
            return int(challenge)
        return None

    async def parsear_webhook(self, request: Request) -> list[MensajeEntrante]:
        """Parsea el payload anidado de Meta Cloud API."""
        body = await request.json()
        mensajes = []
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                # Ignorar notificaciones de estado (sent/delivered/read)
                if value.get("statuses"):
                    continue
                metadata = value.get("metadata", {})
                numero_bot = metadata.get("display_phone_number", "")
                phone_number_id = metadata.get("phone_number_id", "")
                for msg in value.get("messages", []):
                    msg_type = msg.get("type")
                    interactive = msg.get("interactive", {})

                    if msg_type == "interactive" and interactive.get("type") == "nfm_reply":
                        await self._guardar_advisor_survey(msg)
                        continue

                    texto = ""
                    if msg_type == "text":
                        texto = msg.get("text", {}).get("body", "")
                    elif msg_type == "button":
                        button = msg.get("button", {})
                        texto = button.get("text") or button.get("payload") or ""
                    elif msg_type == "interactive" and interactive.get("type") == "button_reply":
                        button_reply = interactive.get("button_reply", {})
                        texto = button_reply.get("title") or button_reply.get("id") or ""
                    elif msg_type == "interactive" and interactive.get("type") == "list_reply":
                        list_reply = interactive.get("list_reply", {})
                        texto = list_reply.get("title") or list_reply.get("id") or ""

                    if not texto:
                        continue
                    remitente = msg.get("from", "")
                    # Es propio si el remitente coincide con el número del bot
                    es_propio = bool(
                        remitente == numero_bot
                        or remitente == phone_number_id
                        or msg.get("from_me", False)
                    )
                    mensajes.append(MensajeEntrante(
                        telefono=remitente,
                        texto=texto,
                        mensaje_id=msg.get("id", ""),
                        es_propio=es_propio,
                        timestamp=int(msg.get("timestamp", 0) or 0),
                    ))
        return mensajes

    async def enviar_mensaje(self, telefono: str, mensaje: str) -> bool:
        """Envía mensaje via Meta WhatsApp Cloud API."""
        if not self.access_token or not self.phone_number_id:
            logger.warning("META_ACCESS_TOKEN o META_PHONE_NUMBER_ID no configurados")
            return False
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": telefono,
            "type": "text",
            "text": {"body": mensaje},
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Error Meta API: {r.status_code} — {r.text}")
            return r.status_code == 200

    async def enviar_plantilla_primer_contacto_sofia(
        self,
        telefono: str,
        nombre_proyecto: str,
    ) -> bool:
        """Envia la plantilla aprobada de primer contacto de Sofia al lead."""
        if not self.access_token or not self.phone_number_id:
            logger.warning("META_ACCESS_TOKEN o META_PHONE_NUMBER_ID no configurados")
            return False

        telefono = telefono.replace("+", "").replace(" ", "").replace("-", "")
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": telefono,
            "type": "template",
            "template": {
                "name": "sofia_primer_contacto_proyecto",
                "language": {"code": "es_CO"},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": nombre_proyecto},
                        ],
                    }
                ],
            },
        }

        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Error plantilla primer contacto Meta API: {r.status_code} - {r.text}")
                return False
            return True

    async def enviar_plantilla_cita_asesor(
        self,
        telefono_asesor: str,
        asesor_nombre: str,
        tipo_cita: str,
        fecha_cita: str,
        hora_cita: str,
        nombre_cliente: str,
        telefono_cliente: str,
        resumen_conversacion: str,
        video_url: str = "",
    ) -> bool:
        """
        Envía la plantilla sucol_nueva_cita_asesor al asesor desde el número de notificaciones.
        7 parámetros fijos — sin video_url para evitar errores por campo vacío.
        Si hay video_url, se envía como mensaje de texto adicional.
        """
        if not self.access_token or not self.notify_phone_number_id:
            logger.warning("META_ACCESS_TOKEN o META_NOTIFY_PHONE_NUMBER_ID no configurados — plantilla no enviada")
            return False

        # Meta API requiere el número sin '+' ni espacios
        telefono_asesor = telefono_asesor.replace("+", "").replace(" ", "").replace("-", "")

        url = f"https://graph.facebook.com/{self.api_version}/{self.notify_phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        parametros = [
            asesor_nombre,
            tipo_cita,
            fecha_cita,
            hora_cita,
            nombre_cliente,
            telefono_cliente,
            resumen_conversacion,
        ]
        payload = {
            "messaging_product": "whatsapp",
            "to": telefono_asesor,
            "type": "template",
            "template": {
                "name": "sucol_nueva_cita_asesor",
                "language": {"code": "es_CO"},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": str(p)} for p in parametros
                        ],
                    }
                ],
            },
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Error plantilla Meta API: {r.status_code} — {r.text}")
                return False

            # Si hay link de videollamada, enviarlo como mensaje de texto adicional
            if video_url:
                await client.post(url, headers=headers, json={
                    "messaging_product": "whatsapp",
                    "to": telefono_asesor,
                    "type": "text",
                    "text": {"body": f"🔗 Enlace de videollamada: {video_url}"},
                })

            return True
