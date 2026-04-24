# agent/providers/meta.py — Adaptador para Meta WhatsApp Cloud API
# Generado por AgentKit

import os
import logging
import httpx
from fastapi import Request
from agent.providers.base import ProveedorWhatsApp, MensajeEntrante

logger = logging.getLogger("agentkit")


class ProveedorMeta(ProveedorWhatsApp):
    """Proveedor de WhatsApp usando la API oficial de Meta (Cloud API)."""

    def __init__(self):
        self.access_token = os.getenv("META_ACCESS_TOKEN")
        self.phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.verify_token = os.getenv("META_VERIFY_TOKEN", "sucol-sofia")
        self.api_version = "v21.0"

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
                    if msg.get("type") != "text":
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
                        texto=msg.get("text", {}).get("body", ""),
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
        video_url: str,
    ) -> bool:
        """Envía la plantilla sucol_cita_virtual_asesor al asesor asignado."""
        if not self.access_token or not self.phone_number_id:
            logger.warning("META_ACCESS_TOKEN o META_PHONE_NUMBER_ID no configurados — plantilla no enviada")
            return False
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
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
            video_url if video_url else "No aplica",
        ]
        payload = {
            "messaging_product": "whatsapp",
            "to": telefono_asesor,
            "type": "template",
            "template": {
                "name": "sucol_cita_virtual_asesor",
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
            return r.status_code == 200
