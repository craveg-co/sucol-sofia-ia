import os
import sys
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.main import (
    LeadIdPayload,
    _bloquea_primer_contacto_por_contacto_activo,
    _bloquea_segundo_contacto_por_contacto_activo,
    _es_mensaje_formulario_meta,
    _resolver_proyecto_pendiente,
)


class TestPayloadIniciar(unittest.TestCase):
    def test_acepta_perfil_en_raiz(self):
        payload = LeadIdPayload(
            id="lead-123",
            telefono="573001112233",
            project="buenavista",
            nombre="Ana",
        )

        self.assertEqual(payload.id_lead(), "lead-123")
        self.assertEqual(payload.perfil_lead(), {
            "id": "lead-123",
            "telefono_principal": "573001112233",
            "proyecto_id": None,
            "proyecto": "buenavista",
            "nombre_completo": "Ana",
        })

    def test_acepta_record_de_webhook_supabase(self):
        payload = LeadIdPayload(record={
            "id": "lead-456",
            "telefono_principal": "+573004445566",
            "proyecto_id": "proyecto-1",
        })

        self.assertEqual(payload.id_lead(), "lead-456")
        self.assertEqual(payload.perfil_lead()["telefono_principal"], "+573004445566")
        self.assertEqual(payload.perfil_lead()["proyecto_id"], "proyecto-1")


class TestMainSafety(unittest.TestCase):
    def test_detecta_formulario_meta_con_texto_real(self):
        texto = (
            "¡Hola! He completado el formulario y me gustaría recibir más información "
            "sobre vuestro negocio.\n\n"
            "Full name: Iles Hoyos\n"
            "Phone number: +573026789910\n"
            "Email: iles0629@gmail.com"
        )

        self.assertTrue(_es_mensaje_formulario_meta(texto))

    def test_no_detecta_mensaje_normal_como_formulario(self):
        self.assertFalse(_es_mensaje_formulario_meta("Sí, cuéntame más"))

    def test_detecta_formulario_meta_recortado(self):
        texto = (
            "¡Hola! Completé el formulario y me gustaría obtener más información "
            "sobre tu negocio."
        )

        self.assertTrue(_es_mensaje_formulario_meta(texto))

    def test_bloquea_primer_contacto_si_hay_otro_proyecto_activo(self):
        contacto = {"proyecto_slug": "santa_elena"}

        self.assertTrue(
            _bloquea_primer_contacto_por_contacto_activo(contacto, "buenavista")
        )

    def test_no_bloquea_primer_contacto_sin_contacto_activo(self):
        self.assertFalse(
            _bloquea_primer_contacto_por_contacto_activo(None, "santa_elena")
        )

    def test_no_bloquea_primer_contacto_del_mismo_proyecto(self):
        contacto = {"proyecto_slug": "santa_elena"}

        self.assertFalse(
            _bloquea_primer_contacto_por_contacto_activo(contacto, "santa_elena")
        )

    def test_bloquea_segundo_contacto_si_hay_otro_proyecto_activo(self):
        contacto = {"proyecto_slug": "santa_elena"}

        self.assertTrue(
            _bloquea_segundo_contacto_por_contacto_activo(contacto, "buenavista")
        )


class TestResolverProyectoPendiente(unittest.IsolatedAsyncioTestCase):
    """Caso Andrés De La Espriella: lead nuevo para otro proyecto mientras ya
    había una conversación activa. Sofía debe preguntar, no cambiar en silencio
    ni quedarse pegada al proyecto viejo."""

    async def test_sin_marca_pendiente_no_hace_nada(self):
        with patch("agent.main.obtener_contacto_whatsapp", AsyncMock(return_value={"proyecto_slug": "vientos_ginebra"})):
            resultado = await _resolver_proyecto_pendiente("+573155620559", "hola")
        self.assertIsNone(resultado)

    async def test_mensaje_ambiguo_pregunta_por_los_dos_proyectos(self):
        contacto = {"proyecto_slug": "vientos_ginebra", "notas_sofia": "PENDIENTE_PROYECTO:reservas_ilama"}
        with patch("agent.main.obtener_contacto_whatsapp", AsyncMock(return_value=contacto)), \
             patch("agent.main.detectar_proyecto_en_mensaje", AsyncMock(return_value=None)), \
             patch("agent.main.obtener_proyecto_por_slug", AsyncMock(side_effect=lambda slug: {
                 "vientos_ginebra": {"slug": "vientos_ginebra", "nombre": "Vientos de Ginebra"},
                 "reservas_ilama": {"slug": "reservas_ilama", "nombre": "Reservas de Ilama"},
             }.get(slug))):
            pregunta = await _resolver_proyecto_pendiente("+573155620559", "Sí, cuéntame más")

        self.assertIsNotNone(pregunta)
        self.assertIn("Vientos de Ginebra", pregunta)
        self.assertIn("Reservas de Ilama", pregunta)

    async def test_cliente_elige_proyecto_limpia_la_marca(self):
        contacto = {"proyecto_slug": "vientos_ginebra", "notas_sofia": "PENDIENTE_PROYECTO:reservas_ilama"}
        mock_limpiar = AsyncMock()
        with patch("agent.main.obtener_contacto_whatsapp", AsyncMock(return_value=contacto)), \
             patch("agent.main.detectar_proyecto_en_mensaje", AsyncMock(return_value={"slug": "reservas_ilama"})), \
             patch("agent.main.crear_o_actualizar_contacto_whatsapp", mock_limpiar):
            resultado = await _resolver_proyecto_pendiente("+573155620559", "Reservas de Ilama por favor")

        self.assertIsNone(resultado)
        mock_limpiar.assert_awaited_once_with("+573155620559", {"notas_sofia": None})


if __name__ == "__main__":
    unittest.main()
