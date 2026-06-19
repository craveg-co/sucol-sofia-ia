import unittest

from agent.brain import (
    _corregir_disponibilidad,
    _construir_contexto_crm,
    _quitar_mencion_asesor_no_solicitada,
    _respuesta_operativa_visita,
    _sanitizar_historial,
    _validar_respuesta_oficial,
)


PROYECTO = {
    "slug": "santa_elena",
    "nombre": "Santa Elena",
    "direccion_visita": "Cra. 10 # 1-1 Villas del Parque Local 1",
    "google_maps_url": "https://maps.app.goo.gl/9tWjSGv824huNMoQA",
}


class DatosOficialesTest(unittest.TestCase):
    def test_contexto_inyecta_direccion_y_mapa_oficiales(self):
        contexto = _construir_contexto_crm(None, [], proyecto=PROYECTO)

        self.assertIn(PROYECTO["direccion_visita"], contexto)
        self.assertIn(PROYECTO["google_maps_url"], contexto)
        self.assertIn("Nunca completes ni deduzcas una dirección", contexto)

    def test_historial_descarta_direccion_inventada(self):
        historial = [
            {"role": "user", "content": "¿Dónde queda?"},
            {
                "role": "assistant",
                "content": "Carrera 10 #13-39, oficina 302 en Jamundí.",
            },
            {"role": "user", "content": "Gracias"},
        ]

        limpio = _sanitizar_historial(historial, PROYECTO)

        self.assertEqual(len(limpio), 2)
        self.assertTrue(
            all("13-39" not in mensaje["content"] for mensaje in limpio)
        )

    def test_respuesta_con_direccion_inventada_se_reemplaza(self):
        respuesta = _validar_respuesta_oficial(
            "Estamos en la Carrera 10 #13-39, oficina 302.",
            PROYECTO,
        )

        self.assertIn(PROYECTO["direccion_visita"], respuesta)
        self.assertIn(PROYECTO["google_maps_url"], respuesta)
        self.assertNotIn("13-39", respuesta)

    def test_respuesta_sin_direccion_no_se_modifica(self):
        respuesta = "Santa Elena está en Jamundí. ¿Quieres agendar una visita?"

        self.assertEqual(
            _validar_respuesta_oficial(respuesta, PROYECTO),
            respuesta,
        )

    def test_proyecto_sin_direccion_no_permite_inventarla(self):
        respuesta = _validar_respuesta_oficial(
            "Visítanos en la Calle 1 #2-3.",
            {"nombre": "Cascata", "direccion_visita": None},
        )

        self.assertIn(
            "No tengo una dirección de visita oficial registrada",
            respuesta,
        )
        self.assertNotIn("Calle 1", respuesta)

    def test_cascata_aplica_protocolo_virtual_antes_de_visita(self):
        respuesta = _respuesta_operativa_visita(
            "¿A dónde tengo que ir para una visita?",
            {
                "slug": "cascata",
                "nombre": "Cascata",
                "direccion_visita": None,
            },
        )

        self.assertIn("https://cascata360.sucol.co", respuesta)
        self.assertIn("autorización de la Dirección Comercial", respuesta)
        self.assertNotIn("asesora", respuesta.lower())

    def test_proyecto_con_direccion_responde_dato_oficial_sin_modelo(self):
        respuesta = _respuesta_operativa_visita(
            "¿Dónde queda el punto para visitar?",
            PROYECTO,
        )

        self.assertIn(PROYECTO["direccion_visita"], respuesta)
        self.assertIn(PROYECTO["google_maps_url"], respuesta)

    def test_elimina_asesora_y_telefono_si_cliente_no_los_pidio(self):
        respuesta = (
            "Santa Elena tiene opciones disponibles.\n\n"
            "Tu asesora Juliana Duque (+573170402005) puede contarte más. "
            "¿Quieres que te conecte con ella?"
        )

        limpia = _quitar_mencion_asesor_no_solicitada(
            respuesta,
            "Cuéntame de Santa Elena",
            {"nombre": "Juliana Duque", "telefono": "+573170402005"},
        )

        self.assertIn("Santa Elena tiene opciones disponibles", limpia)
        self.assertNotIn("Juliana", limpia)
        self.assertNotIn("3170402005", limpia)
        self.assertNotIn("asesora", limpia.lower())

    def test_conserva_contacto_si_cliente_pide_asesora(self):
        respuesta = "Tu asesora Juliana Duque es +573170402005."

        limpia = _quitar_mencion_asesor_no_solicitada(
            respuesta,
            "Dame el WhatsApp de mi asesora",
            {"nombre": "Juliana Duque", "telefono": "+573170402005"},
        )

        self.assertEqual(limpia, respuesta)

    def test_corrige_falsa_indisponibilidad_con_lotes_crm(self):
        respuesta = _corregir_disponibilidad(
            "En este momento no contamos con unidades disponibles.",
            [
                {"area_m2": 48, "precio_total": 39142580},
                {"area_m2": 72, "precio_total": 53884584},
            ],
            PROYECTO,
        )

        self.assertIn("Sí tenemos opciones disponibles", respuesta)
        self.assertIn("48", respuesta)
        self.assertIn("72", respuesta)
        self.assertIn("agendar una visita", respuesta)


if __name__ == "__main__":
    unittest.main()
