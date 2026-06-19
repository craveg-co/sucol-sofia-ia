import unittest

from agent.brain import (
    _construir_contexto_crm,
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


if __name__ == "__main__":
    unittest.main()
