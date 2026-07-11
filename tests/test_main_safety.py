import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.main import (
    _bloquea_primer_contacto_por_contacto_activo,
    _bloquea_segundo_contacto_por_contacto_activo,
    _es_mensaje_formulario_meta,
)


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


if __name__ == "__main__":
    unittest.main()
