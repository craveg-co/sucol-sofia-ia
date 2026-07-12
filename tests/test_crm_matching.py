import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.crm import (
    _normalizar_texto_matching,
    _proyecto_coincide_mensaje,
    _variantes_telefono,
)


class CrmMatchingTest(unittest.TestCase):
    def test_buenas_vista_coincide_con_buenavista(self):
        proyecto = {"slug": "buenavista", "nombre": "Buenavista"}
        mensaje = _normalizar_texto_matching("Buenas vista")

        self.assertTrue(_proyecto_coincide_mensaje(proyecto, mensaje))

    def test_buena_vista_coincide_con_buenavista(self):
        proyecto = {"slug": "buenavista", "nombre": "Buenavista"}
        mensaje = _normalizar_texto_matching("Me interesa Buena Vista")

        self.assertTrue(_proyecto_coincide_mensaje(proyecto, mensaje))

    def test_no_hace_match_con_texto_generico(self):
        proyecto = {"slug": "buenavista", "nombre": "Buenavista"}
        mensaje = _normalizar_texto_matching("Quiero información de Bora")

        self.assertFalse(_proyecto_coincide_mensaje(proyecto, mensaje))

    def test_variantes_telefono_colombia(self):
        variantes = set(_variantes_telefono("+573164261812"))

        self.assertIn("+573164261812", variantes)
        self.assertIn("3164261812", variantes)


if __name__ == "__main__":
    unittest.main()
