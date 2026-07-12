import unittest

from agent.main import _es_mensaje_formulario_meta


class MetaFormFilterTest(unittest.TestCase):
    def test_detecta_mensaje_automatico_de_formulario_meta(self):
        texto = (
            "¡Hola! Completé el formulario y me gustaría obtener más información sobre tu negocio.\n\n"
            "Full name: Jose Reinel Buitrago\n"
            "Phone number: +573188545396\n"
            "¿Para qué propósito deseas comprar el lote?: Construcción de vivienda\n"
            "Email: reinelbuitrago257@gmail.com"
        )

        self.assertTrue(_es_mensaje_formulario_meta(texto))

    def test_no_bloquea_mensaje_real_del_cliente(self):
        self.assertFalse(_es_mensaje_formulario_meta("Hola, quiero información de Bora"))

    def test_detecta_mensaje_automatico_recortado_de_formulario_meta(self):
        texto = "¡Hola! Completé el formulario y me gustaría obtener más información sobre tu negocio."

        self.assertTrue(_es_mensaje_formulario_meta(texto))

    def test_no_bloquea_cliente_que_menciona_email_sin_ser_formulario(self):
        texto = "Mi email es cliente@example.com y quiero saber precios de Bora"

        self.assertFalse(_es_mensaje_formulario_meta(texto))


if __name__ == "__main__":
    unittest.main()
