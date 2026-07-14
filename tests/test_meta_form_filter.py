import unittest

from agent.main import _es_autorespuesta_no_disponible, _es_mensaje_formulario_meta


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

    def test_detecta_autorespuesta_no_disponible(self):
        texto = (
            "Buen dia Gracias por comunicarte con Pst. Daniel Canencio del Ministerio "
            "Enhacore para Las Naciones en este momento no puedo responder pero "
            "déjame tu mensaje y me estaré comunicando lo más pronto posible te mando "
            "un abrazo Dios te bendiga😃"
        )

        self.assertTrue(_es_autorespuesta_no_disponible(texto))

    def test_detecta_autorespuesta_no_puedo_contestar(self):
        texto = (
            "Hola Dios te bendiga Gracias por comunicarte con Pst. Daniel Canencio "
            "en este momento no puedo contestar pero déjame tu mensaje y me estaré "
            "comunicando lo más pronto posible"
        )

        self.assertTrue(_es_autorespuesta_no_disponible(texto))

    def test_no_bloquea_mensaje_comercial_con_no_puedo_responder(self):
        texto = "Ahora no puedo responder, pero me interesa el proyecto Bora y quiero precios de lotes"

        self.assertFalse(_es_autorespuesta_no_disponible(texto))

    def test_detecta_autorespuesta_generica_sin_texto_religioso(self):
        texto = (
            "Gracias por escribirnos. En este momento no estamos disponibles, "
            "deja tu mensaje y te responderemos tan pronto podamos."
        )

        self.assertTrue(_es_autorespuesta_no_disponible(texto))

    def test_detecta_autorespuesta_empresarial_fuera_de_horario(self):
        texto = (
            "Bienvenido a ABC Servicios. Estamos fuera de horario. "
            "Déjanos tu mensaje y nos comunicaremos lo más pronto posible."
        )

        self.assertTrue(_es_autorespuesta_no_disponible(texto))

    def test_no_bloquea_respuesta_humana_corta_no_disponible(self):
        texto = "No puedo responder ahora, escríbeme más tarde."

        self.assertFalse(_es_autorespuesta_no_disponible(texto))

    def test_no_bloquea_cliente_que_pide_retomar_luego_con_proyecto(self):
        texto = "Estoy ocupado, pero me interesa Vientos de Ginebra. Mañana reviso precios."

        self.assertFalse(_es_autorespuesta_no_disponible(texto))


if __name__ == "__main__":
    unittest.main()
