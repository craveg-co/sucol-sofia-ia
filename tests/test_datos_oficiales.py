import unittest

from agent.brain import (
    _cargar_knowledge,
    _corregir_disponibilidad,
    _construir_contexto_crm,
    _procesar_respuesta_cliente,
    _quitar_mencion_asesor_no_solicitada,
    _respuesta_rango_horario,
    _respuesta_cambio_interes,
    _respuesta_recursos_proyecto,
    _respuesta_referencia_publicidad,
    _respuesta_sin_proyecto,
    _resumen_cita_oficial,
    _respuesta_operativa_visita,
    _sanitizar_historial,
    _validar_respuesta_oficial,
    separar_mensajes_whatsapp,
)


PROYECTO = {
    "slug": "santa_elena",
    "nombre": "Santa Elena",
    "direccion_visita": "Cra. 10 # 1-1 Villas del Parque Local 1",
    "google_maps_url": "https://maps.app.goo.gl/9tWjSGv824huNMoQA",
}


class DatosOficialesTest(unittest.TestCase):
    def test_vi_publicidad_no_inventa_detalles(self):
        respuesta = _respuesta_referencia_publicidad(
            "Vi la publicidad",
            {"slug": "buenavista", "nombre": "Buenavista"},
        )

        self.assertIn("publicidad de Buenavista", respuesta)
        self.assertNotIn("km", respuesta.lower())
        self.assertNotIn("$", respuesta)

    def test_en_el_enlace_pide_aclaracion(self):
        respuesta = _respuesta_referencia_publicidad(
            "En el enlace",
            {"slug": "buenavista", "nombre": "Buenavista"},
        )

        self.assertIn("qué información del enlace", respuesta.lower())
        self.assertNotIn("Cra.", respuesta)

    def test_buenavista_jamundi_no_se_trata_como_cambio_de_zona(self):
        proyecto = {"slug": "buenavista", "nombre": "Buenavista"}

        respuesta = _respuesta_cambio_interes(
            "Buenavista Jamundí",
            [],
            proyecto,
        )

        self.assertIsNone(respuesta)

    def test_ubicacion_buenavista_sale_de_la_ficha_oficial(self):
        respuesta = _respuesta_recursos_proyecto(
            "Quiero la ubicación exacta",
            {
                "slug": "buenavista",
                "nombre": "Buenavista",
                "direccion_visita": PROYECTO["direccion_visita"],
                "google_maps_url": PROYECTO["google_maps_url"],
            },
        )

        self.assertIn("sur de Jamundí", respuesta)
        self.assertNotIn(PROYECTO["direccion_visita"], respuesta)
        self.assertNotIn(PROYECTO["google_maps_url"], respuesta)
        self.assertIn("enlace oficial del terreno", respuesta)

    def test_imagenes_sin_galeria_no_fuerzan_una_cita(self):
        respuesta = _respuesta_recursos_proyecto(
            "Quiero ver imágenes del proyecto",
            {"slug": "buenavista", "nombre": "Buenavista"},
        )

        self.assertIn("no tengo una galería oficial", respuesta.lower())
        self.assertNotIn("agend", respuesta.lower())

    def test_imagenes_y_ubicacion_se_responden_juntas(self):
        respuesta = _respuesta_recursos_proyecto(
            "Quiero ver imágenes del lugar exacto y la ubicación exacta",
            {"slug": "buenavista", "nombre": "Buenavista"},
        )

        self.assertIn("galería oficial", respuesta)
        self.assertIn("sur de Jamundí", respuesta)

    def test_slug_vientos_de_ginebra_carga_ficha_local(self):
        contenido = _cargar_knowledge("vientos_de_ginebra")

        self.assertIn("VIENTOS DE GINEBRA", contenido)

    def test_sin_proyecto_no_inventa_un_nombre(self):
        respuesta = _respuesta_sin_proyecto("Hola")

        self.assertIn("qué proyecto o zona", respuesta)
        self.assertNotIn("Ciudadela del Río", respuesta)

    def test_sin_proyecto_pide_identificarlo_antes_de_agendar(self):
        respuesta = _respuesta_sin_proyecto("Programar una llamada")

        self.assertIn("sobre cuál proyecto", respuesta)

    def test_rango_horario_requiere_hora_exacta(self):
        respuesta = _respuesta_rango_horario(
            "Por favor llamarme viernes 10 de julio entre 8 y 9 am"
        )

        self.assertIsNotNone(respuesta)
        self.assertIn("hora exacta", respuesta)

    def test_resumen_cita_usa_proyecto_oficial(self):
        resumen = _resumen_cita_oficial("Llamada", PROYECTO)

        self.assertIn("Santa Elena", resumen)
        self.assertNotIn("Ciudadela del Río", resumen)

    def test_bloquea_nombre_de_proyecto_distinto_al_crm(self):
        respuesta = _procesar_respuesta_cliente(
            "Nuestro proyecto Ciudadela del Río tiene lotes desde 1.000 m².",
            "Hola",
            PROYECTO,
            [],
            None,
        )

        self.assertIn("Santa Elena", respuesta)
        self.assertNotIn("Ciudadela del Río", respuesta)
        self.assertNotIn("1.000", respuesta)

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

    def test_reemplaza_fragmento_para_el_proyecto(self):
        respuesta = _procesar_respuesta_cliente(
            "Para el proyecto",
            "Dame información del proyecto Bora",
            {"slug": "bora", "nombre": "Bora"},
            [
                {"area_m2": 48, "precio_total": 28142985},
                {"area_m2": 145.18, "precio_total": 103230253},
            ],
            {"nombre": "Juliana Duque", "telefono": "+573170402005"},
        )

        self.assertNotEqual(respuesta, "Para el proyecto")
        self.assertIn("Bora tiene 2 opciones disponibles", respuesta)
        self.assertIn("48", respuesta)
        self.assertIn("145.18", respuesta)
        self.assertIn("precios desde $28,142,985", respuesta)
        self.assertIn("agendar una visita", respuesta)

    def test_filtro_de_asesora_no_puede_dejar_respuesta_fragmentada(self):
        respuesta = _procesar_respuesta_cliente(
            "Para el proyecto\n\n"
            "Tu asesora Juliana Duque (+573170402005) puede darte más información.",
            "Dame información del proyecto Bora",
            {"slug": "bora", "nombre": "Bora"},
            [{"area_m2": 48, "precio_total": 28142985}],
            {"nombre": "Juliana Duque", "telefono": "+573170402005"},
        )

        self.assertNotIn("Juliana", respuesta)
        self.assertNotIn("Para el proyecto", respuesta)
        self.assertIn("Bora tiene 1 opciones disponibles", respuesta)

    def test_separa_informacion_y_pregunta_en_dos_mensajes(self):
        mensajes = separar_mensajes_whatsapp(
            "Bora está ubicado en Jamundí y tiene opciones desde 48 m². "
            "¿Quieres más información o prefieres agendar una visita?",
            {"slug": "bora", "nombre": "Bora"},
        )

        self.assertEqual(len(mensajes), 2)
        self.assertIn("Bora está ubicado", mensajes[0])
        self.assertTrue(mensajes[1].startswith("¿"))

    def test_agrega_cta_si_el_modelo_no_la_genero(self):
        mensajes = separar_mensajes_whatsapp(
            "Bora está ubicado en Jamundí y cuenta con opciones para vivienda o negocio.",
            {"slug": "bora", "nombre": "Bora"},
        )

        self.assertEqual(len(mensajes), 2)
        self.assertIn("agendar una visita", mensajes[1])
        self.assertIn("programar una llamada", mensajes[1])

    def test_cascata_no_ofrece_visita_directa_en_cta_generada(self):
        mensajes = separar_mensajes_whatsapp(
            "Cascata ofrece eco-hábitats sustentables en Pance.",
            {"slug": "cascata", "nombre": "Cascata"},
        )

        self.assertIn("recorrido virtual 360°", mensajes[1])
        self.assertNotIn("agendar una visita", mensajes[1])


if __name__ == "__main__":
    unittest.main()
