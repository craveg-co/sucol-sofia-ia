# tests/simular_ilama.py — Conversaciones de ejemplo de Sofía con Reservas de Ilama

"""
Simula clientes de Reservas de Ilama contra la API real de Anthropic.
Las herramientas que escriben en el CRM o avisan a asesores están reemplazadas
por versiones falsas que solo imprimen lo que habrían hecho.

Uso:  venv\\Scripts\\python.exe tests\\simular_ilama.py
"""

import asyncio
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools
from agent.brain import generar_respuesta_con_tools

PROYECTO = {"slug": "reservas_ilama", "nombre": "Reservas de Ilama"}


def _falsa(nombre: str, resultado: str):
    async def herramienta(*args, **kwargs):
        print(f"   [HERRAMIENTA SIMULADA] {nombre}: {kwargs}")
        return resultado
    return herramienta


CONVERSACION_FUERA_CRM = [
    "Hola, buenas tardes, vi un anuncio de un condominio en el lago Calima",
    "Mi nombre es Carlos Rojas, vivo en Cali",
    "Qué servicios trae?",
    "Dónde queda?",
    "Me puede enviar el brochure?",
    "Cuántos lotes quedan y cuánto cuestan?",
    "Está caro, estoy mirando otro proyecto también",
    "Qué descuento hay si pago de contado?",
    "Cuántos pisos puedo construir en el tipo C?",
    "Quiero invertir, tengo unos 80 millones de inicial. Lo decido con mi esposa",
    "Sí, prefiero ir el sábado en la mañana",
    "A las 10am. Mi nombre completo es Carlos Rojas Pérez, celular 3001234567",
    "Ya hablé con mi esposa, quiero separar el lote",
]

CONVERSACION_CRM = [
    "__INICIAR__",
    "Hola, sí me interesa",
    "Qué tipos de lotes hay?",
]


async def correr(titulo: str, mensajes: list[str], lead: dict | None):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    historial: list[dict] = []
    for mensaje in mensajes:
        print(f"\nCLIENTE: {mensaje}")
        respuesta = await generar_respuesta_con_tools(
            mensaje,
            historial,
            contexto_lead=lead,
            telefono="+570000000000",
            proyecto_slug=PROYECTO["slug"],
            proyecto=PROYECTO,
        )
        print(f"SOFÍA:\n{respuesta}")
        if mensaje != "__INICIAR__":
            historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})


async def main():
    with patch.object(tools, "confirmar_cita", _falsa("confirmar_cita", "Cita agendada (simulada)")), \
         patch.object(tools, "escalar_a_asesor", _falsa("escalar_a_asesor", "Asesor notificado (simulado)")), \
         patch.object(tools, "calificar_lead_sin_visita", _falsa("calificar_lead_sin_visita", "Lead registrado (simulado)")), \
         patch.object(tools, "notificar_area_por_correo", _falsa("notificar_area_por_correo", "Correo enviado (simulado)")), \
         patch.object(tools, "consultar_asesor_por_nombre", _falsa("consultar_asesor_por_nombre", "Sin datos (simulado)")):
        await correr("CLIENTE QUE LLEGA POR FUERA DEL CRM", CONVERSACION_FUERA_CRM, None)
        await correr(
            "CLIENTE QUE LLEGA POR EL CRM",
            CONVERSACION_CRM,
            {"nombre_completo": "Andrea Gómez", "canal": "Redes sociales", "proyecto": "Reservas de Ilama"},
        )


if __name__ == "__main__":
    asyncio.run(main())
