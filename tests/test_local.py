# tests/test_local.py — Simulador de chat con Sofía en terminal
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Prueba a Sofía sin necesitar WhatsApp.
Simula una conversación en la terminal como si fueras un cliente de Sucol.
"""

import asyncio
import sys
import os

# Agregar el directorio raíz al path para que encuentre el módulo 'agent'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.brain import generar_respuesta
from agent.memory import inicializar_db, guardar_mensaje, obtener_historial, limpiar_historial

TELEFONO_TEST = "test-local-001"


async def main():
    """Loop principal del chat de prueba con Sofía."""
    await inicializar_db()

    print()
    print("=" * 60)
    print("   Sofía — Agente de Sucol Soluciones Urbanísticas")
    print("   Test Local — Simulador de WhatsApp")
    print("=" * 60)
    print()
    print("  Escribe mensajes como si fueras un cliente de Sucol.")
    print("  Comandos especiales:")
    print("    'limpiar'  — borra el historial de la conversación")
    print("    'salir'    — termina el test")
    print()
    print("-" * 60)
    print()

    while True:
        try:
            mensaje = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nTest finalizado. ¡Hasta luego!")
            break

        if not mensaje:
            continue

        if mensaje.lower() == "salir":
            print("\nTest finalizado. ¡Hasta luego!")
            break

        if mensaje.lower() == "limpiar":
            await limpiar_historial(TELEFONO_TEST)
            print("[Historial borrado — nueva conversación]\n")
            continue

        # Obtener historial ANTES de guardar el mensaje actual
        historial = await obtener_historial(TELEFONO_TEST)

        # Generar respuesta con Sofía (Claude API)
        print("\nSofía: ", end="", flush=True)
        respuesta = await generar_respuesta(mensaje, historial)
        print(respuesta)
        print()

        # Guardar mensaje del cliente y respuesta de Sofía en memoria
        await guardar_mensaje(TELEFONO_TEST, "user", mensaje)
        await guardar_mensaje(TELEFONO_TEST, "assistant", respuesta)


if __name__ == "__main__":
    asyncio.run(main())
