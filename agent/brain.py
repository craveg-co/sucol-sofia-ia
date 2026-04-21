# agent/brain.py — Cerebro de Sofía: conexión con Claude API
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Lógica de IA de Sofía. Soporta prompts dinámicos por proyecto desde el CRM
y un prompt genérico de bienvenida cuando el cliente aún no tiene proyecto asignado.
"""

import os
import yaml
import logging
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("agentkit")

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Prompt base de Sofía — se usa cuando el CRM no tiene proyecto para este cliente
_PROMPT_BIENVENIDA = """Eres Sofía, la asesora virtual de Sucol Soluciones Urbanísticas.

## Tu rol
Atiendes a personas interesadas en adquirir lotes o proyectos urbanísticos de Sucol.
Tu objetivo es entender en qué proyecto está interesado el cliente y conectarlo con la
información correcta.

## Proyectos disponibles
{lista_proyectos}

## Cómo actuar
- Saluda de forma cálida y profesional
- Pregunta por cuál de los proyectos le interesa obtener información
- Una vez que identifiques el proyecto, el sistema te dará información detallada
- Si el cliente no está seguro, descríbele brevemente cada proyecto y ayúdalo a elegir
- NUNCA inventes precios ni datos que no tengas — di que lo conectarás con un asesor

## Reglas
- Responde siempre en español
- Sé empática, clara y profesional
- Mantén las respuestas cortas y útiles
- Termina siempre con una pregunta o invitación a continuar"""


def _cargar_config_prompts() -> dict:
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _prompt_base_yaml() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "system_prompt",
        "Eres Sofía, asesora virtual de Sucol Soluciones Urbanísticas. Responde en español.",
    )


def _mensaje_error() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "error_message",
        "Lo siento, estoy teniendo un pequeño inconveniente técnico. Por favor intenta de nuevo en unos minutos.",
    )


def _mensaje_fallback() -> str:
    config = _cargar_config_prompts()
    return config.get(
        "fallback_message",
        "Disculpa, no entendí bien tu mensaje. ¿Puedes contarme en qué te puedo ayudar?",
    )


def _construir_contexto_lead(lead: dict) -> str:
    """Convierte los datos del lead en texto para agregar al system prompt."""
    if not lead:
        return ""
    partes = ["## Información del cliente en el CRM"]
    if lead.get("nombre_completo"):
        partes.append(f"- Nombre: {lead['nombre_completo']}")
    if lead.get("etapa_lead"):
        partes.append(f"- Etapa: {lead['etapa_lead']}")
    if lead.get("asesor_responsable"):
        partes.append(f"- Asesor asignado: {lead['asesor_responsable']}")
    if lead.get("proyecto"):
        partes.append(f"- Proyecto de interés: {lead['proyecto']}")
    partes.append(
        "\nUsa este contexto para personalizar tu atención. "
        "Si el cliente ya tiene asesor asignado y necesita algo urgente, "
        "ofrécele conectarlo directamente con él/ella."
    )
    return "\n".join(partes)


async def _prompt_bienvenida_con_proyectos() -> str:
    """Genera el prompt genérico listando los proyectos activos del CRM."""
    try:
        from agent.crm import obtener_proyectos_activos
        proyectos = await obtener_proyectos_activos()
        if proyectos:
            lista = "\n".join(f"- {p['nombre']}" for p in proyectos)
        else:
            lista = "- Proyectos urbanísticos Sucol (consulta disponibilidad)"
    except Exception:
        lista = "- Proyectos urbanísticos Sucol (consulta disponibilidad)"
    return _PROMPT_BIENVENIDA.format(lista_proyectos=lista)


async def generar_respuesta(
    mensaje: str,
    historial: list[dict],
    sistema_prompt: str | None = None,
    contexto_lead: dict | None = None,
) -> str:
    """
    Genera una respuesta usando Claude API (claude-sonnet-4-6).

    Args:
        mensaje: El mensaje nuevo del cliente
        historial: Mensajes anteriores [{"role": "...", "content": "..."}]
        sistema_prompt: System prompt del proyecto desde el CRM (prioridad máxima).
                        Si es None, usa prompt genérico de bienvenida.
        contexto_lead: Datos del lead desde el CRM para personalizar la respuesta.

    Returns:
        La respuesta generada por Sofía
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return _mensaje_fallback()

    # Determinar el system prompt a usar
    if sistema_prompt:
        prompt_final = sistema_prompt
    else:
        # Sin proyecto asignado → prompt genérico con lista de proyectos
        prompt_final = await _prompt_bienvenida_con_proyectos()

    # Inyectar contexto del lead si existe
    if contexto_lead:
        prompt_final += "\n\n" + _construir_contexto_lead(contexto_lead)

    mensajes = [{"role": m["role"], "content": m["content"]} for m in historial]
    mensajes.append({"role": "user", "content": mensaje})

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=prompt_final,
            messages=mensajes,
        )
        respuesta = response.content[0].text
        logger.info(
            f"Respuesta generada ({response.usage.input_tokens} in / {response.usage.output_tokens} out)"
        )
        return respuesta

    except Exception as e:
        logger.error(f"Error Claude API: {e}")
        return _mensaje_error()
