import os
from typing import Optional

MAX_RESULT_CHARS = 1500

def run_cursor_agent(
    prompt: str,
    workspace: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    """Ejecuta un agente Cursor local one-shot (Agent.prompt) y devuelve un resumen.

    Requiere: pip install cursor-sdk y CURSOR_API_KEY en el entorno.
    """
    if not prompt or not prompt.strip():
        return "Debes indicar un prompt para el agente de Cursor."

    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError:
        return (
            "Falta el paquete cursor-sdk. Instálalo con: pip install cursor-sdk"
        )

    key = api_key or os.environ.get("CURSOR_API_KEY")
    if not key:
        return (
            "Falta CURSOR_API_KEY. Configura la variable de entorno. "
            "Puedes crearla en Cursor Dashboard > Integrations."
        )

    cwd = os.path.realpath(workspace)
    if not os.path.isdir(cwd):
        return f"El workspace '{workspace}' no existe o no es una carpeta."

    try:
        result = Agent.prompt(
            prompt.strip(),
            AgentOptions(
                api_key=key,
                model=model,
                local=LocalAgentOptions(cwd=cwd),
            ),
        )
    except CursorAgentError as err:
        retryable = getattr(err, "is_retryable", False)
        return f"No se pudo iniciar el agente Cursor: {err}. Reintentable: {retryable}."
    except Exception as e:
        return f"Error inesperado al llamar a Cursor SDK: {e}"

    status = getattr(result, "status", "unknown")
    if status == "error":
        run_id = getattr(result, "id", "?")
        return f"El agente Cursor falló durante la ejecución (run id: {run_id})."

    text = getattr(result, "result", None) or ""
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "..."

    if not text:
        return f"Agente Cursor terminó con estado '{status}' sin texto de respuesta."

    return f"Cursor ({status}): {text}"
