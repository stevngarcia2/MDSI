"""
routes/chat.py — Endpoints de conversación de JOURNALI

Expone el endpoint de conversación con memoria por sesión y
permite limpiar el historial de una sesión específica.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, ErrorResponse
from core.agent_core import agente_service


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversar con JOURNALI",
    description=(
        "Envía una consulta a JOURNALI y recibe una respuesta "
        "generada mediante el sistema de agentes. "
        "La conversación mantiene memoria asociada al session_id."
    ),
    responses={
        500: {"model": ErrorResponse},
    },
)
async def chat(request: ChatRequest):
    """
    Procesa una consulta del usuario mediante JOURNALI.

    Body:
        mensaje:
            Consulta o instrucción del usuario.

        session_id:
            Identificador de la sesión conversacional.
            Si no se proporciona, se utiliza el valor definido
            por el esquema de solicitud.

    Returns:
        ChatResponse con:

        - respuesta: respuesta generada por JOURNALI.
        - session_id: identificador de la sesión.
        - modelo: modelo LLM utilizado.
        - categoria: categoría determinada por el coordinador.
        - fuentes: fuentes documentales recuperadas mediante RAG.
    """

    try:
        resultado = agente_service.chat(
            mensaje=request.mensaje,
            session_id=request.session_id,
        )

        return ChatResponse(
            respuesta=resultado["respuesta"],
            session_id=request.session_id,
            modelo=agente_service.modelo,
            categoria=resultado.get("categoria"),
            fuentes=resultado.get("fuentes", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la consulta en JOURNALI: {str(e)}",
        )


@router.delete(
    "/chat/{session_id}",
    summary="Limpiar sesión de JOURNALI",
    description=(
        "Elimina el historial conversacional asociado "
        "a una sesión específica de JOURNALI."
    ),
)
async def limpiar_sesion(session_id: str):
    """
    Elimina el historial de una sesión conversacional.

    Args:
        session_id:
            Identificador de la sesión que se desea eliminar.

    Returns:
        Mensaje indicando si la sesión fue eliminada.
    """

    eliminado = agente_service.limpiar_sesion(session_id)

    if eliminado:
        return {
            "mensaje": (
                f"Sesión '{session_id}' de JOURNALI "
                "eliminada correctamente."
            )
        }

    return {
        "mensaje": (
            f"Sesión '{session_id}' no encontrada."
        )
    }