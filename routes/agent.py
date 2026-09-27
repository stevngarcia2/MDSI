"""
routes/agent.py — Endpoint de tareas de JOURNALI

Expone un endpoint para ejecutar tareas específicas de investigación
que pueden requerir contexto adicional.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import AgentRequest, AgentResponse, ErrorResponse
from core.agent_core import agente_service


router = APIRouter()


@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Ejecutar tarea de investigación",
    description=(
        "Ejecuta una tarea específica mediante JOURNALI. "
        "Permite proporcionar contexto adicional y mantener "
        "la sesión conversacional."
    ),
    responses={
        500: {"model": ErrorResponse},
    },
)
async def ejecutar_agente(request: AgentRequest):
    """
    Ejecuta una tarea específica mediante JOURNALI.

    Body:
        tarea:
            Descripción de la tarea de investigación.

        contexto:
            Información adicional que debe considerar JOURNALI.

        session_id:
            Identificador de la sesión conversacional.

    Returns:
        AgentResponse con:

        - resultado: respuesta generada.
        - tarea: tarea solicitada.
        - session_id: identificador de sesión.
        - categoria: categoría determinada por el agente.
        - fuentes: fuentes documentales recuperadas mediante RAG.
    """

    try:
        resultado = agente_service.ejecutar_tarea(
            tarea=request.tarea,
            contexto=request.contexto,
            session_id=request.session_id,
        )

        return AgentResponse(
            resultado=resultado["respuesta"],
            tarea=request.tarea,
            session_id=request.session_id,
            categoria=resultado.get("categoria"),
            fuentes=resultado.get("fuentes", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ejecutando la tarea en JOURNALI: {str(e)}",
        )