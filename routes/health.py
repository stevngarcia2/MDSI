"""
routes/health.py — Endpoints de monitoreo de JOURNALI

Proporciona información sobre el estado operativo del agente,
su proveedor LLM, modelo y herramientas disponibles.
"""

from fastapi import APIRouter

from models.schemas import HealthResponse
from core.agent_core import agente_service


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de JOURNALI",
    description=(
        "Verifica que JOURNALI esté activo y devuelve "
        "información básica del servicio."
    ),
)
async def health_check():
    """
    Verificación básica de disponibilidad.

    Este endpoint está diseñado para comprobaciones de salud
    del servicio y despliegues en plataformas como Railway.
    """

    return HealthResponse(
        status="ok",
        agente=agente_service.nombre,
        proveedor=agente_service.proveedor,
    )


@router.get(
    "/health/detail",
    summary="Estado detallado de JOURNALI",
    description=(
        "Devuelve información detallada sobre JOURNALI, "
        "incluyendo modelo LLM y herramientas disponibles."
    ),
)
async def health_detail():
    """
    Devuelve el estado completo del agente.
    """

    return {
        "status": "ok",
        "agente": agente_service.nombre,
        "rol": "agente inteligente de apoyo a la investigación científica",
        "proveedor": agente_service.proveedor,
        "modelo": agente_service.modelo,
        "herramientas": agente_service.herramientas,
        "dominio": "investigación científica y publicación académica",
    }