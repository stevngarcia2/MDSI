"""
models/schemas.py — Modelos Pydantic de JOURNALI

Define los modelos de entrada y salida utilizados por la API
de JOURNALI.

Estos modelos permiten:

- Validar las solicitudes recibidas por FastAPI.
- Estructurar las respuestas del agente.
- Generar automáticamente la documentación OpenAPI/Swagger.
- Transportar categorías y fuentes documentales recuperadas
  mediante RAG.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================
# MODELOS DE ENTRADA
# ============================================================

class ChatRequest(BaseModel):
    """Solicitud de conversación con JOURNALI."""

    mensaje: str = Field(
        ...,
        description="Consulta o mensaje del usuario.",
        min_length=1,
        max_length=4000,
    )

    session_id: str = Field(
        default="default",
        description="Identificador de la sesión conversacional.",
        min_length=1,
        max_length=100,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "mensaje": (
                    "¿Cuáles son las principales tendencias "
                    "de investigación en agricultura de precisión?"
                ),
                "session_id": "investigacion_001",
            }
        }


class AgentRequest(BaseModel):
    """Solicitud para ejecutar una tarea específica de investigación."""

    tarea: str = Field(
        ...,
        description="Tarea de investigación que debe ejecutar JOURNALI.",
        min_length=1,
        max_length=5000,
    )

    contexto: Optional[str] = Field(
        default=None,
        description=(
            "Contexto adicional para orientar el análisis, "
            "como disciplina, problema, población o alcance."
        ),
        max_length=10000,
    )

    session_id: str = Field(
        default="default",
        description="Identificador de la sesión.",
        min_length=1,
        max_length=100,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tarea": (
                    "Analiza la literatura disponible e identifica "
                    "las principales líneas de investigación sobre "
                    "digitalización sostenible de procesos logísticos."
                ),
                "contexto": (
                    "Interés en tendencias tecnológicas, "
                    "metodologías de investigación y vacíos "
                    "identificados en la literatura."
                ),
                "session_id": "investigacion_001",
            }
        }


# ============================================================
# MODELOS DE SALIDA
# ============================================================

class ChatResponse(BaseModel):
    """Respuesta generada por JOURNALI mediante /chat."""

    respuesta: str = Field(
        ...,
        description="Respuesta generada por JOURNALI.",
    )

    session_id: str = Field(
        ...,
        description="Identificador de la sesión utilizada.",
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Fecha y hora de generación de la respuesta.",
    )

    modelo: str = Field(
        ...,
        description="Modelo LLM utilizado.",
    )

    categoria: Optional[str] = Field(
        default=None,
        description=(
            "Ruta determinada por el coordinador: "
            "conversacion, documentos, analisis o reporte."
        ),
    )

    fuentes: List[str] = Field(
        default_factory=list,
        description=(
            "Fuentes documentales recuperadas mediante el sistema RAG."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "respuesta": (
                    "La literatura consultada muestra tendencias "
                    "relacionadas con sensores IoT, análisis de datos, "
                    "teledetección y automatización de procesos agrícolas."
                ),
                "session_id": "investigacion_001",
                "timestamp": "2026-09-25T10:30:00",
                "modelo": "openai/gpt-oss-120b",
                "categoria": "analisis",
                "fuentes": [
                    "documento_agricultura_precision.pdf"
                ],
            }
        }


class AgentResponse(BaseModel):
    """Respuesta generada por JOURNALI mediante /agent."""

    resultado: str = Field(
        ...,
        description="Resultado de la tarea de investigación.",
    )

    tarea: str = Field(
        ...,
        description="Tarea solicitada al agente.",
    )

    session_id: str = Field(
        ...,
        description="Identificador de la sesión.",
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Fecha y hora de generación del resultado.",
    )

    categoria: Optional[str] = Field(
        default=None,
        description=(
            "Ruta determinada por el coordinador: "
            "conversacion, documentos, analisis o reporte."
        ),
    )

    fuentes: List[str] = Field(
        default_factory=list,
        description=(
            "Fuentes documentales recuperadas mediante RAG."
        ),
    )


class HealthResponse(BaseModel):
    """Estado básico del servicio JOURNALI."""

    status: str = Field(
        ...,
        description="Estado operativo del servicio.",
    )

    agente: str = Field(
        ...,
        description="Nombre del agente.",
    )

    proveedor: str = Field(
        ...,
        description="Proveedor del modelo de lenguaje.",
    )

    version: str = Field(
        default="1.0.0",
        description="Versión de la API.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "agente": "JOURNALI",
                "proveedor": "groq",
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """Respuesta estándar de error de la API."""

    error: str = Field(
        ...,
        description="Descripción del error.",
    )

    detalle: Optional[str] = Field(
        default=None,
        description="Detalle técnico del error.",
    )
