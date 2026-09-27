"""
core/structured.py — Esquemas Pydantic para JOURNALI

Modelos de salida estructurada para apoyar procesos de
investigación científica y análisis de literatura.
"""

from typing import List
from pydantic import BaseModel, Field


class Hallazgo(BaseModel):
    """Hallazgo identificado durante el análisis documental."""

    descripcion: str = Field(
        description="Descripción clara del hallazgo identificado."
    )

    evidencia: str = Field(
        description="Evidencia documental que sustenta el hallazgo."
    )

    fuente: str = Field(
        description="Documento o fuente de la que se obtiene la evidencia."
    )


class ReporteInvestigacion(BaseModel):
    """Reporte estructurado para apoyar procesos de investigación científica."""

    tema: str = Field(
        description="Tema principal de la investigación."
    )

    contexto: str = Field(
        description="Contexto general del tema analizado."
    )

    conceptos_principales: List[str] = Field(
        description="Conceptos fundamentales identificados en la literatura."
    )

    hallazgos: List[Hallazgo] = Field(
        description="Principales hallazgos derivados del análisis documental."
    )

    tendencias: List[str] = Field(
        description="Tendencias identificadas en la literatura o documentación."
    )

    vacios_investigacion: List[str] = Field(
        description="Vacíos, limitaciones o problemas identificados."
    )

    lineas_investigacion: List[str] = Field(
        description="Posibles líneas o oportunidades de investigación."
    )

    conclusion: str = Field(
        description="Conclusión general del análisis realizado."
    )