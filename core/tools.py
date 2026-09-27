"""
core/tools.py — Herramientas especializadas de JOURNALI

JOURNALI es un agente inteligente de apoyo a la investigación
científica y publicación académica.

Las herramientas deben estar directamente relacionadas con:

- consulta documental
- recuperación de literatura
- análisis de información científica
- apoyo a procesos de investigación

La herramienta principal utiliza el motor RAG del proyecto.
"""

import os
import sys

from langchain_core.tools import tool

# Permite importar módulos desde la raíz del proyecto
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from core.rag import obtener_motor


# ============================================================
# BÚSQUEDA DOCUMENTAL
# ============================================================

@tool
def buscar_documentos(pregunta: str) -> str:
    """
    Busca información relevante en el corpus documental
    disponible para JOURNALI.

    Utiliza el motor RAG para recuperar información de los
    documentos disponibles en data/documentos.

    Úsala cuando el usuario solicite:

    - información contenida en documentos;
    - conceptos o definiciones presentes en el corpus;
    - autores o trabajos documentados;
    - comparación de información documental;
    - síntesis basada en los documentos;
    - identificación de información relevante para una
      investigación.
    """

    try:
        motor = obtener_motor()

        texto, fuentes = motor.buscar(pregunta)

        if not texto:
            return (
                "No se encontró información relevante en "
                "el corpus documental disponible."
            )

        respuesta = texto

        if fuentes:
            respuesta += (
                "\n\nFuentes consultadas: "
                + ", ".join(fuentes)
            )

        return respuesta

    except Exception as e:
        return (
            "No fue posible realizar la búsqueda documental. "
            f"Detalle técnico: {e}"
        )


# ============================================================
# HERRAMIENTAS BASE DE JOURNALI
# ============================================================

TOOLS_BASE = [
    buscar_documentos,
]