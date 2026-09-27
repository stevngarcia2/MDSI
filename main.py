"""
main.py — Servidor principal de JOURNALI

JOURNALI es un agente inteligente de apoyo a la investigación
científica y la publicación académica.

La aplicación expone:

- Interfaz web en /
- Chat conversacional en /chat
- Ejecución de tareas del agente en /agent
- Estado del sistema en /health
- Estado detallado en /health/detail
- Información de la API en /api
- Documentación interactiva en /docs
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN DEL ENTORNO
# ============================================================

# Cargar variables antes de importar los módulos de la aplicación.
load_dotenv()


# ============================================================
# FASTAPI
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from routes.chat import router as chat_router
from routes.agent import router as agent_router
from routes.health import router as health_router


# ============================================================
# IDENTIDAD DE JOURNALI
# ============================================================

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "JOURNALI")

ROL_AGENTE = os.getenv(
    "AGENT_ROLE",
    "agente inteligente de apoyo a la investigación científica",
)

DOMINIO_AGENTE = os.getenv(
    "AGENT_DOMAIN",
    "investigación científica y publicación académica",
)

PROPOSITO_AGENTE = os.getenv(
    "SYSTEM_PURPOSE",
    (
        "apoyar procesos de investigación científica mediante "
        "consulta documental, análisis de literatura, identificación "
        "de tendencias, síntesis de información y elaboración de "
        "reportes de investigación"
    ),
)


# ============================================================
# APLICACIÓN
# ============================================================

app = FastAPI(
    title=f"{NOMBRE_AGENTE} — Agente de Investigación Científica",
    description=f"""
# {NOMBRE_AGENTE}

Agente inteligente de apoyo a la investigación científica y la
publicación académica.

**Rol:** {ROL_AGENTE}

**Dominio:** {DOMINIO_AGENTE}

**Propósito:** {PROPOSITO_AGENTE}

## Capacidades principales

- Consulta de documentos científicos y académicos.
- Recuperación de información mediante RAG.
- Análisis de literatura científica.
- Identificación de tendencias.
- Síntesis de información.
- Identificación de hallazgos y vacíos de investigación.
- Propuesta de líneas de investigación.
- Elaboración de reportes estructurados.
- Memoria conversacional por sesión.

## Endpoints principales

- **POST /chat** — Conversación con JOURNALI.
- **POST /agent** — Ejecución de tareas de investigación.
- **GET /health** — Estado básico del servicio.
- **GET /health/detail** — Estado detallado del agente.
- **GET /api** — Información general de la API.
- **GET /** — Interfaz web de JOURNALI.
- **GET /docs** — Documentación interactiva.
""",
    version="1.0.0",
    contact={
        "name": "JOURNALI",
    },
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    health_router,
    tags=["Monitoreo"],
)

app.include_router(
    chat_router,
    tags=["Chat"],
)

app.include_router(
    agent_router,
    tags=["Agente"],
)


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_INDEX = Path(__file__).parent / "frontend" / "index.html"


@app.get("/", tags=["JOURNALI"])
async def root():
    """Sirve la interfaz web de JOURNALI."""
    return FileResponse(
        FRONTEND_INDEX,
        media_type="text/html; charset=utf-8",
    )


# ============================================================
# INFORMACIÓN DE LA API
# ============================================================

@app.get("/api", tags=["JOURNALI"])
async def info():
    """Devuelve información general de JOURNALI."""
    return {
        "application": "JOURNALI",
        "agent": NOMBRE_AGENTE,
        "role": ROL_AGENTE,
        "domain": DOMINIO_AGENTE,
        "purpose": PROPOSITO_AGENTE,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
        "agent_endpoint": "/agent",
    }


# ============================================================
# ARRANQUE DIRECTO
# ============================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "True").lower() == "true"

    print()
    print("=" * 65)
    print("  JOURNALI — AGENTE DE INVESTIGACIÓN CIENTÍFICA")
    print("=" * 65)
    print(f"  Agente : {NOMBRE_AGENTE}")
    print(f"  Rol    : {ROL_AGENTE}")
    print(f"  Dominio: {DOMINIO_AGENTE}")
    print(f"  URL    : http://localhost:{port}")
    print(f"  Docs   : http://localhost:{port}/docs")
    print("=" * 65)
    print()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
    )