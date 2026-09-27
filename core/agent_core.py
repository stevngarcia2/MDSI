"""
core/agent_core.py

Servicio principal de JOURNALI.

JOURNALI es un agente inteligente de apoyo a la
investigación científica y publicación académica.

Integra:

- Memoria conversacional.
- Modelo LLM configurable.
- Herramientas de investigación.
- Recuperación documental mediante RAG.
- Análisis científico.
- Síntesis de información.
- Generación de reportes estructurados.
- Coordinación mediante LangGraph.

Patrón:
    Singleton

Una única instancia de AgentService se crea al iniciar
el servidor y se reutiliza en las diferentes peticiones.
"""

import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

from core.graph import construir_grafo
from core.tools import TOOLS_BASE


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

NOMBRE_AGENTE = os.getenv(
    "AGENT_NAME",
    "JOURNALI"
)


# ============================================================
# SERVICIO PRINCIPAL
# ============================================================

class AgentService:
    """
    Servicio principal que encapsula el grafo de JOURNALI.

    El servicio administra:

    - sesiones de conversación;
    - memoria;
    - configuración del modelo;
    - ejecución del grafo;
    - herramientas disponibles.
    """

    def __init__(self):

        # ----------------------------------------------------
        # Memoria por sesión
        # ----------------------------------------------------

        self._historiales: dict[
            str,
            ChatMessageHistory
        ] = {}

        # ----------------------------------------------------
        # Configuración del modelo
        # ----------------------------------------------------

        self._modelo = os.getenv(
            "GROQ_MODEL",
            os.getenv(
                "OLLAMA_MODEL",
                "desconocido"
            )
        )

        self._proveedor = os.getenv(
            "LLM_PROVIDER",
            "groq"
        )

        # ----------------------------------------------------
        # Configuración de memoria
        # ----------------------------------------------------

        self._memory_type = os.getenv(
            "MEMORY_TYPE",
            "buffer"
        )

        self._window = int(
            os.getenv(
                "MEMORY_WINDOW_SIZE",
                "6"
            )
        )

        # ----------------------------------------------------
        # Construcción del grafo
        # ----------------------------------------------------

        self._grafo = construir_grafo()

        print(
            f"✅ {NOMBRE_AGENTE} listo — "
            f"{self._proveedor}/"
            f"{self._modelo} · "
            f"memoria={self._memory_type}"
        )


    # ========================================================
    # MEMORIA
    # ========================================================

    def _obtener_historial(
        self,
        session_id: str
    ) -> ChatMessageHistory:

        if session_id not in self._historiales:

            self._historiales[
                session_id
            ] = ChatMessageHistory()

        historial = self._historiales[
            session_id
        ]

        # ----------------------------------------------------
        # Memoria tipo window
        # ----------------------------------------------------

        if (
            self._memory_type == "window"
            and len(historial.messages)
            > self._window
        ):

            historial.messages = (
                historial.messages[
                    -self._window:
                ]
            )

        return historial


    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        mensaje: str,
        session_id: str = "default"
    ) -> dict:

        """
        Procesa un mensaje mediante el grafo completo
        de JOURNALI.

        El coordinador determina si la solicitud corresponde
        a:

        - conversación;
        - consulta documental;
        - análisis científico;
        - reporte estructurado.
        """

        historial = self._obtener_historial(
            session_id
        )

        estado_inicial = {
            "entrada_usuario": mensaje,
            "historial": historial.messages,
            "categoria": "",
            "respuesta_final": "",
            "fuentes": [],
        }

        # ----------------------------------------------------
        # Ejecución del grafo
        # ----------------------------------------------------

        try:

            resultado = self._grafo.invoke(
                estado_inicial
            )

        except Exception as error:

            raise RuntimeError(
                f"Error en JOURNALI: {error}"
            ) from error

        # ----------------------------------------------------
        # Respuesta
        # ----------------------------------------------------

        respuesta = resultado[
            "respuesta_final"
        ]

        # ----------------------------------------------------
        # Actualización de memoria
        # ----------------------------------------------------

        historial.add_user_message(
            mensaje
        )

        historial.add_ai_message(
            respuesta
        )

        # ----------------------------------------------------
        # Resultado de la API
        # ----------------------------------------------------

        return {
            "respuesta": respuesta,
            "categoria": resultado.get(
                "categoria",
                ""
            ),
            "fuentes": resultado.get(
                "fuentes",
                []
            ),
        }


    # ========================================================
    # EJECUCIÓN DE TAREAS
    # ========================================================

    def ejecutar_tarea(
        self,
        tarea: str,
        contexto: str = None,
        session_id: str = "default"
    ) -> dict:

        """
        Ejecuta una tarea específica.

        Puede recibir contexto adicional para enriquecer
        la solicitud.
        """

        entrada = tarea

        if contexto:

            entrada = (
                f"Contexto:\n"
                f"{contexto}\n\n"
                f"Tarea:\n"
                f"{tarea}"
            )

        return self.chat(
            entrada,
            session_id
        )


    # ========================================================
    # LIMPIEZA DE SESIÓN
    # ========================================================

    def limpiar_sesion(
        self,
        session_id: str
    ) -> bool:

        """
        Elimina la memoria de una sesión.
        """

        if session_id in self._historiales:

            del self._historiales[
                session_id
            ]

            return True

        return False


    # ========================================================
    # PROPIEDADES
    # ========================================================

    @property
    def nombre(self) -> str:

        return NOMBRE_AGENTE


    @property
    def proveedor(self) -> str:

        return self._proveedor


    @property
    def modelo(self) -> str:

        return self._modelo


    @property
    def herramientas(self) -> list:

        return [
            herramienta.name
            for herramienta in TOOLS_BASE
        ]


# ============================================================
# INSTANCIA GLOBAL
# ============================================================

# Una única instancia del servicio se crea al iniciar
# la aplicación.

agente_service = AgentService()