"""
core/rag.py
JOURNALI — Motor RAG para investigación científica.

Función:
    Cargar documentos científicos desde data/documentos,
    fragmentarlos, recuperar evidencia relevante y devolver
    contexto + fuentes para los agentes de JOURNALI.

Fuentes soportadas:
    - PDF
    - CSV

La carpeta _plantilla_original no se procesa porque el motor
solo busca archivos directamente dentro de data/documentos.

Arquitectura:
    documentos
        ↓
    extracción de texto
        ↓
    páginas
        ↓
    fragmentación
        ↓
    búsqueda híbrida
        ↓
    contexto + fuentes
"""

import os
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

# ------------------------------------------------------------
# Permitir imports desde la raíz del proyecto
# ------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# ------------------------------------------------------------
# LangChain
# ------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# CONFIGURACIÓN
# ============================================================

DOCUMENTOS_DIR = Path(
    os.getenv(
        "DOCUMENTOS_DIR",
        "data/documentos"
    )
)

TAMANO_FRAGMENTO = int(
    os.getenv(
        "TAMANO_FRAGMENTO",
        "1000"
    )
)

SUPERPOSICION_FRAGMENTO = int(
    os.getenv(
        "SUPERPOSICION_FRAGMENTO",
        "200"
    )
)

K_FRAGMENTOS = int(
    os.getenv(
        "K_FRAGMENTOS",
        "5"
    )
)


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS_ES = {
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "han",
    "hasta",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "más",
    "mas",
    "me",
    "mi",
    "mis",
    "muy",
    "no",
    "nos",
    "o",
    "para",
    "pero",
    "por",
    "que",
    "qué",
    "se",
    "sin",
    "sobre",
    "son",
    "su",
    "sus",
    "también",
    "te",
    "tú",
    "un",
    "una",
    "uno",
    "unos",
    "y",
    "ya",
}


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalizar(texto: str) -> str:
    """
    Convierte el texto a minúsculas, elimina tildes y
    reemplaza signos de puntuación por espacios.
    """

    if not texto:
        return ""

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    texto = "".join(
        caracter
        if caracter.isalnum() or caracter.isspace()
        else " "
        for caracter in texto
    )

    return texto


def palabras_clave(texto: str) -> set:
    """
    Extrae palabras relevantes de un texto.
    """

    tokens = normalizar(texto).split()

    return {
        token
        for token in tokens
        if token not in STOPWORDS_ES
        and len(token) > 1
    }


# ============================================================
# CARGA DE DOCUMENTOS
# ============================================================

def _cargar_pdfs() -> List[Document]:
    """
    Carga únicamente los PDF ubicados directamente
    en data/documentos.
    """

    documentos = []

    if not DOCUMENTOS_DIR.exists():
        return documentos

    for ruta in sorted(
        DOCUMENTOS_DIR.glob("*.pdf")
    ):

        try:

            cargados = PyPDFLoader(
                str(ruta)
            ).load()

            # Garantizar que cada página conserve
            # el nombre real del archivo.
            for documento in cargados:

                documento.metadata[
                    "source"
                ] = str(ruta)

                documento.metadata[
                    "filename"
                ] = ruta.name

                documento.metadata[
                    "file_type"
                ] = "pdf"

            documentos.extend(
                cargados
            )

            print(
                f"  📄 PDF cargado: "
                f"{ruta.name} "
                f"({len(cargados)} páginas)"
            )

        except Exception as error:

            print(
                f"  ⚠️ Error leyendo "
                f"{ruta.name}: {error}"
            )

    return documentos


def _cargar_csv() -> List[Document]:
    """
    Carga archivos CSV ubicados directamente
    en data/documentos.

    Cada CSV se considera un documento.
    """

    documentos = []

    if not DOCUMENTOS_DIR.exists():
        return documentos

    for ruta in sorted(
        DOCUMENTOS_DIR.glob("*.csv")
    ):

        try:

            texto = ruta.read_text(
                encoding="utf-8",
                errors="replace"
            )

            documento = Document(
                page_content=texto,
                metadata={
                    "source": str(ruta),
                    "filename": ruta.name,
                    "page": 0,
                    "file_type": "csv",
                },
            )

            documentos.append(
                documento
            )

            print(
                f"  📊 CSV cargado: "
                f"{ruta.name}"
            )

        except Exception as error:

            print(
                f"  ⚠️ Error leyendo "
                f"{ruta.name}: {error}"
            )

    return documentos


# ============================================================
# MOTOR RAG
# ============================================================

class MotorRAG:
    """
    Motor de Retrieval Augmented Generation de JOURNALI.

    Mantiene:

        paginas
            Documentos/páginas originales.

        fragmentos
            Fragmentos generados para recuperación.

    buscar()
        Devuelve contexto documental y fuentes.
    """

    def __init__(self):

        self.paginas: List[Document] = []

        self.fragmentos: List[Document] = []

        self.archivos: List[str] = []

        self._construir()


    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL ÍNDICE
    # --------------------------------------------------------

    def _construir(self):

        if not DOCUMENTOS_DIR.exists():

            print(
                f"  ⚠️ Carpeta de documentos "
                f"no encontrada: "
                f"{DOCUMENTOS_DIR}"
            )

            return

        # Cargar únicamente archivos directamente
        # ubicados en data/documentos.
        documentos = (
            _cargar_pdfs()
            + _cargar_csv()
        )

        self.paginas = documentos

        # Registrar archivos únicos.
        self.archivos = sorted(
            {
                documento.metadata.get(
                    "filename",
                    "desconocido"
                )
                for documento in documentos
            }
        )

        # ----------------------------------------------------
        # Fragmentación
        # ----------------------------------------------------

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=TAMANO_FRAGMENTO,
                chunk_overlap=SUPERPOSICION_FRAGMENTO,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    ", ",
                    " ",
                ],
            )
        )

        self.fragmentos = (
            splitter.split_documents(
                documentos
            )
        )

        print(
            "  📚 RAG JOURNALI listo: "
            f"{len(self.archivos)} archivo(s), "
            f"{len(self.paginas)} página(s), "
            f"{len(self.fragmentos)} fragmentos"
        )


    # --------------------------------------------------------
    # BÚSQUEDA
    # --------------------------------------------------------

    def buscar(
        self,
        pregunta: str,
        k: int = K_FRAGMENTOS
    ) -> Tuple[str, List[str]]:
        """
        Busca evidencia relevante.

        Retorna:

            (
                contexto_documental,
                fuentes
            )
        """

        if not pregunta:
            return "", []

        if not self.fragmentos:
            return "", []

        palabras_pregunta = (
            palabras_clave(
                pregunta
            )
        )

        if not palabras_pregunta:
            return "", []

        # ----------------------------------------------------
        # Primera capa:
        # coincidencia de palabras en fragmentos
        # ----------------------------------------------------

        puntuados = []

        for fragmento in self.fragmentos:

            palabras_fragmento = (
                palabras_clave(
                    fragmento.page_content
                )
            )

            coincidencias = (
                palabras_pregunta
                & palabras_fragmento
            )

            if not coincidencias:
                continue

            # Puntuación básica.
            score = len(
                coincidencias
            )

            # Bonus por cobertura.
            cobertura = (
                len(coincidencias)
                / max(
                    len(palabras_pregunta),
                    1
                )
            )

            score_final = (
                score
                + cobertura
            )

            puntuados.append(
                (
                    score_final,
                    fragmento
                )
            )

        puntuados.sort(
            key=lambda elemento: elemento[0],
            reverse=True
        )

        # ----------------------------------------------------
        # Selección evitando duplicados
        # ----------------------------------------------------

        elegidos = {}

        for score, fragmento in puntuados:

            clave = (
                fragmento.metadata.get(
                    "source"
                ),
                fragmento.metadata.get(
                    "page",
                    0
                ),
            )

            if clave not in elegidos:

                elegidos[clave] = (
                    score,
                    fragmento
                )

            if len(elegidos) >= k:

                break

        # ----------------------------------------------------
        # Segunda capa:
        # respaldo sobre páginas completas
        # ----------------------------------------------------

        umbral = max(
            2,
            len(palabras_pregunta) // 2
        )

        for pagina in self.paginas:

            palabras_pagina = (
                palabras_clave(
                    pagina.page_content
                )
            )

            coincidencias = (
                palabras_pregunta
                & palabras_pagina
            )

            if len(coincidencias) < umbral:
                continue

            clave = (
                pagina.metadata.get(
                    "source"
                ),
                pagina.metadata.get(
                    "page",
                    0
                ),
            )

            if clave not in elegidos:

                elegidos[clave] = (
                    float(
                        len(coincidencias)
                    ),
                    pagina
                )

            if len(elegidos) >= k + 2:

                break

        # ----------------------------------------------------
        # Construcción del contexto
        # ----------------------------------------------------

        if not elegidos:
            return "", []

        seleccionados = list(
            elegidos.values()
        )

        seleccionados.sort(
            key=lambda elemento: elemento[0],
            reverse=True
        )

        contexto_partes = []

        fuentes = []

        for _, documento in seleccionados[
            : k + 2
        ]:

            archivo = os.path.basename(
                str(
                    documento.metadata.get(
                        "source",
                        "desconocido"
                    )
                )
            )

            pagina = documento.metadata.get(
                "page",
                0
            )

            contenido = (
                documento.page_content
                .strip()
            )

            if not contenido:
                continue

            contexto_partes.append(
                (
                    f"[FUENTE: {archivo} | "
                    f"PÁGINA: {pagina}]\n"
                    f"{contenido}"
                )
            )

            fuente = (
                f"{archivo} "
                f"(pág. {pagina})"
            )

            if fuente not in fuentes:
                fuentes.append(
                    fuente
                )

        contexto = (
            "\n\n"
            .join(
                contexto_partes
            )
        )

        return contexto, fuentes


# ============================================================
# INSTANCIA ÚNICA
# ============================================================

_motor: MotorRAG | None = None


def obtener_motor() -> MotorRAG:
    """
    Devuelve la instancia única del motor RAG.
    """

    global _motor

    if _motor is None:

        _motor = MotorRAG()

    return _motor


# ============================================================
# PRUEBA DIRECTA
# ============================================================

if __name__ == "__main__":

    motor = obtener_motor()

    print()
    print("========== JOURNALI RAG ==========")
    print(
        "Archivos:",
        len(motor.archivos)
    )
    print(
        "Páginas:",
        len(motor.paginas)
    )
    print(
        "Fragmentos:",
        len(motor.fragmentos)
    )

    if motor.archivos:

        print()
        print("ARCHIVOS CARGADOS:")

        for archivo in motor.archivos:
            print(
                f" - {archivo}"
            )

    print()
    print("===================================")