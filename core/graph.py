# ============================================================
# JOURNALI
# Agente inteligente de apoyo a la investigación científica
# ============================================================

import os
import re
from typing import TypedDict, Literal

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

from core.rag import obtener_motor
from s6_llm_factory import crear_llm


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "JOURNALI")
ROL_AGENTE = os.getenv(
    "AGENT_ROLE",
    "agente inteligente de apoyo a la investigación científica",
)
DOMINIO_AGENTE = os.getenv(
    "AGENT_DOMAIN",
    "investigación científica y publicación académica",
)


# ============================================================
# ESTADO DEL AGENTE
# ============================================================

class EstadoAgente(TypedDict):
    entrada_usuario: str
    historial: list
    categoria: Literal[
        "CONVERSACION",
        "DOCUMENTOS",
        "ANALISIS",
        "REPORTE",
    ]
    respuesta_final: str
    fuentes: list


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_pensamiento(texto: str) -> str:
    """Elimina posibles bloques de razonamiento del modelo."""

    if texto is None:
        return ""

    texto = str(texto)

    patrones = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
    ]

    for patron in patrones:
        texto = re.sub(
            patron,
            "",
            texto,
            flags=re.DOTALL | re.IGNORECASE,
        )

    return texto.strip()


def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""

    return texto.lower().strip()


def construir_contexto_conversacional(estado: EstadoAgente) -> str:
    """
    Construye contexto con las últimas interacciones para resolver
    preguntas de seguimiento como:
    '¿Qué tecnologías aparecen relacionadas con ese tema?'
    """

    historial = estado.get("historial", [])

    if not historial:
        return ""

    partes = []

    for mensaje in historial[-6:]:
        try:
            tipo = getattr(mensaje, "type", "")
            contenido = getattr(mensaje, "content", "")

            if not contenido:
                continue

            if tipo == "human":
                rol = "Usuario"
            elif tipo == "ai":
                rol = "JOURNALI"
            else:
                rol = str(tipo or "Mensaje")

            contenido = str(contenido).strip()

            # Evitar enviar respuestas gigantes al modelo.
            if len(contenido) > 2500:
                contenido = contenido[:2500] + "..."

            partes.append(f"{rol}: {contenido}")

        except Exception:
            continue

    if not partes:
        return ""

    return "\n".join(partes)


def construir_consulta_contextualizada(estado: EstadoAgente) -> str:
    """
    Combina el historial reciente con la nueva pregunta.
    Esto permite que el RAG interprete referencias como:
    'ese tema', 'lo anterior', 'esas tecnologías', etc.
    """

    pregunta = estado.get("entrada_usuario", "").strip()
    historial = construir_contexto_conversacional(estado)

    if not historial:
        return pregunta

    return (
        "CONTEXTO CONVERSACIONAL PREVIO:\n"
        f"{historial}\n\n"
        "NUEVA CONSULTA DEL INVESTIGADOR:\n"
        f"{pregunta}"
    )


def extraer_fuentes_unicas(fuentes: list) -> list:
    resultado = []
    vistas = set()

    for fuente in fuentes or []:
        clave = str(fuente).strip()

        if clave and clave not in vistas:
            vistas.add(clave)
            resultado.append(clave)

    return resultado


# ============================================================
# NODO: COORDINADOR
# ============================================================

def nodo_coordinador(estado: EstadoAgente) -> dict:

    pregunta = estado["entrada_usuario"]

    texto = normalizar_texto(pregunta)

    print("\n[JOURNALI / COORDINADOR] Clasificando consulta...")

    # --------------------------------------------------------
    # REGLAS DETERMINÍSTICAS
    # --------------------------------------------------------

    palabras_reporte = [
        "genera un informe",
        "generar un informe",
        "elabora un informe",
        "elaborar un informe",
        "realiza un informe",
        "realizar un informe",
        "reporte de investigación",
        "informe de investigación",
        "informe científico",
        "reporte científico",
        "genera un reporte",
        "generar un reporte",
        "elabora un reporte",
        "elaborar un reporte",
    ]

    palabras_documentos = [
        "en los documentos",
        "en el documento",
        "según los documentos",
        "según el documento",
        "en los artículos",
        "en el artículo",
        "busca en los documentos",
        "buscar en los documentos",
        "consulta los documentos",
        "consulta en los documentos",
        "qué dicen los documentos",
        "que dicen los documentos",
    ]

    for patron in palabras_reporte:
        if patron in texto:
            print("[COORDINADOR] → REPORTE")
            return {"categoria": "REPORTE"}

    for patron in palabras_documentos:
        if patron in texto:
            print("[COORDINADOR] → DOCUMENTOS")
            return {"categoria": "DOCUMENTOS"}

    # --------------------------------------------------------
    # CONSULTAS CIENTÍFICAS
    # --------------------------------------------------------

    palabras_cientificas = [
        "investigación",
        "investigacion",
        "investigar",
        "tendencias",
        "líneas de investigación",
        "lineas de investigacion",
        "metodología",
        "metodologia",
        "hallazgos",
        "literatura científica",
        "literatura cientifica",
        "artículos científicos",
        "articulos cientificos",
        "estado del arte",
        "vacíos de investigación",
        "vacios de investigacion",
        "revisión bibliográfica",
        "revision bibliografica",
        "revisión sistemática",
        "revision sistematica",
        "agricultura de precisión",
        "agricultura de precision",
        "iot",
        "internet de las cosas",
        "inteligencia artificial",
        "machine learning",
        "aprendizaje automático",
        "aprendizaje automatico",
        "digital twin",
        "digital twins",
        "sensores",
        "big data",
        "uav",
        "drones",
        "tecnología agrícola",
        "tecnologia agricola",
    ]

    if any(palabra in texto for palabra in palabras_cientificas):
        print("[COORDINADOR] → ANALISIS")
        return {"categoria": "ANALISIS"}

    # --------------------------------------------------------
    # CLASIFICACIÓN LLM COMO RESPALDO
    # --------------------------------------------------------

    print("[COORDINADOR] Usando clasificación LLM...")

    llm = crear_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres el coordinador de JOURNALI.

Clasifica la solicitud del usuario en UNA de las siguientes categorías:

CONVERSACION
DOCUMENTOS
ANALISIS
REPORTE

CRITERIOS:

CONVERSACION:
Saludo, conversación general o pregunta que no requiera investigación documental.

DOCUMENTOS:
La persona solicita consultar explícitamente los documentos disponibles.

ANALISIS:
Pregunta científica, conceptual, metodológica, sobre literatura,
tendencias, hallazgos, tecnologías, investigación o estado del arte.

REPORTE:
Solicita explícitamente elaborar un informe, reporte o documento estructurado.

RESPONDE ÚNICAMENTE con una categoría.
""",
            ),
            (
                "human",
                "{pregunta}",
            ),
        ]
    )

    cadena = prompt | llm | StrOutputParser()

    resultado = cadena.invoke({"pregunta": pregunta})

    resultado = str(resultado).strip().upper()

    categorias_validas = {
        "CONVERSACION",
        "DOCUMENTOS",
        "ANALISIS",
        "REPORTE",
    }

    if resultado not in categorias_validas:
        resultado = "CONVERSACION"

    print(f"[COORDINADOR] → {resultado}")

    return {"categoria": resultado}


# ============================================================
# NODO: CONVERSACIÓN
# ============================================================

def nodo_conversacion(estado: EstadoAgente) -> dict:

    print("[JOURNALI / CONVERSACIÓN] Procesando conversación...")

    llm = crear_llm(temperature=0.3)

    historial = construir_contexto_conversacional(estado)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
Eres {NOMBRE_AGENTE}.

Rol:
{ROL_AGENTE}

Dominio:
{DOMINIO_AGENTE}

Responde de manera clara, profesional y útil.

Puedes utilizar el contexto conversacional previo para comprender
referencias del usuario.

Si la solicitud requiere consultar literatura científica o documentos,
no inventes información.

Responde en español.
""",
            ),
            (
                "human",
                """
CONTEXTO CONVERSACIONAL:
{historial}

PREGUNTA ACTUAL:
{pregunta}
""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    respuesta = chain.invoke(
        {
            "historial": historial or "No existe contexto previo.",
            "pregunta": estado["entrada_usuario"],
        }
    )

    return {
        "respuesta_final": limpiar_pensamiento(respuesta),
        "fuentes": [],
    }


# ============================================================
# NODO: DOCUMENTOS / RAG
# ============================================================

def nodo_documentos(estado: EstadoAgente) -> dict:

    print("[JOURNALI / RAG] Buscando en documentos...")

    motor = obtener_motor()

    consulta = construir_consulta_contextualizada(estado)

    contexto, fuentes = motor.buscar(consulta)

    fuentes = extraer_fuentes_unicas(fuentes)

    if not contexto:
        return {
            "respuesta_final": (
                "No se encontró información relevante en los documentos "
                "disponibles para responder esta consulta."
            ),
            "fuentes": [],
        }

    llm = crear_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
Eres {NOMBRE_AGENTE}, {ROL_AGENTE}.

Estás respondiendo una consulta utilizando exclusivamente
el contexto documental recuperado por el sistema RAG.

REGLAS:

1. No inventes información.
2. No atribuyas a los documentos información que no aparezca en ellos.
3. No utilices conocimiento externo para completar vacíos.
4. Si el contexto es insuficiente, dilo.
5. Diferencia evidencia de interpretación.
6. Conserva la trazabilidad documental.
7. No inventes referencias bibliográficas.
8. Responde en español.
9. Utiliza redacción académica y clara.

El historial sirve únicamente para comprender referencias
como 'ese tema', 'lo anterior' o 'esas tecnologías'.
La evidencia válida debe provenir del contexto documental.
""",
            ),
            (
                "human",
                """
CONTEXTO DOCUMENTAL:

{contexto}

PREGUNTA ACTUAL:

{pregunta}

Responde exclusivamente con base en el contexto documental.
""",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    respuesta = chain.invoke(
        {
            "contexto": contexto,
            "pregunta": estado["entrada_usuario"],
        }
    )

    return {
        "respuesta_final": limpiar_pensamiento(respuesta),
        "fuentes": fuentes,
    }


# ============================================================
# NODO: ANÁLISIS CIENTÍFICO
# ============================================================

def nodo_analisis(estado: EstadoAgente) -> dict:

    print("[JOURNALI / ANÁLISIS] Iniciando análisis científico...")

    motor = obtener_motor()

    consulta = construir_consulta_contextualizada(estado)

    contexto, fuentes = motor.buscar(consulta)

    fuentes = extraer_fuentes_unicas(fuentes)

    if not contexto:
        return {
            "respuesta_final": (
                "No se encontró información documental suficiente en el "
                "corpus de JOURNALI para responder esta consulta con "
                "evidencia recuperada."
            ),
            "fuentes": [],
        }

    llm = crear_llm(temperature=0.2)

    # --------------------------------------------------------
    # ANÁLISIS
    # --------------------------------------------------------

    prompt_analisis = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
Eres {NOMBRE_AGENTE}.

Rol:
{ROL_AGENTE}

Dominio:
{DOMINIO_AGENTE}

Realizas análisis científico utilizando EXCLUSIVAMENTE
la evidencia documental recuperada por JOURNALI.

REGLAS OBLIGATORIAS:

1. Utiliza exclusivamente la información contenida en el contexto.
2. No inventes autores.
3. No inventes artículos.
4. No inventes años.
5. No inventes cifras.
6. No inventes metodologías.
7. No inventes resultados.
8. No inventes referencias bibliográficas.
9. No utilices conocimiento externo.
10. Si una afirmación no está sustentada por el contexto,
    no la presentes como un hecho.
11. Diferencia evidencia, interpretación y vacío de información.
12. Cuando sea posible, relaciona hallazgos con [FUENTE] y [PÁGINA].
13. Si la evidencia es insuficiente, indícalo.
14. Responde en español.
15. Mantén lenguaje académico y preciso.
""",
            ),
            (
                "human",
                """
PREGUNTA DEL INVESTIGADOR:

{pregunta}

==================================================
CONTEXTO DOCUMENTAL RECUPERADO
==================================================

{contexto}

==================================================

Realiza el análisis científico.

Organiza la respuesta en:

1. Hallazgos principales.
2. Conceptos y tecnologías relevantes.
3. Relaciones y patrones.
4. Tendencias identificadas.
5. Vacíos o limitaciones.
6. Posibles líneas de investigación.

Todos los puntos deben derivarse del contexto documental.
""",
            ),
        ]
    )

    cadena_analisis = prompt_analisis | llm | StrOutputParser()

    analisis = cadena_analisis.invoke(
        {
            "pregunta": estado["entrada_usuario"],
            "contexto": contexto,
        }
    )

    analisis = limpiar_pensamiento(analisis)

    # --------------------------------------------------------
    # SÍNTESIS
    # --------------------------------------------------------

    print("[JOURNALI / SINTETIZADOR] Construyendo respuesta...")

    prompt_sintesis = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
Eres {NOMBRE_AGENTE}.

Construye la respuesta final de una consulta científica utilizando
únicamente:

1. El contexto documental recuperado.
2. El análisis científico derivado de dicho contexto.

REGLAS:

1. No agregues información externa.
2. No inventes autores.
3. No inventes estudios.
4. No inventes referencias.
5. No inventes cifras.
6. No conviertas interpretaciones en hechos.
7. Si existe incertidumbre, indícala.
8. Conserva la trazabilidad de las fuentes.
9. Responde en español.
10. Utiliza estructura académica clara.

La prioridad es la fidelidad a la evidencia.
""",
            ),
            (
                "human",
                """
PREGUNTA:

{pregunta}

==================================================
CONTEXTO DOCUMENTAL
==================================================

{contexto}

==================================================
ANÁLISIS
==================================================

{analisis}

==================================================

Construye la respuesta final.

Incluye cuando corresponda:

- respuesta directa;
- hallazgos;
- conceptos y tecnologías;
- relaciones y patrones;
- tendencias;
- vacíos;
- líneas de investigación.

No agregues información que no esté sustentada.
""",
            ),
        ]
    )

    cadena_sintesis = prompt_sintesis | llm | StrOutputParser()

    respuesta = cadena_sintesis.invoke(
        {
            "pregunta": estado["entrada_usuario"],
            "contexto": contexto,
            "analisis": analisis,
        }
    )

    return {
        "respuesta_final": limpiar_pensamiento(respuesta),
        "fuentes": fuentes,
    }


# ============================================================
# NODO: REPORTE ESTRUCTURADO
# ============================================================

def nodo_reporte(estado: EstadoAgente) -> dict:

    print("[JOURNALI / REPORTE] Generando reporte estructurado...")

    motor = obtener_motor()

    consulta = construir_consulta_contextualizada(estado)

    contexto, fuentes = motor.buscar(consulta)

    fuentes = extraer_fuentes_unicas(fuentes)

    if not contexto:
        return {
            "respuesta_final": (
                "No se encontró evidencia documental suficiente "
                "para generar el reporte."
            ),
            "fuentes": [],
        }

    llm = crear_llm(temperature=0.2)

    from core.structured import ReporteInvestigacion

    # --------------------------------------------------------
    # PRIMER INTENTO: SALIDA ESTRUCTURADA
    # --------------------------------------------------------

    try:

        llm_estructurado = llm.with_structured_output(
            ReporteInvestigacion
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Eres JOURNALI, agente inteligente de apoyo a la investigación científica.

Genera un reporte estructurado utilizando EXCLUSIVAMENTE
la evidencia documental proporcionada.

REGLAS:

- No inventes información.
- No inventes referencias.
- No inventes autores.
- No inventes resultados.
- No uses conocimiento externo.
- Si un aspecto no está sustentado, indícalo como vacío o limitación.
- Mantén lenguaje académico.

IMPORTANTE:

El campo hallazgos es obligatorio.

Debes generar entre 3 y 6 hallazgos concretos.

Cada hallazgo debe contener:

- descripcion: qué se identifica.
- evidencia: qué evidencia del contexto lo sustenta.
- fuente: archivo y/o página cuando estén disponibles.

No dejes hallazgos vacío si existe evidencia documental relevante.
""",
                ),
                (
                    "human",
                    """
PREGUNTA:

{pregunta}

==================================================
EVIDENCIA DOCUMENTAL
==================================================

{contexto}

==================================================

Genera el reporte estructurado.
""",
                ),
            ]
        )

        cadena = prompt | llm_estructurado

        reporte = cadena.invoke(
            {
                "pregunta": estado["entrada_usuario"],
                "contexto": contexto,
            }
        )

        respuesta = reporte.model_dump()

        # Si el modelo devuelve hallazgos, usamos la salida estructurada.
        hallazgos = respuesta.get("hallazgos", [])

        if hallazgos:

            partes = []

            partes.append(
                f"# {respuesta.get('tema', 'Reporte de investigación')}"
            )

            contexto_reporte = respuesta.get("contexto", "")
            if contexto_reporte:
                partes.append(
                    f"\n## Contexto\n{contexto_reporte}"
                )

            conceptos = respuesta.get(
                "conceptos_principales",
                [],
            )

            if conceptos:
                partes.append("\n## Conceptos principales")
                for item in conceptos:
                    partes.append(f"- {item}")

            partes.append("\n## Hallazgos")

            for hallazgo in hallazgos:

                if hasattr(hallazgo, "model_dump"):
                    hallazgo = hallazgo.model_dump()

                descripcion = hallazgo.get(
                    "descripcion",
                    "",
                )

                evidencia = hallazgo.get(
                    "evidencia",
                    "",
                )

                fuente = hallazgo.get(
                    "fuente",
                    "",
                )

                partes.append(
                    f"\n### {descripcion}\n"
                    f"Evidencia: {evidencia}\n"
                    f"Fuente: {fuente}"
                )

            tendencias = respuesta.get(
                "tendencias",
                [],
            )

            if tendencias:
                partes.append("\n## Tendencias")
                for item in tendencias:
                    partes.append(f"- {item}")

            vacios = respuesta.get(
                "vacios_investigacion",
                [],
            )

            if vacios:
                partes.append("\n## Vacíos de investigación")
                for item in vacios:
                    partes.append(f"- {item}")

            lineas = respuesta.get(
                "lineas_investigacion",
                [],
            )

            if lineas:
                partes.append("\n## Líneas de investigación")
                for item in lineas:
                    partes.append(f"- {item}")

            conclusion = respuesta.get(
                "conclusion",
                "",
            )

            if conclusion:
                partes.append(
                    f"\n## Conclusión\n{conclusion}"
                )

            return {
                "respuesta_final": "\n".join(partes),
                "fuentes": fuentes,
            }

    except Exception as error:

        print(
            f"[REPORTE] Salida estructurada no disponible: {error}"
        )

    # --------------------------------------------------------
    # FALLBACK ROBUSTO: REPORTE MARKDOWN
    # --------------------------------------------------------

    print("[REPORTE] Utilizando generación Markdown de respaldo...")

    prompt_fallback = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres JOURNALI, agente inteligente de apoyo a la investigación científica.

Genera un reporte académico exclusivamente a partir de la evidencia
documental suministrada.

NO utilices conocimiento externo.

El reporte debe contener exactamente estas secciones:

# Tema

## Contexto

## Conceptos principales

## Hallazgos

Incluye entre 3 y 6 hallazgos.
Cada hallazgo debe indicar:
- qué se encontró;
- evidencia documental;
- fuente y página cuando estén disponibles.

## Tendencias

## Vacíos de investigación

## Líneas de investigación

## Conclusión

No inventes autores, artículos, cifras, resultados ni referencias.
Si algo no puede determinarse a partir del corpus, indícalo.
""",
            ),
            (
                "human",
                """
SOLICITUD DEL INVESTIGADOR:

{pregunta}

==================================================
CORPUS DOCUMENTAL
==================================================

{contexto}

==================================================

Genera el reporte.
""",
            ),
        ]
    )

    cadena_fallback = prompt_fallback | llm | StrOutputParser()

    respuesta_fallback = cadena_fallback.invoke(
        {
            "pregunta": estado["entrada_usuario"],
            "contexto": contexto,
        }
    )

    return {
        "respuesta_final": limpiar_pensamiento(
            respuesta_fallback
        ),
        "fuentes": fuentes,
    }


# ============================================================
# ROUTING
# ============================================================

def enrutar_categoria(estado: EstadoAgente) -> str:

    categoria = estado.get(
        "categoria",
        "CONVERSACION",
    )

    mapa = {
        "CONVERSACION": "conversacion",
        "DOCUMENTOS": "documentos",
        "ANALISIS": "analisis",
        "REPORTE": "reporte",
    }

    destino = mapa.get(
        categoria,
        "conversacion",
    )

    print(
        f"[JOURNALI / ROUTER] {categoria} → {destino}"
    )

    return destino


# ============================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================

def construir_grafo():

    print("[JOURNALI] Construyendo grafo...")

    grafo = StateGraph(EstadoAgente)

    grafo.add_node(
        "coordinador",
        nodo_coordinador,
    )

    grafo.add_node(
        "conversacion",
        nodo_conversacion,
    )

    grafo.add_node(
        "documentos",
        nodo_documentos,
    )

    grafo.add_node(
        "analisis",
        nodo_analisis,
    )

    grafo.add_node(
        "reporte",
        nodo_reporte,
    )

    grafo.add_edge(
        START,
        "coordinador",
    )

    grafo.add_conditional_edges(
        "coordinador",
        enrutar_categoria,
        {
            "conversacion": "conversacion",
            "documentos": "documentos",
            "analisis": "analisis",
            "reporte": "reporte",
        },
    )

    grafo.add_edge(
        "conversacion",
        END,
    )

    grafo.add_edge(
        "documentos",
        END,
    )

    grafo.add_edge(
        "analisis",
        END,
    )

    grafo.add_edge(
        "reporte",
        END,
    )

    app = grafo.compile()

    print("[JOURNALI] Grafo compilado correctamente.")

    return app


# ============================================================
# PRUEBA DIRECTA
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "==================================================\n"
        "JOURNALI - PRUEBA DEL GRAFO\n"
        "==================================================\n"
    )

    grafo = construir_grafo()

    estado_inicial = {
        "entrada_usuario": (
            "¿Qué tendencias de investigación se identifican "
            "en agricultura de precisión mediante IoT, "
            "inteligencia artificial y sensores?"
        ),
        "historial": [],
        "categoria": "CONVERSACION",
        "respuesta_final": "",
        "fuentes": [],
    }

    resultado = grafo.invoke(estado_inicial)

    print(
        "\n=================================================="
    )

    print("\nRESPUESTA:\n")

    print(
        resultado.get(
            "respuesta_final",
            "",
        )
    )

    print("\nFUENTES:\n")

    for fuente in resultado.get("fuentes", []):
        print(f"- {fuente}")

    print(
        "\n=================================================="
    )