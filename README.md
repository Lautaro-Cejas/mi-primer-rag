# 📄 Chat with PDF (RAG Pipeline)

Este proyecto es una aplicación de Inteligencia Artificial que permite a los usuarios "chatear" con sus propios documentos PDF. Utiliza la técnica **RAG (Retrieval-Augmented Generation)** para buscar información relevante dentro del documento y generar respuestas precisas.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange)

## 🚀 Características

- **Carga Dinámica:** Sube cualquier archivo PDF desde la interfaz web.
- **Búsqueda Vectorial:** Utiliza **FAISS** (CPU) para indexar y buscar fragmentos relevantes del texto localmente.
- **LLM de Alta Velocidad:** Integración con **Groq** (modelo `llama-3.1-8b-instant`) para respuestas en milisegundos.
- **Memoria de Conversación:** El chat mantiene el contexto de las preguntas anteriores.
- **Privacidad:** Los embeddings se generan localmente con HuggingFace y los archivos no se almacenan permanentemente.

## 🛠️ Stack Tecnológico

* **Lenguaje:** Python
* **Orquestador:** LangChain
* **Frontend:** Streamlit
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **LLM:** Groq API (`llama-3.1-8b-instant`)

## ⚙️ Instalación y Uso

Sigue estos pasos para correr el proyecto en tu máquina local:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Lautaro-Cejas/mi-primer-rag.git](https://github.com/Lautaro-Cejas/mi-primer-rag.git)
    cd mi-primer-rag
    ```

2.  **Crear un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # En Mac/Linux: source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar Variables de Entorno:**
    * El proyecto incluye un archivo de ejemplo `.env.example`.
    * Crea un archivo `.env` basado en él y agrega tu API Key de Groq:
    ```bash
    cp .env.example .env
    ```
    * Abre el archivo `.env` y pega tu clave (consíguela en [console.groq.com](https://console.groq.com/keys)):
    ```env
    GROQ_API_KEY=gsk_tu_clave_secreta_aqui...
    ```

5.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```

## 📂 Estructura del Proyecto

```text
├── venv/               # Entorno virtual (no se sube a GitHub)
├── .env                # Tus claves reales (Ignorado por Git)
├── .env.example        # Plantilla de variables de entorno (Público)
├── .gitignore          # Archivos ignorados (venv, .env, archivos temporales)
├── app.py              # Aplicación principal (Interfaz Streamlit)
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación