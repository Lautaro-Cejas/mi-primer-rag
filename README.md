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
    git clone [https://github.com/TU_USUARIO/mi-primer-rag.git](https://github.com/TU_USUARIO/mi-primer-rag.git)
    cd mi-primer-rag
    ```

2.  **Crear un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar Variables de Entorno:**
    * Crea un archivo `.env` en la raíz del proyecto.
    * Agrega tu API Key de Groq (consíguela en [console.groq.com](https://console.groq.com/keys)):
    ```env
    GROQ_API_KEY=gsk_tu_clave_secreta_aqui...
    ```

5.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```

## 📂 Estructura del Proyecto

```text
├── .env                # Variables de entorno (API Keys) - NO SUBIR A GITHUB
├── .gitignore          # Archivos ignorados por Git
├── app.py              # Aplicación principal (Streamlit)
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación