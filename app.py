import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA  # <--- Usamos la clase clásica y robusta
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Chat con tu PDF", page_icon="📄")
st.title("📄 Chat con tu PDF (Dinámico)")

# Cargar variables de entorno
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ No se encontró la API Key de Groq. Revisa tu archivo .env")
    st.stop()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Configuración")
    uploaded_file = st.file_uploader("Sube tu PDF aquí", type="pdf")
    st.markdown("---")
    st.markdown("🔹 **Modelo:** Llama-3.1-8b-instant")

# --- PASO 1: CACHEAR SOLO LO PESADO (VectorStore) ---
# Separamos la creación de la base de datos para que Streamlit no se rompa con el LLM
@st.cache_resource
def crear_base_vectorial(pdf_path):
    # 1. Cargar
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Dividir
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. VectorStore
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- PASO 2: CREAR LA CADENA (Al vuelo) ---
def crear_cadena_rag(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Prompt personalizado (Adaptado para RetrievalQA)
    template = """
    Eres un asistente experto. Usa el siguiente contexto para responder la pregunta al final.
    Si no sabes la respuesta, di simplemente que no lo sabes.
    
    <context>
    {context}
    </context>
    
    Pregunta: {question}
    Respuesta:
    """
    prompt = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- LÓGICA PRINCIPAL ---

if uploaded_file is not None:
    # Guardar temp
    temp_filepath = "temp_uploaded.pdf"
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    # 1. Crear/Cargar Base Vectorial (Esto sí se cachea)
    with st.spinner("🔄 Indexando documento..."):
        try:
            vectorstore = crear_base_vectorial(temp_filepath)
            st.success("✅ Documento listo.")
        except Exception as e:
            st.error(f"Error al procesar el PDF: {e}")
            st.stop()

    # 2. Crear Cadena (Esto se hace fresco para evitar el error de NoneType)
    rag_chain = crear_cadena_rag(vectorstore)

    # 3. Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Input Usuario
    if prompt := st.chat_input("Escribe tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando..."):
                # RetrievalQA usa "query" en lugar de "input"
                response = rag_chain.invoke({"query": prompt}) 
                respuesta = response["result"]
                st.markdown(respuesta)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})

else:
    st.info("👋 Sube un PDF para comenzar.")