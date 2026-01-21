import os
from dotenv import load_dotenv

# 1. Cargar variables de entorno
load_dotenv()

# Importaciones
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def main():
    print("🤖 Iniciando el RAG con FAISS...")

    # Cargar el PDF
    pdf_path = "data/CVCejasLautaro-v2.0.0.pdf" 
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No encontré el archivo en {pdf_path}")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ PDF cargado: {len(docs)} páginas.")

    # Cortar el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ Documento dividido en {len(splits)} fragmentos.")

    # Embeddings
    print("⏳ Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Base de datos vectorial con FAISS
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    print("✅ Base de datos vectorial (FAISS) creada.")

    # Configurar el modelo LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0
    )

    # Configurar el chain de RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # --- Loop de Chat ---
    print("\n💬 ¡Listo! Pregúntale algo a tu PDF (escribe 'salir' para terminar).")
    while True:
        query = input("\nTu pregunta: ")
        if query.lower() in ["salir", "exit", "chau"]:
            break
        
        response = qa_chain.invoke({"query": query})
        print(f"🤖 Respuesta: {response['result']}")

if __name__ == "__main__":
    main()