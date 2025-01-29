# Streamlit Web Application
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
import asyncio
import os
import ollama
import logging
from typing import List
from agents.orchestrator import OrchestratorAgent
import traceback

EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./chroma_db"
VECTOR_STORE_NAME = "medical-info-rag"
DOC_PATH = "./docs/"
PDF_DOCS = ['2023-Directory-Diagnosis-Categories-Guide.pdf', 'guideline-170-en.pdf']

logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Medical Agency",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
        .stProgress .st-bo {
            background-color: #00a0dc;
        }
        .success-text {
            color: #00c853;
        }
        .warning-text {
            color: #ffd700;
        }
        .error-text {
            color: #ff5252;
        }
        .st-emotion-cache-1v0mbdj.e115fcil1 {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

async def search_documents(vector_db: Chroma, query: str, k: int = 6) -> List[str]:
    """Get relevant documents from vector_db for RAG to send along to agents"""
    docs = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

async def invoke_agents(vector_db: Chroma, query: str) -> dict:
    """Process symptoms through the AI agent pipeline"""
    try:
        relevant_docs = await search_documents(vector_db, query)
        context = "\n\n".join(relevant_docs)
        orchestrator = OrchestratorAgent()
        return await orchestrator.process_symptoms(
            st.session_state.selected_ollama_model.model,
            context,
            query
        )
    except Exception as e:
        logging.error(f"Error processing symptoms: {str(e)}")
        print(traceback.format_exc())
        raise

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF documents
        chunks = []
        for pdf in PDF_DOCS:
            data = ingest_pdf(DOC_PATH + pdf)
            if data is None:
                return None

            # Split the documents into chunks
            chunks.extend(split_documents(data))

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db


def main():
    st.title("AI Medical Agency")
    st.markdown(
        "Type your symptoms to get a diagnosis and prescription."
    )

    with st.sidebar:
        valid_models = [model for model in list(ollama.list()['models']) if 'embed' not in model.model]
        st.selectbox(
            "Selected model:",
            valid_models,
            format_func=lambda s: s.model,
            key="selected_ollama_model",
        )
    
    with st.spinner("Generating vector database..."):
        try:
            # Load the vector database
            vector_db = load_vector_db()
            if vector_db is None:
                st.error("Failed to load or create the vector database.")
                return
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(traceback.format_exc())

    user_input = st.text_input("Enter your symptoms:", "")

    if user_input:
        with st.spinner("Generating response..."):
            # Create placeholder for progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
    
            try:
                status_text.text("Analyzing symptoms...")
                progress_bar.progress(25)

                # Run analysis asynchronously
                result = asyncio.run(invoke_agents(vector_db, user_input))
                if result["status"] == "completed":
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    # Display results in tabs
                    tab1, tab2 = st.tabs(
                        [
                            "🎯 Diagnosis",
                            "💡 Prescription",
                        ]
                    )
                    with tab1:
                        st.subheader("Diagnosis")
                        st.write(result["diagnosis_data"]["diagnosis"])

                    with tab2:
                        st.subheader("Prescription")
                        st.write(result["prescription_data"]["prescription"])
                else:
                    st.error(
                        f"Process failed at stage: {result['current_stage']}\n"
                        f"Error: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(traceback.format_exc())
    else:
        st.info("Please enter a list of symptoms to get started.")

if __name__ == "__main__":
    main()
