import streamlit as st
import os
from dotenv import load_dotenv
import tempfile # For creating temporary files reliably
import pandas as pd # For basic CSV/Excel preview if needed (though not directly used for loading into LangChain)

# --- Configuration ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Hardcoded System Prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

# Hugging Face Embedding Model (CPU friendly)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text Splitting parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Groq Model
GROQ_MODEL_NAME = "llama3-8b-8192"

# --- Import necessary Langchain and other modules ---
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
# For Excel and Images (and other unstructured data), we use UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Streamlit App ---
st.set_page_config(page_title="Multi-Document Q&A AI Agent", layout="wide")

st.title("📚 Multi-Document Q&A AI Agent")

st.write(
    """
    This application allows you to chat with your documents.
    Upload PDF, Excel (.xlsx), CSV files as your knowledge base.
    """
)

# --- Check for Groq API Key ---
if not groq_api_key:
    st.error("Error: **GROQ_API_KEY** environment variable not found.")
    st.info("Please set the `GROQ_API_KEY` environment variable on your system or in a `.env` file.")
    st.stop() # Stop the app if API key is missing

# --- File Upload ---
st.subheader("Upload Knowledge Document(s)")
uploaded_files = st.file_uploader(
    "Choose one or more documents (PDF, Excel, CSV, Image)",
    type=["pdf", "xlsx", "xls", "csv", "png", "jpg", "jpeg"], # Allowed file types
    accept_multiple_files=True,
    help="Supported formats: PDF, Excel (.xlsx, .xls), CSV, and Image files (.png, .jpg, .jpeg). For images, Tesseract OCR must be installed on the system."
)

# --- Process Documents and Setup RAG ---
@st.cache_resource(show_spinner="Setting up Agent with your documents...")
def setup_rag_system(files, groq_key, system_prompt_tpl):
    if not files:
        return None, "Please upload documents to proceed."

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    all_docs = []
    
    # Use tempfile.TemporaryDirectory to manage temporary files securely
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            
            # Write uploaded file to a temporary location
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.markdown(f"**Loading**: {uploaded_file.name}...") # Informative message during processing
            try:
                docs_from_file = []
                if file_extension == ".pdf":
                    loader = PyMuPDFLoader(temp_filepath)
                    docs_from_file = loader.load()
                elif file_extension == ".csv":
                    loader = CSVLoader(temp_filepath)
                    docs_from_file = loader.load()
                elif file_extension in [".xlsx", ".xls"]:
                    # UnstructuredFileLoader can handle Excel files.
                    # It requires 'openpyxl' and other unstructured dependencies.
                    loader = UnstructuredFileLoader(temp_filepath)
                    docs_from_file = loader.load()
                elif file_extension in [".png", ".jpg", ".jpeg"]:
                    # UnstructuredFileLoader can perform OCR on images.
                    # This requires Tesseract OCR engine installed on the system.
                    loader = UnstructuredFileLoader(temp_filepath, mode="elements") # mode="elements" helps with richer parsing
                    docs_from_file = loader.load()
                    if not docs_from_file:
                        st.warning(f"No text extracted from image: **{uploaded_file.name}**. Ensure Tesseract OCR is installed and the image contains readable text.")
                else:
                    st.warning(f"Unsupported file type for **{uploaded_file.name}**: {file_extension}. Skipping.")
                    continue # Skip to next file

                if not docs_from_file:
                    st.warning(f"Warning: Document '**{uploaded_file.name}**' loaded but contains no content. Skipping.")
                    continue
                all_docs.extend(docs_from_file)
                st.markdown(f"**Loaded**: {uploaded_file.name} (extracted {len(docs_from_file)} parts)")

            except Exception as e:
                st.error(f"Error processing **{uploaded_file.name}**: {e}")
                # Don't return None here, try to process other files
                continue # Skip to next file

    if not all_docs:
        return None, "No valid content could be extracted from the uploaded documents. Please check file formats and content."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(all_docs)

    try:
        vectorstore = FAISS.from_documents(
            documents=all_splits,
            embedding=embeddings
        )
    except Exception as e:
        return None, f"Error creating vector store: {e}"

    retriever = vectorstore.as_retriever()

    try:
        llm = ChatGroq(
            api_key=groq_key,
            model_name=GROQ_MODEL_NAME
        )
    except Exception as e:
        return None, f"Error initializing Groq LLM. Check your API key: {e}"

    prompt = ChatPromptTemplate.from_template(system_prompt_tpl)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, None

rag_chain, setup_error = None, None
if uploaded_files:
    rag_chain, setup_error = setup_rag_system(uploaded_files, groq_api_key, DEFAULT_SYSTEM_PROMPT)

if setup_error:
    st.error(setup_error)

# --- Chat Interface ---
if rag_chain:
    st.subheader("Ask a Question to your Documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                response = rag_chain.invoke(prompt)
                full_response = response
            except Exception as e:
                full_response = f"An error occurred: {e}"
                if "groq.exceptions" in str(e):
                    full_response += "\n\nPotential Groq API error. Check your API key, model name, or account usage."
            
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
elif not uploaded_files:
    st.info("Please upload one or more documents (PDF, Excel, CSV, Image) to start chatting.")

st.sidebar.markdown("---")
st.sidebar.info("The FAISS index is in-memory and will be cleared when you close or refresh the tab.")
