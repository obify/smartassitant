import streamlit as st
import os
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file (for local development)
# In production, ensure GROQ_API_KEY is set in your environment
load_dotenv()

# Get Groq API key from environment variable
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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Streamlit App ---
st.set_page_config(page_title="PDF RAG System", layout="wide")

st.title("📄 PDF RAG System with Groq & Streamlit")

st.write(
    """
    This application allows you to chat with your PDF documents using a Retrieval-Augmented Generation (RAG) system.
    Upload one or more PDF files and start asking questions!
    """
)

# --- Check for Groq API Key ---
if not groq_api_key:
    st.error("Error: **GROQ_API_KEY** environment variable not found.")
    st.info("Please set the `GROQ_API_KEY` environment variable on your system or in a `.env` file.")
    st.stop() # Stop the app if API key is missing

# --- PDF Upload ---
st.subheader("Upload PDF Document(s)")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Select the PDF documents you want to use as knowledge base."
)

# --- Process Documents and Setup RAG ---
# st.cache_resource show_spinner handles the progress message
@st.cache_resource(show_spinner="Setting up RAG system with your documents...")
def setup_rag_system(files, groq_key, system_prompt_tpl):
    if not files:
        return None, "Please upload PDF files to proceed."

    # --- Embedding Model ---
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    all_docs = []
    temp_dir = "temp_pdf_uploads"
    os.makedirs(temp_dir, exist_ok=True) # Ensure temp directory exists

    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyMuPDFLoader(temp_filepath)
            docs = loader.load()
            if not docs:
                # Still show a warning if a specific PDF is empty
                st.warning(f"Warning: PDF document '**{uploaded_file.name}**' loaded but contains no pages/content. Skipping.")
                continue
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error loading PDF '**{uploaded_file.name}**': {e}")
            return None, f"Error loading PDF '**{uploaded_file.name}**': {e}"
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    # Clean up the temporary directory if it's empty
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    if not all_docs:
        return None, "No valid documents were loaded from the uploaded PDFs."

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
            st.markdown(message["content"]) # THIS LINE IS CORRECTED

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
    st.info("Please upload one or more PDF documents to start chatting.")

st.sidebar.markdown("---")
st.sidebar.info("The FAISS index is in-memory and will be cleared when you close or refresh the tab.")
