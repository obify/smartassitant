import streamlit as st
import os
from dotenv import load_dotenv

# --- Configuration (Moved to Streamlit secrets or direct input where applicable) ---

# Load environment variables from .env file (for local development)
load_dotenv()

# Hugging Face Embedding Model (CPU friendly)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text Splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Import necessary Langchain and other modules ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterCharacterTextSplitter
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
    Upload one or more PDF files, provide a system prompt, and start asking questions!
    """
)

# --- Groq API Key Input ---
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key", type="password", help="You can get your API key from app.groq.com"
)

if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API Key to proceed.")

# --- System Prompt Input ---
st.sidebar.subheader("System Prompt")
default_system_prompt = """
You are a helpful AI assistant.
Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
system_prompt_template = st.sidebar.text_area(
    "Customize the system prompt:",
    value=default_system_prompt,
    height=200,
    help="This prompt guides the AI's behavior. The `{context}` and `{question}` placeholders are required."
)

# Validate system prompt
if "{context}" not in system_prompt_template or "{question}" not in system_prompt_template:
    st.sidebar.error("Error: The system prompt must contain `{context}` and `{question}` placeholders.")
    system_prompt_template = None # Invalidate the prompt if it's incorrect

# --- PDF Upload ---
st.subheader("Upload PDF Document(s)")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Select the PDF documents you want to use as knowledge base."
)

# --- Process Documents and Setup RAG ---
@st.cache_resource(show_spinner="Loading embedding model and setting up RAG system...")
def setup_rag_system(files, groq_key, system_prompt_tpl):
    if not files or not groq_key or not system_prompt_tpl:
        return None, "Please upload PDF files, enter your Groq API key, and ensure the system prompt is valid."

    # --- Embedding Model ---
    st.info(f"Loading Hugging Face embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        return None, f"Error loading embedding model: {e}"
    st.success("Embedding model loaded.")

    all_docs = []
    for uploaded_file in files:
        # Save uploaded file temporarily to process
        with open(os.path.join("./", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"Loading document from {uploaded_file.name}...")
        try:
            loader = PyMuPDFLoader(uploaded_file.name)
            docs = loader.load()
            if not docs:
                st.warning(f"Warning: PDF document '{uploaded_file.name}' loaded but contains no pages/content. Skipping.")
                continue
            all_docs.extend(docs)
            st.success(f"Successfully loaded {len(docs)} pages from {uploaded_file.name}.")
        except Exception as e:
            st.error(f"Error loading PDF '{uploaded_file.name}': {e}")
            return None, f"Error loading PDF '{uploaded_file.name}': {e}"
        finally:
            # Clean up temporary file
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)

    if not all_docs:
        return None, "No valid documents were loaded from the uploaded PDFs."

    st.info(f"Splitting {len(all_docs)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(all_docs)
    st.success(f"Created {len(all_splits)} document chunks.")

    st.info("Creating IN-MEMORY FAISS vector store...")
    try:
        vectorstore = FAISS.from_documents(
            documents=all_splits,
            embedding=embeddings
        )
    except Exception as e:
        return None, f"Error creating vector store: {e}"
    st.success("IN-MEMORY FAISS vector store ready.")

    retriever = vectorstore.as_retriever()

    st.info(f"Initializing Groq LLM...")
    try:
        llm = ChatGroq(
            api_key=groq_key,
            model_name="llama3-8b-8192" # Hardcoding for simplicity, could be a selectbox
        )
    except Exception as e:
        return None, f"Error initializing Groq LLM. Check your API key: {e}"
    st.success("Groq LLM initialized.")

    st.info("Setting up RAG chain...")
    prompt = ChatPromptTemplate.from_template(system_prompt_tpl)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.success("RAG chain setup complete.")

    return rag_chain, None

rag_chain, setup_error = None, None
if uploaded_files and groq_api_key and system_prompt_template:
    rag_chain, setup_error = setup_rag_system(uploaded_files, groq_api_key, system_prompt_template)

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
                # Invoke the RAG chain
                response = rag_chain.invoke(prompt)
                full_response = response
            except Exception as e:
                full_response = f"An error occurred: {e}"
                if "groq.exceptions" in str(e):
                    full_response += "\n\nPotential Groq API error. Check your API key, model name, or account usage."
            
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
elif uploaded_files and (not groq_api_key or not system_prompt_template):
    st.info("Please complete the configuration in the sidebar (API key and valid system prompt) to enable the chat.")
elif not uploaded_files and groq_api_key and system_prompt_template:
    st.info("Please upload one or more PDF documents to start chatting.")
else:
    st.info("Upload PDF documents and provide your Groq API key and system prompt to begin.")

st.sidebar.markdown("---")
st.sidebar.info("The FAISS index is in-memory and will be cleared when you close or refresh the tab.")
