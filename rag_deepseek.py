import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    /* Modern Cyberpunk Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    /* Title and Headers */
    h1 {
        background: linear-gradient(120deg, #00ffaa 0%, #00a3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #00ffaa !important;
        font-weight: 600 !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(30, 30, 40, 0.6) !important;
        border: 1px solid rgba(0, 255, 170, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* User Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 3px solid #00ffaa !important;
    }
    
    /* Assistant Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #00a3ff !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(30, 30, 40, 0.6) !important;
        border: 1px solid rgba(0, 255, 170, 0.2) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
    }
    
    /* Input Field */
    .stChatInput input {
        background: rgba(30, 30, 40, 0.8) !important;
        border: 1px solid rgba(0, 255, 170, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 0.8rem !important;
    }
    
    .stChatInput input:focus {
        border-color: #00ffaa !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 170, 0.2) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(20, 20, 30, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: #00ffaa !important;
    }
    
    /* Success Message */
    .stSuccess {
        background: rgba(0, 255, 170, 0.1) !important;
        border: 1px solid rgba(0, 255, 170, 0.2) !important;
        color: #00ffaa !important;
    }
    
    /* Custom Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00ffaa, transparent);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NexusThink")
st.markdown("### Your Private Document Intelligence Companion")
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
