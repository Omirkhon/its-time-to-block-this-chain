import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Free embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama  # Local LLM
from langchain_core.prompts import PromptTemplate

# Initialize session state
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Page configuration
st.set_page_config(page_title="Free Document Chat", page_icon=":books:")
st.title("Document Chat (Free Local Version)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown("""
    **Using:**
    - Ollama (local LLM)
    - Sentence Transformers (free embeddings)
    - No API keys needed!
    """)

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"]
    )

    process_button = st.button("Process Documents")

# Document processing
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        # Save uploaded files
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_files.append(file_path)

        # Load documents
        documents = []
        for file_path in st.session_state.uploaded_files:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                continue

            documents.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings (free alternative)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Lightweight free model
            model_kwargs={'device': 'cpu'}  # Use GPU if available: change to 'cuda'
        )

        # Create vector store
        st.session_state.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        st.session_state.documents_processed = True
        st.success("Documents processed successfully!")

# Chat UI
st.header("Chat with Your Documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.documents_processed:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents first.")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Please upload and process documents first."
        })
    else:
        # Initialize local LLM (Ollama)
        llm = Ollama(model="llama2")  # or "mistral", "gemma", etc.

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        # Custom prompt
        custom_template = """Use the following context to answer the question.
        If you don't know, say you don't know. Don't make up answers.

        Context: {context}
        Question: {question}

        Helpful Answer:"""

        CUSTOM_QUESTION_PROMPT = PromptTemplate(
            template=custom_template,
            input_variables=["context", "question"]
        )

        retriever = st.session_state.vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
            return_source_documents=True
        )

        with st.spinner("Thinking..."):
            response = qa_chain({"question": prompt})

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"]
        })