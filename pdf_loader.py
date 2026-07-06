import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from utils.embeddings import get_embedding_model


# -------------------------------
# Load Document
# -------------------------------
def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return loader.load()


# -------------------------------
# Split Documents
# -------------------------------
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


# -------------------------------
# Store in ChromaDB
# -------------------------------
def add_file_to_database(file_path):

    documents = load_document(file_path)

    chunks = split_documents(documents)

    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    vector_store.add_documents(chunks)

    return len(chunks)