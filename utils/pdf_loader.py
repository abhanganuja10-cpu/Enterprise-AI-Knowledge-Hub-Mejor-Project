import os
import pandas as pd
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from utils.embeddings import get_embedding_model


def load_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    elif ext == ".txt":
        loader = TextLoader(file_path)
        return loader.load()

    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()

    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    elif ext == ".xlsx":

        df = pd.read_excel(file_path)

        text = df.fillna("").to_csv(index=False)

        return [
            Document(
                page_content=text,
                
                metadata={
              "source": os.path.basename(file_path),
               "page":1,
               "file_type": ext
            }
                
                
            )
        ]

    elif ext == ".md":

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": 1
                }
            )
        ]

    elif ext == ".html":

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        text = " ".join(soup.stripped_strings)

        return [
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": 1
                }
            )
        ]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
def add_file_to_database(file_path):

    pages = load_document(file_path)
    print("Pages Loaded:", len(pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(pages)
    print("Chunks Created:", len(docs))

    embedding = get_embedding_model()

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    vector_store.add_documents(docs)

    print("Documents Added Successfully")

    return len(docs)