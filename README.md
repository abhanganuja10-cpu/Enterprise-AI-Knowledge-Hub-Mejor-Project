# 🤖 Enterprise AI Knowledge Hub

## 📖 Project Description

Enterprise AI Knowledge Hub is an intelligent document-based Question Answering System developed using Retrieval-Augmented Generation (RAG). The application enables users to upload multiple document formats, including PDF, DOCX, TXT, and PPTX, and ask questions in natural language. Instead of searching documents manually, the system retrieves the most relevant information using semantic search and generates accurate answers based on the uploaded content.

The application uses Hugging Face sentence-transformer embeddings to convert document text into vector representations and stores them in ChromaDB, allowing fast and efficient similarity search. When a user submits a question, the system retrieves the most relevant document chunks and provides a context-aware answer using a Large Language Model (LLM). The application also displays the source document and relevant content used to generate the answer, improving transparency and reliability.

The project is developed using Python, Streamlit, LangChain, Hugging Face Transformers, and ChromaDB. It provides a simple and interactive web interface where users can upload documents, ask questions, view AI-generated answers, inspect the retrieved document sections, and access chat history.

This project demonstrates the practical implementation of Retrieval-Augmented Generation (RAG) for enterprise knowledge management, educational resources, digital libraries, research document analysis, and intelligent document retrieval systems.

# Features

- Upload multiple document formats (PDF, DOCX, TXT, PPTX)
- Automatic document preprocessing and text chunking
- Semantic search using vector embeddings
- ChromaDB vector database for efficient retrieval
- AI-generated answers using Retrieval-Augmented Generation (RAG)
- Displays source document and retrieved content
- Interactive Streamlit web interface
- Chat history support
- Fast and accurate document search

# Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- ChromaDB
- PyPDF

# How to Run

```bash
pip install -r requirements.txt

streamlit run app.py

```

# Project Structure


Enterprise-AI-Knowledge-Hub/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── embeddings.py
│   ├── pdf_loader.py
│   ├── rag_pipeline.py
│   └── __init__.py
│
└── screenshots/


# Future Scope

- Support additional document formats such as Excel, HTML, and Markdown.
- Integration with cloud storage services.
- User authentication and role-based access.
- Conversation memory for multi-turn question answering.
- OCR support for scanned documents.
- Voice-based document querying.
- Deployment on cloud platforms.
