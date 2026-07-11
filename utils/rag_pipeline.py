from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma

from utils.embeddings import get_embedding_model

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = get_embedding_model()

# -----------------------------
# Load Chroma Database
# -----------------------------
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.20
    }
)


# -----------------------------
# Load LLM
# -----------------------------
generator = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens=300,
    do_sample=False
)

llm = HuggingFacePipeline(
    pipeline=generator
)

# -----------------------------
# Ask Question
# -----------------------------
def ask_question(question):

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Retrieve documents
    docs = retriever.invoke(question)

    # Remove duplicates
    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    docs = unique_docs

    # Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Empty context check
    if len(context.strip()) < 30:
        return "I could not find the answer in the uploaded documents.", docs

    # Prompt
    prompt = f"""
You are a document question-answering assistant.

Use ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt).strip()

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer, docs