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
    search_kwargs={"k": 3}
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
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY from the given context.

If the answer is not found, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer, docs