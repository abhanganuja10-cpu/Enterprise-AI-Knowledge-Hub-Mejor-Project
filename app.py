import streamlit as st
import tempfile
import os

from utils.pdf_loader import add_file_to_database
from utils.rag_pipeline import ask_question

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Knowledge Hub")
st.write("Upload your documents and ask questions using AI.")

# -----------------------------
# Session State
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "pptx"]
)

if uploaded_file is not None:

    extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("Processing document..."):

        try:
            chunks = add_file_to_database(temp_path)
            st.success(f"Document processed successfully ({chunks} chunks added).")

        except Exception as e:
            st.error(f"Error: {e}")

    os.remove(temp_path)

st.divider()

# -----------------------------
# Question Section
# -----------------------------
question = st.text_input("Ask a question")

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching ..."):

            answer, docs = ask_question(question)

        st.session_state.history.append(
            ("You", question)
        )

        st.session_state.history.append(
            ("AI", answer)
        )

        st.subheader(" Answer")
        st.write(answer)

        st.divider()

        st.subheader("📄 Relevant Documents")

        if len(docs) == 0:
            st.info("No relevant documents found.")

        else:

            for i, doc in enumerate(docs, start=1):

                source = doc.metadata.get("source", "Unknown File")
                page = doc.metadata.get("page", None)

                with st.expander(f"Document {i}"):

                    st.write(f"**Source:** {source}")

                    if page is not None:
                        st.write(f"**Page:** {page + 1}")

                    st.write(doc.page_content)

st.divider()

# -----------------------------
# Chat History
# -----------------------------
st.subheader("💬 Chat History")

for role, message in reversed(st.session_state.history):

    if role == "You":
        st.markdown(f"**🧑 You:** {message}")

    else:
        st.markdown(f"**🤖 AI:** {message}")