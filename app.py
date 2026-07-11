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
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "pptx", "xlsx", "md", "html", "htm"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp:

            tmp.write(uploaded_file.read())
            file_path = tmp.name

        with st.spinner(f"Processing {uploaded_file.name}..."):

            chunks = add_file_to_database(file_path)

        st.success(
            f"✅ {uploaded_file.name} uploaded successfully! ({chunks} chunks added)"
        )

        os.remove(file_path)

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