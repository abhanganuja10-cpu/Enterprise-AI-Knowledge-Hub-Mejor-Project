from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the embedding model used to convert text into vector embeddings.
    """

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embedding