from embeddings.HuggingFaceEmbedder import HuggingFaceEmbedder
from langchain_core.documents import Document


def test_huggingface_embeddings():
    embedder = HuggingFaceEmbedder()

    docs = [
        Document(page_content="Dogs are great pets."),
        Document(page_content="Cats are independent animals.")
    ]

    vectors = embedder.embed_documents(docs)

    # Assertions
    assert len(vectors) == 2
    assert len(vectors[0]) == embedder.embedding_dim