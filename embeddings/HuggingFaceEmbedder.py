from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    """
    CPU-based Hugging Face embedding wrapper.
    Responsible only for converting text chunks into vectors.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load model on CPU by default
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Convert a list of LangChain Documents into embedding vectors.
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        """
        return self.model.encode(query).tolist()