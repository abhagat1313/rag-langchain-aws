import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document


class FaissVectorStore:
    """
    Simple FAISS-based vector store.
    """

    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Document] = []

    def add_documents(self, embeddings: List[List[float]], documents: List[Document]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        return [self.documents[i] for i in indices[0] if i != -1]