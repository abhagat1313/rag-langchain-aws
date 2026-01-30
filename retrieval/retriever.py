from typing import List
from langchain_core.documents import Document
from vectorstore.vector_store_interface import VectorStoreInterface

class Retriever:
    """
    Embedding-agnostic retriever.
    """
    def __init__(self, vector_store: VectorStoreInterface, top_k: int = 10):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query_vector: List[float]) -> List[Document]:
        """
        Return top-K nearest documents for the query vector.
        """
        return self.vector_store.similarity_search(query_vector, top_k=self.top_k)

