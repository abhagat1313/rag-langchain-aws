from typing import List
from langchain_core.documents import Document

class VectorStoreInterface:
    """
    Generic interface for any vector store.
    """
    def similarity_search(self, query_vector: List[float], top_k: int) -> List[Document]:
        raise NotImplementedError

    def get_embedding(self, doc: Document) -> List[float]:
        raise NotImplementedError
