from typing import List, Optional
from langchain_core.documents import Document
import numpy as np

class Reranker:
    """
    Weighted or heavy reranker. Only works with vectors.
    """
    @staticmethod
    def weighted_rerank(
        query_vector: List[float],
        candidate_docs: List[Document],
        candidate_embeddings: List[List[float]],
        metadata_weights: Optional[List[float]] = None, # Optional weights from metadata, recency or importance
        alpha: float = 0.7, # Weight for vector similarity vs metadata
        beta: float = 0.3
    ) -> List[Document]:

        if metadata_weights is None:
            metadata_weights = [1.0] * len(candidate_docs)

        final_scores = []
        q = np.array(query_vector, dtype=np.float32)

        for emb, meta_w in zip(candidate_embeddings, metadata_weights):
            v = np.array(emb, dtype=np.float32)
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))
            final_scores.append(alpha * sim + beta * meta_w)

        sorted_docs = [doc for _, doc in sorted(zip(final_scores, candidate_docs), reverse=True)]
        return sorted_docs

    @staticmethod
    def heavy_rerank(
        query_vector: List[float],
        candidate_docs: List[Document],
        candidate_embeddings: List[List[float]],
        heavy_query_vector: List[float]
    ) -> List[Document]:
        """
        Heavy rerank using a more precise embedding.
        """
        return Reranker.weighted_rerank(
            query_vector=heavy_query_vector,
            candidate_docs=candidate_docs,
            candidate_embeddings=candidate_embeddings
        )
