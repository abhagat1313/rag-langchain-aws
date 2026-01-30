class PipelineOrchestrator:
    """
    Orchestrates embedding, retrieval, reranking, and optional LLM.
    """
    def __init__(self, embedder, retriever, reranker, heavy_embedder=None):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.heavy_embedder = heavy_embedder

    def run(self, query: str, metadata_weights=None, use_heavy_rerank=False):
        # 1️⃣ Embed query
        query_vector = self.embedder.embed_query(query)

        # 2️⃣ Retrieve top-K
        top_docs = self.retriever.retrieve(query_vector)

        # 3️⃣ Gather candidate embeddings
        candidate_embeddings = [self.retriever.vector_store.get_embedding(doc) for doc in top_docs]

        # 4️⃣ Lightweight rerank
        reranked_docs = self.reranker.weighted_rerank(
            query_vector=query_vector,
            candidate_docs=top_docs,
            candidate_embeddings=candidate_embeddings,
            metadata_weights=metadata_weights
        )

        # 5️⃣ Optional heavy rerank
        if use_heavy_rerank and self.heavy_embedder:
            heavy_query_vector = self.heavy_embedder.embed_query(query)
            reranked_docs = self.reranker.heavy_rerank(
                query_vector=query_vector,
                candidate_docs=reranked_docs,
                candidate_embeddings=candidate_embeddings,
                heavy_query_vector=heavy_query_vector
            )

        return reranked_docs
