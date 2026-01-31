class PipelineOrchestrator:
    """
    Orchestrates embedding, retrieval, reranking, and optional heavy reranking.
    """

    def __init__(self, embedder, retriever, reranker, heavy_embedder=None, cache=None):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.heavy_embedder = heavy_embedder
        self.cache = cache

    def run(
        self,
        query: str,
        metadata_weights=None,
        use_heavy_rerank: bool = False,
        use_cache: bool = True
    ):
        # ----------------------------
        # 1️⃣ Build cache key
        # ----------------------------
        cache_key = (
            f"retrieval:"
            f"{query}:"
            f"topk={self.retriever.top_k}:"
            f"heavy={use_heavy_rerank}"
        )

        # ----------------------------
        # 2️⃣ Try cache
        # ----------------------------
        if self.cache and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return self.retriever.vector_store.get_documents_by_ids(
                    cached["doc_ids"]
                )

        # ----------------------------
        # 3️⃣ Embed query
        # ----------------------------
        query_vector = self.embedder.embed_query(query)

        # ----------------------------
        # 4️⃣ Retrieve top-K
        # ----------------------------
        top_docs = self.retriever.retrieve(query_vector)

        # ----------------------------
        # 5️⃣ Gather candidate embeddings
        # ----------------------------
        candidate_embeddings = [
            self.retriever.vector_store.get_embedding(doc)
            for doc in top_docs
        ]

        # ----------------------------
        # 6️⃣ Lightweight rerank
        # ----------------------------
        reranked_docs = self.reranker.weighted_rerank(
            query_vector=query_vector,
            candidate_docs=top_docs,
            candidate_embeddings=candidate_embeddings,
            metadata_weights=metadata_weights,
        )

        # ----------------------------
        # 7️⃣ Optional heavy rerank
        # ----------------------------
        if use_heavy_rerank and self.heavy_embedder:
            heavy_query_vector = self.heavy_embedder.embed_query(query)
            reranked_docs = self.reranker.heavy_rerank(
                query_vector=query_vector,
                candidate_docs=reranked_docs,
                candidate_embeddings=candidate_embeddings,
                heavy_query_vector=heavy_query_vector,
            )

        # ----------------------------
        # 8️⃣ Store in cache
        # ----------------------------
        if self.cache and use_cache:
            self.cache.set(
                cache_key,
                {
                    "doc_ids": [doc.metadata["id"] for doc in reranked_docs]
                },
                ttl_seconds=300,  # 5 minutes
            )

        return reranked_docs
