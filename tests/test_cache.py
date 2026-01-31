from langchain_core.documents import Document

from orchestrator.pipeline import PipelineOrchestrator
from cache.redis_cache import RedisCache

class FakeEmbedder:
    def embed_query(self, query: str):
        # Always return same vector for same query
        return [1.0, 0.0, 0.0]
class FakeVectorStore:
    def __init__(self):
        self.docs = {
            "1": Document(page_content="Doc 1", metadata={"id": "1"}),
            "2": Document(page_content="Doc 2", metadata={"id": "2"}),
        }

    def get_embedding(self, doc):
        return [1.0, 0.0, 0.0]

    def get_documents_by_ids(self, ids):
        return [self.docs[i] for i in ids]
class FakeRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.top_k = 2
        self.call_count = 0

    def retrieve(self, query_vector):
        self.call_count += 1
        return list(self.vector_store.docs.values())
class FakeReranker:
    def weighted_rerank(
        self,
        query_vector,
        candidate_docs,
        candidate_embeddings=None,
        metadata_weights=None,
    ):
        return candidate_docs

def test_pipeline_uses_cache():
    # -------------------------
    # Setup
    # -------------------------
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = FakeRetriever(vector_store)
    reranker = FakeReranker()

    cache = RedisCache()
    cache.client.flushdb()  # Clear Redis before test

    pipeline = PipelineOrchestrator(
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        cache=cache,
    )

    query = "test query"

    # -------------------------
    # First run (cache MISS)
    # -------------------------
    result_1 = pipeline.run(query)

    assert retriever.call_count == 1
    assert len(result_1) == 2

    # -------------------------
    # Second run (cache HIT)
    # -------------------------
    result_2 = pipeline.run(query)

    assert retriever.call_count == 1  # unchanged!
    assert len(result_2) == 2
    print("First run result:", [doc.metadata["id"] for doc in result_1])
    print("Second run result:", [doc.metadata["id"] for doc in result_2])
    print("Retriever call count:", retriever.call_count)
