from embeddings.HuggingFaceEmbedder import HuggingFaceEmbedder
from vectorstore.FaissVectorStore import FaissVectorStore
from retrieval.retriever import Retriever
from reranker.reranker import Reranker
from orchestrator.pipeline import PipelineOrchestrator
from langchain_core.documents import Document

# 1. Prepare documents
docs = [
    Document(page_content="Weekly HR update: New hires and policy changes."),
    Document(page_content="Engineering update: Deployed new microservice."),
    Document(page_content="Finance report: Quarterly earnings and projections."),
]

# 2. Embed documents
embedder = HuggingFaceEmbedder()
doc_embeddings = embedder.embed_documents(docs)

# 3. Create FAISS store
vector_store = FaissVectorStore(embedding_dim=embedder.embedding_dim)
vector_store.add_documents(embeddings=doc_embeddings, documents=docs)

# 4. Retriever, Reranker, heavy embedder
retriever = Retriever(vector_store=vector_store, top_k=3)
reranker = Reranker()
heavy_embedder = HuggingFaceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Orchestrator
pipeline = PipelineOrchestrator(embedder, retriever, reranker, heavy_embedder=heavy_embedder)

# 6. Query
query = "Summarize last week's HR updates"
results = pipeline.run(query, use_heavy_rerank=True)

# 7. Print results
for i, doc in enumerate(results, 1):
    print(f"{i}: {doc.page_content}")


