# rag-langchain-aws 🚀

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Experimental-orange)

This repository provides an **overview and implementation of LangChain workflows** in the project **`rag-langchain-aws`**. It includes utilities for document ingestion, chunking, embeddings, retrieval, reranking, caching, and orchestration — designed for building **LLM-powered applications** with AWS and generative AI.

---

## Table of Contents

- [Overview](#overview)  
- [Folder Structure](#folder-structure)  
- [Key Concepts](#key-concepts)  
- [Document Processing & Chunking](#document-processing--chunking)  
- [Embeddings](#embeddings)  
- [Caching & Storage](#caching--storage)  
- [Getting Started](#getting-started)  
- [FAQs & Notes](#faqs--notes)  
- [Future Work](#future-work)  
- [References](#references)  

---

## Overview

`rag-langchain-aws` enables structured workflows for LLM applications:

- Ingest data from PDFs, APIs, and text sources  
- Chunk content efficiently for LLM input  
- Generate embeddings for semantic search and retrieval  
- Cache results for fast access  
- Integrate vector storage for persistent embeddings  
- Orchestrate pipelines with agents and chains  

---

## Folder Structure

rag-langchain-aws/
├── chunking/ # Utilities for splitting documents into chunks
├── data/ # Raw and processed data
├── embedding/ # Embedding generation and management
├── ingestion/ # Scripts to ingest data from external sources
├── orchestrator/ # Main workflows and chain orchestrations
├── reranker/ # Modules to rank results from retrievals
├── retrieval/ # Retrieval logic using embeddings or search indices
├── vector_store/ # Persistent storage for embeddings / vectors
├── cache/ # Caching layer (e.g., Redis)
├── tests/ # Unit and integration tests
├── requirements.txt # Python dependencies


> The addition of `cache/` explicitly separates caching from vector storage for clarity.

---

## Key Concepts

- **LLMs**: Core engine for text generation and understanding  
- **Chains**: Modular workflows connecting LLMs with business logic  
- **Agents**: Decision-making components powered by LLMs  
- **Memory**: Maintains context across interactions  
- **Embeddings**: Vector representations of text for semantic search  
- **Cache**: Fast, temporary storage to reduce repeated computations  

---

## Document Processing & Chunking

- **Tokenizer-based chunking**: Split text into tokens within model limits  
- **Page-based chunking**: Preserves semantic boundaries in PDFs  
- **Overlap strategy**: Prevents loss of context between chunks  
- **Cleaning**: Remove repeated headers, footers, and irrelevant content  

> Proper cleaning is crucial; overlapping is a safeguard, not a replacement.  

---

## Embeddings

- Convert text chunks into numeric vectors for semantic search  
- Respect model input limits during embedding generation  
- Store embeddings efficiently in `vector_store/` for retrieval  

---

## Caching & Storage

- **Cache (`cache/`)**: Temporary storage for quick lookups (e.g., Redis)  
- **Vector Store (`vector_store/`)**: Persistent storage for embeddings, optimized for similarity search  
- **Difference**: Cache = fast temporary access, vector store = long-term embedding retrieval  

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-langchain-aws.git
cd rag-langchain-aws
```
### 2. Set up a Python environment

### On macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
After activation, your terminal prompt should show (venv) indicating the virtual environment is active.
```

### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Run tests
```
pytest tests/
```
### 5. Example workflow
```
1. Ingest documents via ingestion/
2. Chunk and clean them using chunking/
3. Generate embeddings in embedding/
4. Store vectors in vector_store/
5. Cache results in cache/ for fast retrieval
6. Query using retrieval/ and reranker/
7. Orchestrate workflows via orchestrator/
```
Each folder contains example scripts or README for detailed usage.

### FAQs & Notes
  PDF chunking: Page-aware chunking improves semantic accuracy
  Messy documents: Always clean headers, footers, and repeated content
  Cache vs Vector Store: Cache = temporary fast storage, vector store = persistent embeddings

###Future Work
  Explore advanced chunking strategies
  Optimize embedding storage, cache, and retrieval
  Implement agentic AI workflows integrated with LangChain
