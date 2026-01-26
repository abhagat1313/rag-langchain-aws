from copy import deepcopy
from typing import List

from langchain_core.documents import Document
from transformers import AutoTokenizer

# 1️⃣ Load tokenizer once (proxy for token counting)
TOKENIZER = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def token_chunk_document(
    document: Document,
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[Document]:
    """
    Split a Document into token-based chunks using a HuggingFace tokenizer.

    Args:
        document: LangChain Document
        chunk_size: Max number of tokens per chunk (conservative, < model max)
        overlap: Number of tokens to overlap between consecutive chunks

    Returns:
        List of chunked Document objects with metadata
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.page_content

    # 2️⃣ Convert text → token IDs (proxy measurement)
    token_ids = TOKENIZER.encode(text, add_special_tokens=False)

    chunks: List[Document] = []
    start = 0
    chunk_index = 0

    # 3️⃣ Slide over tokens, create chunks with overlap
    while start < len(token_ids):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]

        # 4️⃣ Convert tokens back to text
        chunk_text = TOKENIZER.decode(chunk_token_ids)

        # 5️⃣ Copy and update metadata
        metadata = deepcopy(document.metadata)
        metadata.update({
            "chunk_index": chunk_index,
            "chunk_type": "token",
            "token_count": len(chunk_token_ids),
        })

        # 6️⃣ Add chunked document
        chunks.append(Document(
            page_content=chunk_text,
            metadata=metadata,
        ))

        chunk_index += 1
        start = end - overlap  # maintain overlap

    return chunks