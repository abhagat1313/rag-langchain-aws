from copy import deepcopy
from typing import List

from langchain_core.documents import Document


def character_chunk_document(
    document: Document,
    chunk_size: int = 1000,
    overlap: int = 0,
) -> List[Document]:
    """
    Split a single Document into character-based chunks.

    Args:
        document: LangChain Document containing text + metadata
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of new Document objects (chunks)
    """

    text = document.page_content

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Document] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Copy metadata so we never mutate the original document
        metadata = deepcopy(document.metadata)
        metadata["chunk_index"] = chunk_index
        metadata["chunk_type"] = "character"

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

        chunk_index += 1
        start = end - overlap

    return chunks