from langchain_core.documents import Document

from chunking.character import character_chunk_document


def test_character_chunking_basic():
    doc = Document(
        page_content="abcdefghijklmnopqrstuvwxyz",
        metadata={"source": "test.txt"},
    )

    chunks = character_chunk_document(
        doc,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 3

    assert chunks[0].page_content == "abcdefghij"
    assert chunks[1].page_content == "ijklmnopqr"
    assert chunks[2].page_content == "qrstuvwxyz"

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["chunk_type"] == "character"
        assert chunk.metadata["source"] == "test.txt"