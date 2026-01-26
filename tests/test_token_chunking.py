import unittest
from langchain_core.documents import Document
from chunking.token_chunker import token_chunk_document, TOKENIZER


class TestTokenChunking(unittest.TestCase):

    def setUp(self):
        self.text = (
            "This is a sample document to test token-aware chunking. "
            "It contains multiple sentences to ensure overlap and chunking works properly."
        )
        self.doc = Document(
            page_content=self.text,
            metadata={"source": "test_doc.txt"}
        )

    def test_chunking_basic(self):
        chunk_size = 10  # small for testing
        overlap = 3

        chunks = token_chunk_document(self.doc, chunk_size=chunk_size, overlap=overlap)

        # 1️⃣ Check that we got chunks
        self.assertGreater(len(chunks), 0, "No chunks were created")

        # 2️⃣ Each chunk is smaller than chunk_size + some safety
        for c in chunks:
            tokens = TOKENIZER.encode(c.page_content, add_special_tokens=False)
            self.assertLessEqual(len(tokens), chunk_size, "Chunk exceeded token size")

        # 3️⃣ Check overlap is preserved
        if len(chunks) > 1:
            first_end_tokens = TOKENIZER.encode(chunks[0].page_content, add_special_tokens=False)[-overlap:]
            second_start_tokens = TOKENIZER.encode(chunks[1].page_content, add_special_tokens=False)[:overlap]
            self.assertEqual(first_end_tokens, second_start_tokens, "Overlap tokens not preserved")

        # 4️⃣ Metadata contains chunk info
        for i, c in enumerate(chunks):
            self.assertIn("chunk_index", c.metadata)
            self.assertEqual(c.metadata["chunk_index"], i)
            self.assertIn("token_count", c.metadata)


if __name__ == "__main__":
    unittest.main()