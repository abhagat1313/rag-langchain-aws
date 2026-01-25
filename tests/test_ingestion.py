from ingestion.loaders import load_documents


def pretty_print_docs(docs, max_chars=200):
    for i, doc in enumerate(docs):
        print("=" * 50)
        print(f"Document {i + 1}")
        print(f"Content preview: {doc.page_content[:max_chars]}...")
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    docs = load_documents("data/raw")
    print(f"\nTotal documents loaded: {len(docs)}\n")
    pretty_print_docs(docs)