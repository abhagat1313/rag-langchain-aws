from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
)


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and preserve page-level metadata.
    Each page becomes a Document.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = Path(file_path).name
        doc.metadata["doc_type"] = "pdf"

    return documents


def load_text(file_path: str) -> List[Document]:
    """
    Load long text documents (txt, md).
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = Path(file_path).name
        doc.metadata["doc_type"] = "text"

    return documents


def load_html(file_path: str) -> List[Document]:
    """
    Load HTML documents and strip boilerplate.
    """
    loader = BSHTMLLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = Path(file_path).name
        doc.metadata["doc_type"] = "html"

    return documents


def load_documents(path: str) -> List[Document]:
    """
    Load all supported documents from a directory or file.
    """
    documents: List[Document] = []
    path_obj = Path(path)

    if path_obj.is_file():
        return _load_single_file(path_obj)

    for file in path_obj.iterdir():
        if file.is_file():
            documents.extend(_load_single_file(file))

    return documents


def _load_single_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(str(file_path))
    elif suffix in [".txt", ".md"]:
        return load_text(str(file_path))
    elif suffix in [".html", ".htm"]:
        return load_html(str(file_path))
    else:
        print(f"Skipping unsupported file: {file_path.name}")
        return []