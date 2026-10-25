"""
Document Loader Module for RAG Pipeline.

Provides unified loaders for various document sources:
- Text files (.txt, .md)
- PDF files (.pdf)
- Web URLs (HTTP/HTTPS)
- Local Directories
- Raw in-memory strings
"""

import os
from typing import List, Optional
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)


class DocumentLoader:
    """Unified Document Loader supporting multiple input formats."""

    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> List[Document]:
        """Load document from a text file (.txt, .md)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        loader = TextLoader(file_path, encoding=encoding)
        return loader.load()

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load document from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_web(url: str) -> List[Document]:
        """Load document content from a webpage URL."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        loader = WebBaseLoader(url)
        return loader.load()

    @staticmethod
    def load_directory(
        dir_path: str,
        glob_pattern: str = "*.txt",
        recursive: bool = True,
    ) -> List[Document]:
        """Load all matching documents in a directory."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        loader = DirectoryLoader(
            path=dir_path,
            glob=glob_pattern,
            recursive=recursive,
        )
        return loader.load()

    @staticmethod
    def load_raw_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Wrap raw in-memory string as a LangChain Document."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        meta = metadata or {"source": "raw_input"}
        return [Document(page_content=text, metadata=meta)]

    @classmethod
    def load(cls, source: str) -> List[Document]:
        """
        Auto-detect source type and load documents.

        Args:
            source: A file path, URL, or raw text.

        Returns:
            List of LangChain Document instances.
        """
        # Check if source is a web URL
        if source.startswith("http://") or source.startswith("https://"):
            return cls.load_web(source)

        # Check if source is an existing directory
        if os.path.isdir(source):
            return cls.load_directory(source)

        # Check if source is an existing file
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == ".pdf":
                return cls.load_pdf(source)
            else:
                return cls.load_text(source)

        # Otherwise treat as raw text
        return cls.load_raw_text(source)
