"""
Text Splitter & Chunking Module for RAG Pipeline.

Provides flexible chunking strategies:
1. Recursive Character Text Splitter (Recommended default)
2. Character Text Splitter (Delimiter/paragraph based)
3. Semantic Chunker (Embeddings-driven breakpoint detection)
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    Language,
)

from config.config import TextSplitterConfig


class TextChunker:
    """Manages document chunking strategies."""

    def __init__(self, config: Optional[TextSplitterConfig] = None, embeddings=None):
        self.config = config or TextSplitterConfig()
        self.embeddings = embeddings

    def get_splitter(self):
        """Build and return text splitter based on configuration."""
        strategy = self.config.strategy.lower()

        if strategy == "character":
            return CharacterTextSplitter(
                separator=self.config.separators[0] if self.config.separators else "\n\n",
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

        elif strategy == "semantic":
            if not self.embeddings:
                raise ValueError("Embeddings model is required for Semantic Chunker.")
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                return SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                    breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
                )
            except ImportError:
                raise ImportError(
                    "langchain_experimental is required for SemanticChunker. "
                    "Install via: pip install langchain-experimental"
                )

        else:
            # Default: RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators,
                length_function=len,
            )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Document objects into smaller chunks."""
        if not documents:
            return []
        splitter = self.get_splitter()
        chunks = splitter.split_documents(documents)
        return chunks

    def split_text(self, text: str) -> List[str]:
        """Split raw text into smaller text strings."""
        if not text:
            return []
        splitter = self.get_splitter()
        return splitter.split_text(text)

    @staticmethod
    def get_code_splitter(language: Language = Language.PYTHON, chunk_size: int = 500, chunk_overlap: int = 50):
        """Create language-aware splitter for source code."""
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
