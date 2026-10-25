"""
Vector Store & Embeddings Module for RAG Pipeline.

Handles:
- Embedding model instantiation (OpenAI / Google Gemini)
- Vector database management using ChromaDB
- Document indexing, persistence, and similarity search
- Configurable retriever generation
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.config import EmbeddingConfig, VectorDBConfig, config as default_config


def get_embeddings(embedding_config: Optional[EmbeddingConfig] = None) -> Embeddings:
    """
    Factory function to initialize the appropriate Embedding model.

    Args:
        embedding_config: Configuration for embedding provider and model.

    Returns:
        Embeddings model instance.
    """
    cfg = embedding_config or default_config.embedding
    provider = cfg.provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = default_config.api.openai_api_key
            return OpenAIEmbeddings(
                model=cfg.model_name,
                openai_api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. "
                "Install via: pip install langchain-openai"
            )

    elif provider in ("google_genai", "gemini", "google"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = default_config.api.google_api_key
            return GoogleGenerativeAIEmbeddings(
                model=cfg.model_name,
                google_api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google GenAI embeddings. "
                "Install via: pip install langchain-google-genai"
            )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


class VectorStoreManager:
    """ChromaDB Vector Store Manager for document indexing and retrieval."""

    def __init__(
        self,
        vector_config: Optional[VectorDBConfig] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.config = vector_config or default_config.vector_db
        self.embeddings = embeddings or get_embeddings()
        self._vector_store = None

    @property
    def vector_store(self):
        """Lazy load or initialize Chroma vector store."""
        if self._vector_store is None:
            self._vector_store = self._init_chroma()
        return self._vector_store

    def _init_chroma(self):
        """Initialize ChromaDB client with persistence."""
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                raise ImportError(
                    "chromadb and langchain-chroma (or langchain-community) are required. "
                    "Install via: pip install chromadb langchain-chroma"
                )

        os.makedirs(self.config.persist_directory, exist_ok=True)
        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add document chunks to vector database and persist."""
        if not documents:
            return []
        ids = self.vector_store.add_documents(documents)
        # In modern Langchain/Chroma, persistence is automatic
        return ids

    def as_retriever(
        self,
        search_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Convert vector store to LangChain Retriever.

        Args:
            search_type: 'similarity' or 'mmr'
            top_k: Number of relevant chunks to retrieve

        Returns:
            VectorStoreRetriever instance
        """
        stype = search_type or self.config.search_type
        k = top_k or self.config.top_k
        return self.vector_store.as_retriever(
            search_type=stype,
            search_kwargs={"k": k},
        )

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search for a query string."""
        count = k or self.config.top_k
        return self.vector_store.similarity_search(query, k=count)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with distance scores."""
        count = k or self.config.top_k
        return self.vector_store.similarity_search_with_score(query, k=count)

    def delete_collection(self) -> None:
        """Delete the current ChromaDB collection."""
        if self._vector_store:
            self._vector_store.delete_collection()
            self._vector_store = None
