"""
Configuration Module for RAG Pipeline.

Manages application configurations including:
- API Keys (OpenAI, Google Gemini)
- LLM parameters (model, temperature, max_tokens)
- Embedding parameters
- Text Splitting configurations
- Vector Database parameters (ChromaDB)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """API Keys and Credentials."""
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


@dataclass
class LLMConfig:
    """Large Language Model settings."""
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower()
    )  # 'openai' or 'google_genai'
    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    max_tokens: Optional[int] = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024")) if os.getenv("LLM_MAX_TOKENS") else None
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "60"))
    )


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""
    provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    )  # 'openai' or 'google_genai'
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME",
            "text-embedding-3-small"
        )
    )
    # Default for Google GenAI embeddings is often "models/gemini-embedding-001" or "models/text-embedding-004"


@dataclass
class TextSplitterConfig:
    """Document chunking configurations."""
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    strategy: str = field(
        default_factory=lambda: os.getenv("CHUNKING_STRATEGY", "recursive")
    )  # 'recursive', 'character', 'semantic'
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""]
    )
    # For SemanticChunker
    breakpoint_threshold_type: str = "standard_deviation"
    breakpoint_threshold_amount: float = 1.0


@dataclass
class VectorDBConfig:
    """Vector Database configurations (ChromaDB)."""
    provider: str = "chroma"
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
    )
    persist_directory: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    distance_metric: str = "cosine"  # 'cosine', 'l2', or 'ip'
    top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "4"))
    )
    search_type: str = "similarity"  # 'similarity' or 'mmr'


@dataclass
class RAGConfig:
    """Master Application Configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    text_splitter: TextSplitterConfig = field(default_factory=TextSplitterConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)

    def validate(self) -> None:
        """Validate critical configuration fields based on chosen providers."""
        if self.llm.provider == "openai" or self.embedding.provider == "openai":
            if not self.api.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI for LLM or Embeddings. "
                    "Please set it in your environment or .env file."
                )

        if self.llm.provider == "google_genai" or self.embedding.provider == "google_genai":
            if not self.api.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY (or GEMINI_API_KEY) is required when using Google GenAI. "
                    "Please set it in your environment or .env file."
                )


# Export a default singleton config instance
config = RAGConfig()
