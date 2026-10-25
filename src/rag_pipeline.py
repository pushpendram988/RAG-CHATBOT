"""
RAG Pipeline Module.

Orchestrates the complete Retrieval-Augmented Generation workflow:
1. Document Ingestion & Chunking
2. Embedding & Vector Indexing (ChromaDB)
3. Context Retrieval
4. Prompt Augmentation
5. LLM Answer Generation (OpenAI / Google Gemini) via LCEL
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.config import RAGConfig, config as global_config
from src.document_loader import DocumentLoader
from src.text_splitter import TextChunker
from src.vector_store import VectorStoreManager, get_embeddings
from src.prompts import get_prompt_template, PromptTemplate


@dataclass
class RAGResult:
    """Structured response from RAG query execution."""
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    execution_time_sec: float


class RAGPipeline:
    """End-to-end industry-grade RAG pipeline."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        prompt_type: str = "standard",
    ):
        self.config = config or global_config
        self.prompt_type = prompt_type

        # Initialize Embeddings & Vector Store
        self.embeddings = get_embeddings(self.config.embedding)
        self.vector_store = VectorStoreManager(
            vector_config=self.config.vector_db,
            embeddings=self.embeddings,
        )

        # Initialize Chunker
        self.chunker = TextChunker(
            config=self.config.text_splitter,
            embeddings=self.embeddings,
        )

        # Initialize LLM & Chain
        self.llm = self._init_llm()
        self.prompt = get_prompt_template(self.prompt_type)
        self._rag_chain = None

    def _init_llm(self):
        """Initialize the LLM based on configuration (OpenAI or Google GenAI)."""
        provider = self.config.llm.provider.lower()

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    api_key=self.config.api.openai_api_key,
                    timeout=self.config.llm.request_timeout,
                )
            except ImportError:
                raise ImportError(
                    "langchain-openai is required for OpenAI LLM. "
                    "Install via: pip install langchain-openai"
                )

        elif provider in ("google_genai", "gemini", "google"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_tokens,
                    google_api_key=self.config.api.google_api_key,
                )
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is required for Google GenAI LLM. "
                    "Install via: pip install langchain-google-genai"
                )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Format retrieved documents into a clean context string."""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            meta_str = f"[Doc {i+1} | Source: {source}" + (f", Page: {page}" if page else "") + "]"
            formatted.append(f"{meta_str}\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    def ingest(self, source: str) -> int:
        """
        Ingest documents from a file path, URL, directory, or raw text.

        Args:
            source: Path to file/directory, URL, or raw text.

        Returns:
            Number of chunks indexed into vector store.
        """
        # 1. Load documents
        documents = DocumentLoader.load(source)
        if not documents:
            return 0

        # 2. Split documents into chunks
        chunks = self.chunker.split_documents(documents)

        # 3. Add chunks to vector store
        self.vector_store.add_documents(chunks)

        # Invalidate cached chain so new retriever instance is picked up
        self._rag_chain = None
        return len(chunks)

    def _build_chain(self):
        """Construct the LCEL RAG chain."""
        retriever = self.vector_store.as_retriever()
        parser = StrOutputParser()

        # Chain: {context, question} -> prompt -> llm -> parser
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | parser
        )
        return chain

    @property
    def rag_chain(self):
        """Cached LCEL RAG chain."""
        if self._rag_chain is None:
            self._rag_chain = self._build_chain()
        return self._rag_chain

    def query(self, question: str) -> RAGResult:
        """
        Query the RAG pipeline with a user question.

        Args:
            question: The user query string.

        Returns:
            RAGResult containing answer, retrieved context chunks, sources, and timing.
        """
        start_time = time.time()

        # Retrieve relevant chunks directly to provide transparency to user
        retriever = self.vector_store.as_retriever()
        retrieved_docs = retriever.invoke(question)

        # Generate answer using LCEL chain
        answer = self.rag_chain.invoke(question)
        duration = round(time.time() - start_time, 3)

        # Extract unique sources and chunk details
        sources = list({doc.metadata.get("source", "unknown") for doc in retrieved_docs})
        chunks_detail = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ]

        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks_detail,
            execution_time_sec=duration,
        )

    def summarize(self, text_or_filepath: str) -> str:
        """Summarize text or document using the summarization prompt."""
        if os.path.exists(text_or_filepath):
            docs = DocumentLoader.load(text_or_filepath)
            content = " ".join([d.page_content for d in docs])
        else:
            content = text_or_filepath

        summarization_prompt = get_prompt_template("summarization")
        chain = summarization_prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": content})
