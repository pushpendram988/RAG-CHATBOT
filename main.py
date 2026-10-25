"""
Main Entrypoint for the Production-Grade RAG System.

Orchestrates:
1. Environment and configuration verification
2. Document ingestion and vector indexing
3. User query execution (single, batch, or interactive)
"""

import os
import numpy as np
import sys
from config.config import config
from src.rag_pipeline import RAGPipeline
from user_queries import (
    run_batch_queries,
    run_single_query,
    run_interactive_mode,
    SAMPLE_QUERIES,
    print_banner,
)


def main():
    print_banner("🚀 Starting Production-Grade RAG Pipeline System")
    print(f"• LLM Provider:        {config.llm.provider.upper()} ({config.llm.model_name})")
    print(f"• Embedding Provider:  {config.embedding.provider.upper()} ({config.embedding.model_name})")
    print(f"• Chunking Strategy:   {config.text_splitter.strategy} (size={config.text_splitter.chunk_size}, overlap={config.text_splitter.chunk_overlap})")
    print(f"• Vector Store:        {config.vector_db.provider.upper()} (dir: {config.vector_db.persist_directory})")

    # Ingestion Source (Default sample data)
    sample_doc_path = os.path.join(os.path.dirname(__file__), "data", "cricket.txt")

    try:
        # 1. Initialize RAG Pipeline
        pipeline = RAGPipeline(config=config, prompt_type="standard")

        # 2. Ingest document if sample exists
        if os.path.exists(sample_doc_path):
            print(f"\n📥 Ingesting document: {sample_doc_path}")
            num_chunks = pipeline.ingest(sample_doc_path)
            print(f"✅ Successfully ingested and indexed {num_chunks} chunks.")
        else:
            print(f"\n⚠️  Sample document not found at {sample_doc_path}. Skipping default ingestion.")

        # 3. Check CLI flags
        if "--interactive" in sys.argv or "-i" in sys.argv:
            run_interactive_mode(pipeline)
        elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            user_question = " ".join(sys.argv[1:])
            run_single_query(pipeline, user_question, show_chunks=True)
        else:
            print("\n🔍 Running default sample queries...")
            run_batch_queries(pipeline, SAMPLE_QUERIES[:3], show_chunks=True)

    except ValueError as ve:
        print(f"\n⚠️  Configuration Error: {ve}")
        print("💡 Hint: Copy .env.example to .env and configure your API keys.")
    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")


if __name__ == "__main__":
    main()
