"""
User Queries Module.

Handles defining, managing, and executing user queries against the RAG pipeline.
Features:
- Predefined query suites (domain-specific and benchmark tests)
- Single query runner with detailed chunk and source citation display
- Batch query runner for automated testing
- Interactive CLI mode for conversational question-answering
"""

import sys
from typing import List, Optional
from src.rag_pipeline import RAGPipeline, RAGResult
from config.config import config


# ============================================================================
# Predefined Sample Queries (Matching sample documents)
# ============================================================================
SAMPLE_QUERIES = [
    "What is cricket and how is it played?",
    "What are the means of dismissal for a batter in cricket?",
    "What formats of cricket are played at the international level?",
    "How long is a cricket pitch and what is at each end?",
    "What is the capital of France?",  # Edge case query (testing anti-hallucination / ungrounded behavior)
]


def print_banner(title: str, char: str = "=") -> None:
    """Print formatted section banner."""
    line = char * 70
    print(f"\n{line}\n  {title}\n{line}")


def display_rag_result(result: RAGResult, show_chunks: bool = True) -> None:
    """Pretty-print the RAG result with metadata and citations."""
    print(f"\n❓ Question: {result.question}")
    print(f"\n💡 Answer:\n{result.answer}")
    print(f"\n📚 Sources Cited: {', '.join(result.sources) if result.sources else 'None'}")
    print(f"⏱️  Execution Time: {result.execution_time_sec}s")

    if show_chunks and result.retrieved_chunks:
        print("\n🔍 Retrieved Context Chunks:")
        for idx, chunk in enumerate(result.retrieved_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "N/A")
            content_preview = chunk.get("content", "").replace("\n", " ").strip()
            if len(content_preview) > 160:
                content_preview = content_preview[:160] + "..."
            print(f"   [{idx}] Source: {source}")
            print(f"       Preview: \"{content_preview}\"")


def run_single_query(
    pipeline: RAGPipeline,
    query: str,
    show_chunks: bool = True,
) -> RAGResult:
    """
    Execute a single user query against the pipeline.

    Args:
        pipeline: Initialized RAGPipeline instance
        query: User question string
        show_chunks: Whether to display retrieved chunks

    Returns:
        RAGResult object
    """
    print_banner(f"Query: {query}", char="-")
    result = pipeline.query(query)
    display_rag_result(result, show_chunks=show_chunks)
    return result


def run_batch_queries(
    pipeline: RAGPipeline,
    queries: Optional[List[str]] = None,
    show_chunks: bool = False,
) -> List[RAGResult]:
    """
    Execute a batch of user queries sequentially.

    Args:
        pipeline: Initialized RAGPipeline instance
        queries: List of query strings (defaults to SAMPLE_QUERIES)
        show_chunks: Whether to display retrieved chunks

    Returns:
        List of RAGResult objects
    """
    query_list = queries or SAMPLE_QUERIES
    print_banner(f"Running Batch of {len(query_list)} Queries")

    results = []
    for i, q in enumerate(query_list, 1):
        print(f"\n[{i}/{len(query_list)}] Processing query...")
        res = run_single_query(pipeline, q, show_chunks=show_chunks)
        results.append(res)

    print_banner("Batch Execution Completed")
    return results


def run_interactive_mode(pipeline: RAGPipeline) -> None:
    """Start an interactive question-answering CLI session."""
    print_banner("Interactive RAG Query Session")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("👉 Enter your query: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting session. Goodbye!")
                break

            run_single_query(pipeline, user_input, show_chunks=True)
            print("-" * 70)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    print_banner("RAG Pipeline Query Runner")

    # In a full run, we initialize pipeline and execute queries
    try:
        pipeline = RAGPipeline()
        if len(sys.argv) > 1:
            # Query passed via command line arguments
            cli_query = " ".join(sys.argv[1:])
            run_single_query(pipeline, cli_query)
        else:
            # Default to running sample queries
            run_batch_queries(pipeline, SAMPLE_QUERIES, show_chunks=True)
    except Exception as e:
        print(f"\n[Notice] Could not run against live LLM: {e}")
        print("Please check your .env file with valid API keys.")
