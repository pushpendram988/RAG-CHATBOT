"""
Prompt Templates for RAG Pipeline.

Centralizes all system and user prompt templates used for:
- Standard Question Answering with context
- Strict Grounded Question Answering (anti-hallucination)
- Document Summarization
- Conversational RAG with chat history
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# ============================================================================
# 1. Standard RAG QA Prompt
# ============================================================================
STANDARD_RAG_TEMPLATE = """You are a helpful and knowledgeable AI assistant.
Answer the user's question based strictly on the provided context below.
Provide a clear, accurate, and concise answer.

Context:
{context}

Question:
{question}

Answer:"""

STANDARD_RAG_PROMPT = PromptTemplate(
    template=STANDARD_RAG_TEMPLATE,
    input_variables=["context", "question"],
)


# ============================================================================
# 2. Strict Grounded RAG Prompt (Anti-Hallucination)
# ============================================================================
STRICT_RAG_TEMPLATE = """You are a precise and faithful research assistant.
Use ONLY the following pieces of retrieved context to answer the question at the end.
If the answer cannot be determined strictly from the context, respond with:
"I cannot find the answer based on the provided documents."
Do NOT fabricate, extrapolate, or use outside knowledge.

Context:
{context}

Question:
{question}

Helpful & Factual Answer:"""

STRICT_RAG_PROMPT = PromptTemplate(
    template=STRICT_RAG_TEMPLATE,
    input_variables=["context", "question"],
)


# ============================================================================
# 3. Document Summarization Prompt
# ============================================================================
SUMMARIZATION_TEMPLATE = """You are an expert summarizer.
Please read the following text carefully and provide a concise, high-quality summary
highlighting the key points and core takeaways.

Text:
{text}

Summary:"""

SUMMARIZATION_PROMPT = PromptTemplate(
    template=SUMMARIZATION_TEMPLATE,
    input_variables=["text"],
)


# ============================================================================
# 4. Conversational RAG Prompt (with History)
# ============================================================================
CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you do not know the answer, say that you don't know.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_prompt_template(prompt_type: str = "standard") -> PromptTemplate:
    """
    Factory function to retrieve prompt templates by name.

    Args:
        prompt_type: 'standard', 'strict', or 'summarization'

    Returns:
        PromptTemplate instance.
    """
    prompts = {
        "standard": STANDARD_RAG_PROMPT,
        "strict": STRICT_RAG_PROMPT,
        "summarization": SUMMARIZATION_PROMPT,
    }
    if prompt_type not in prompts:
        raise ValueError(
            f"Unknown prompt type '{prompt_type}'. Available options: {list(prompts.keys())}"
        )
    return prompts[prompt_type]
