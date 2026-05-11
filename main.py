from generation.prompt import build_prompt
from utils.query_cache import query_cache


def build_rewrite_prompt(query):
    return f"""
Rewrite the user question into a short standalone search query for retrieval.
Preserve the original meaning.
Resolve ambiguous references only if the intent is clear from the question.
Return only the rewritten query.

Question:
{query}
"""


def rewrite_query(query, llm):
    try:
        rewritten_query = llm.invoke(build_rewrite_prompt(query)).strip()
    except Exception:
        return query

    if not rewritten_query:
        return query

    return rewritten_query.splitlines()[0].strip()


def run(query, pipeline, llm):
    cache_namespace = pipeline.get("cache_namespace")
    cached_result = query_cache.get(query, namespace=cache_namespace)
    if cached_result is not None:
        return cached_result["answer"]

    rewritten_query = rewrite_query(query, llm)
    retrieved_docs = pipeline["retriever"].retrieve(rewritten_query)

    # Unpack both results
    retrieval_docs, generation_docs = pipeline["reranker"].rerank(
        rewritten_query, retrieved_docs
    )

    prompt = build_prompt(query, generation_docs)  # ← generation_k chunks
    answer = llm.invoke(prompt)

    query_cache.set(
        query,
        {
            "rewritten_query": rewritten_query,
            "retrieved_docs": retrieval_docs,
            "reranked_docs": generation_docs,
            "answer": answer,
        },
        namespace=cache_namespace,
    )
    return answer

def run_with_context(query, pipeline, llm):
    rewritten_query = rewrite_query(query, llm)
    retrieved_docs = pipeline["retriever"].retrieve(rewritten_query)

    # Unpack both — use generation docs for LLM, retrieval docs for eval
    retrieval_docs, generation_docs = pipeline["reranker"].rerank(
        rewritten_query, retrieved_docs
    )

    prompt = build_prompt(query, generation_docs)  # ← only 2-3 chunks
    answer = llm.invoke(prompt)

    context_texts = [
        doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
        for doc in generation_docs
    ]

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "context_chunks": generation_docs,
        "context_texts": context_texts,
        "retrieval_docs": retrieval_docs,  # kept for retrieval eval
    }