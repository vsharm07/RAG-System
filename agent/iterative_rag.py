# agent/iterative_rag.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import call_llm
from main import run_with_context


def check_sufficiency(query, answer, context_texts):
    """Ask LLM if the answer sufficiently addresses the query."""
    context = "\n\n".join(context_texts)

    prompt = f"""You are evaluating whether a generated answer sufficiently addresses a query.

Query: {query}
Context used: {context}
Generated answer: {answer}

Evaluate the answer on two criteria:
1. Does it directly address what was asked?
2. Is it specific enough or too vague?

Respond with JSON only:
{{"is_sufficient": true/false, "missing": "<what is still missing, or null if sufficient>", "reason": "<one sentence>"}}"""

    try:
        raw = call_llm(prompt)
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        result = json.loads(clean[start:end])
        return result
    except Exception as e:
        print(f"[Sufficiency Check] Failed to parse: {e}")
        return {"is_sufficient": True, "missing": None, "reason": "parse failed, assuming sufficient"}


def reformulate_query(original_query, answer, missing_info):
    """Generate a better query based on what's missing."""
    prompt = f"""You are helping improve a RAG system query.

Original query: {original_query}
Answer so far: {answer}
What is missing: {missing_info}

Generate a new, more specific search query that would retrieve 
the missing information. Return only the query, nothing else."""

    try:
        new_query = call_llm(prompt).strip()
        # Take first line only in case LLM adds explanation
        return new_query.splitlines()[0].strip()
    except Exception:
        return original_query


def iterative_rag(query, pipeline, llm, max_iterations=3):
    """
    Iteratively retrieve and generate until answer is sufficient
    or max iterations reached.
    """
    print(f"\n[Iterative RAG] Starting with query: '{query}'")
    print(f"[Iterative RAG] Max iterations: {max_iterations}")

    current_query = query
    best_answer = None
    all_context = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iterative RAG] Iteration {iteration}/{max_iterations}")
        print(f"[Iterative RAG] Query: '{current_query}'")

        # Retrieve and generate
        result = run_with_context(current_query, pipeline, llm)
        answer = result["answer"]
        context_texts = result["context_texts"]

        # Accumulate context across iterations
        all_context.extend(context_texts)
        best_answer = answer

        print(f"[Iterative RAG] Answer: {answer}")

        # Check if answer is sufficient
        sufficiency = check_sufficiency(query, answer, context_texts)
        print(f"[Iterative RAG] Sufficient: {sufficiency['is_sufficient']}")
        print(f"[Iterative RAG] Reason: {sufficiency['reason']}")

        if sufficiency["is_sufficient"]:
            print(f"[Iterative RAG] ✅ Answer accepted at iteration {iteration}")
            return {
                "answer": answer,
                "iterations": iteration,
                "final_query": current_query,
                "converged": True
            }

        # Not sufficient — reformulate if iterations remain
        if iteration < max_iterations:
            missing = sufficiency.get("missing", "more specific information")
            print(f"[Iterative RAG] Missing: {missing}")
            current_query = reformulate_query(query, answer, missing)
            print(f"[Iterative RAG] Reformulated query: '{current_query}'")

    # Max iterations reached
    print(f"[Iterative RAG] ⚠️ Max iterations reached, returning best answer")
    return {
        "answer": best_answer,
        "iterations": max_iterations,
        "final_query": current_query,
        "converged": False
    }