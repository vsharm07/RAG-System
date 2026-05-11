import json
from llm import call_llm


def _ask_judge(prompt, retries=3):
    for attempt in range(retries):
        try:
            raw = call_llm(prompt)
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            # Sometimes llama3 adds text before the JSON — find the first {
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"[Judge] No JSON found, retry {attempt+1}/{retries}")
                continue
            return json.loads(clean[start:end])
        except Exception as e:
            print(f"[Judge] Failed attempt {attempt+1}/{retries}: {e}")
    return None


def evaluate_faithfulness(query, context_texts, answer):
    """
    Are all claims in the answer grounded in the retrieved context?
    Score: 0.0 (hallucinated) → 1.0 (fully grounded)
    """
    context = "\n\n".join(context_texts)

    prompt = f"""You are an evaluation judge for a RAG system.

Context retrieved:
{context}

Generated answer:
{answer}

Task: Check if every claim in the answer is supported by the context above.
An answer is unfaithful if it adds facts not present in the context, even if those facts are true.

Respond with valid JSON only, no explanation outside the JSON:
{{"faithfulness_score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

    result = _ask_judge(prompt)
    if result is None:
        return {"faithfulness_score": None, "reason": "judge failed"}
    return result


def evaluate_answer_relevance(query, answer):
    """
    Does the answer actually address the question asked?
    Score: 0.0 (irrelevant) → 1.0 (directly answers the question)
    """
    prompt = f"""You are an evaluation judge for a RAG system.

Question: {query}

Generated answer:
{answer}

Task: Score how well the answer addresses the question.
A score of 1.0 means the answer directly and completely answers the question.
A score of 0.0 means the answer is off-topic or doesn't address the question at all.

Respond with valid JSON only, no explanation outside the JSON:
{{"answer_relevance_score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

    result = _ask_judge(prompt)
    if result is None:
        return {"answer_relevance_score": None, "reason": "judge failed"}
    return result


def evaluate_context_precision(query, context_texts, answer):
    """
    Of the retrieved chunks, how many were actually useful for the answer?
    Score: 0.0 (all noise) → 1.0 (every chunk was used)
    """
    chunk_list = "\n\n".join([
        f"Chunk {i+1}: {text}" for i, text in enumerate(context_texts)
    ])

    prompt = f"""You are an evaluation judge for a RAG system.

Question: {query}

Retrieved chunks:
{chunk_list}

Generated answer:
{answer}

Task: For each chunk, decide if it contributed to the answer.
Then compute precision = (useful chunks) / (total chunks).

Respond with valid JSON only, no explanation outside the JSON:
{{"context_precision_score": <float 0.0-1.0>, "useful_chunks": <int>, "total_chunks": <int>, "reason": "<one sentence>"}}"""

    result = _ask_judge(prompt)
    if result is None:
        return {"context_precision_score": None, "reason": "judge failed"}
    return result


def evaluate_generation(query, context_texts, answer):
    """Run all three generation metrics. Returns combined result."""
    print(f"\n[Generation Eval] Query: {query}")

    faithfulness = evaluate_faithfulness(query, context_texts, answer)
    relevance = evaluate_answer_relevance(query, answer)
    precision = evaluate_context_precision(query, context_texts, answer)

    result = {
        "query": query,
        "answer": answer,
        "faithfulness": faithfulness.get("faithfulness_score"),
        "faithfulness_reason": faithfulness.get("reason"),
        "answer_relevance": relevance.get("answer_relevance_score"),
        "answer_relevance_reason": relevance.get("reason"),
        "context_precision": precision.get("context_precision_score"),
        "context_precision_reason": precision.get("reason"),
    }

    print(f"  Faithfulness:      {result['faithfulness']}")
    print(f"  Answer Relevance:  {result['answer_relevance']}")
    print(f"  Context Precision: {result['context_precision']}")

    return result