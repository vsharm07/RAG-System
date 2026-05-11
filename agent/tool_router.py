# agent/tool_router.py
import json
from llm import call_llm


TOOLS = {
    "rag": "Search the knowledge base and answer from retrieved documents. Use for specific factual questions about dogs, travel, programming, cities, technology.",
    "direct_answer": "Answer directly from general knowledge without retrieval. Use for math, common facts, greetings, or questions clearly outside the knowledge base.",
    "clarify": "Ask the user for clarification. Use when the query is too vague or ambiguous to answer reliably."
}

def build_routing_prompt(query):
    return f"""You are a query router. Your ONLY job is to classify query intent.
You are NOT answering the question.

Classify based on intent only:

"rag" — the user wants specific factual information that would 
        likely exist in a document or knowledge base.
        Signs: asks "what", "why", "how", "where" about specific topics,
        asks for explanations, definitions, or comparisons.

"direct_answer" — the user wants something that requires no documents:
        math calculations, greetings, opinions, small talk.
        Signs: numbers, arithmetic, social phrases.

"clarify" — the query is too vague or ambiguous to act on.
        Signs: no clear subject, missing context, open-ended with 
        no specific direction.

User query: {query}

Respond with JSON only:
{{"thought": "<one sentence reasoning about intent>", "tool": "<rag|direct_answer|clarify>"}}"""

def route_query(query):
    prompt = build_routing_prompt(query)
    raw = call_llm(prompt)

    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        result = json.loads(clean[start:end])
        tool = result.get("tool", "rag")
        thought = result.get("thought", "")
        print(f"[Router] Thought: {thought}")
        print(f"[Router] Tool selected: {tool}")
        return tool
    except Exception as e:
        print(f"[Router] Failed to parse, defaulting to rag: {e}")
        return "rag"