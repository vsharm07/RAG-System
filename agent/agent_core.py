# agent/agent_core.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .tool_router import route_query
from llm import call_llm
from main import run_with_context


def direct_answer(query, llm):
    """Answer without retrieval — for math, common facts, greetings."""
    prompt = f"""Answer this question directly and concisely from your own knowledge.
Do not mention any knowledge base or retrieval system.

Question: {query}
Answer:"""
    return call_llm(prompt)


def clarify(query):
    """Ask for clarification when query is too vague."""
    return f"Could you clarify what you mean by '{query}'? I want to make sure I give you the most relevant answer."


def run_agent(query, pipeline, llm):
    print(f"\n[Agent] Query: {query}")
    print("[Agent] Step 1: Routing...")

    tool = route_query(query)

    print(f"[Agent] Step 2: Executing tool '{tool}'...")

    if tool == "rag":
        result = run_with_context(query, pipeline, llm)
        answer = result["answer"]
        print(f"[Agent] Retrieved {len(result['context_chunks'])} chunks")

    elif tool == "direct_answer":
        answer = direct_answer(query, llm)

    elif tool == "clarify":
        answer = clarify(query)

    else:
        print(f"[Agent] Unknown tool '{tool}', falling back to rag")
        result = run_with_context(query, pipeline, llm)
        answer = result["answer"]

    print(f"[Agent] Answer: {answer}")
    return answer
