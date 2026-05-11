# agent/run_agent.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent_core import run_agent
from ingestion.chunker import Chunker
from run_rag import build_pipeline, build_vector_store, load_documents, LLMWrapper


def main():
    print("[Setup] Building pipeline...")
    documents = load_documents()
    chunks = Chunker().chunk(documents)
    vector_store = build_vector_store(chunks)
    pipeline = build_pipeline(vector_store, chunks)
    llm = LLMWrapper()

    # Test queries — mix of types
    test_queries = [
        "Why are dogs loyal?",           # → should route to rag
        "What is 15 multiplied by 4?",   # → should route to direct_answer
        "What is the capital of France?", # → should route to rag
        "Hello, how are you?",            # → should route to direct_answer
        "Tell me about something",        # → should route to clarify
    ]

    for query in test_queries:
        print("\n" + "="*60)
        run_agent(query, pipeline, llm)
        print("="*60)


if __name__ == "__main__":
    main()