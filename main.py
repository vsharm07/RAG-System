from ingestion.chunker import Chunker
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.prompt import build_prompt


def run(query, vector_store, documents, llm):

    # Step 1: Chunk (if not precomputed)
    chunker = Chunker()
    chunks = chunker.chunk(documents)
    print("chunksssssssssss", chunks)
    # Step 2: Retrievers
    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)

    # Step 3: Retrieve
    retrieved_docs = hybrid.retrieve(query)

    print("\n--- Retrieved Docs ---")
    for d in retrieved_docs[:3]:
        print(d["text"][:100])

    # Step 4: Rerank
    reranker = Reranker()
    top_docs = reranker.rerank(query, retrieved_docs)

    print("\n--- After Rerank ---")
    for d in top_docs:
        print(d["text"][:100])

    # Step 5: Prompt
    prompt = build_prompt(query, top_docs)

    # Step 6: LLM
    answer = llm.invoke(prompt)

    return answer