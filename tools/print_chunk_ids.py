# tools/print_chunk_ids.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_rag import load_documents
from ingestion.chunker import Chunker

documents = load_documents()
chunks = Chunker().chunk(documents)

for chunk in chunks:
    print(f"{chunk['chunk_id']} | doc: {chunk['doc_id']} | preview: {chunk['text'][:80]}")