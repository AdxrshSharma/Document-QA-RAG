import argparse

from rag_core import CHUNK_OVERLAP, CHUNK_SIZE, ingest

ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
a = ap.parse_args()
print(f"Indexed {ingest(a.chunk_size, a.overlap)} chunks")
