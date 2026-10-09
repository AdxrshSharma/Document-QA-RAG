"""100% free, fully local document Q&A. No API keys, no internet after the first model download.

  question -> embed (sentence-transformers) -> top-k chunks (ChromaDB)
           -> DistilBERT extractive QA reads each chunk -> best answer span + citation

Extractive QA means the answer is an exact span copied from your document, so it cannot
invent facts. The trade-off: it cannot write fluent multi-sentence answers like an LLM.

Speed notes: models are loaded ONCE and cached (see _cache / warm_up), and the answer-span
search runs on plain Python lists instead of indexing tensors element by element.
"""
import json
import urllib.error
import urllib.request
import re

import chromadb
from chromadb.utils import embedding_functions

from chunking import load_pdf_chunks

DATA_DIR = "data"
DB_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"                        # ~90 MB
QA_MODEL = "distilbert-base-cased-distilled-squad"      # ~250 MB
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_SCORE = 0.10   # below this confidence we say "not found" (SQuAD v1 models always output *something*)
NO_ANSWER = "I couldn't find that in the documents."

_cache = {}   # holds the loaded models/clients so they are created only once per process


def _embedder():
    """Load the embedding model + vector DB client once, then reuse them."""
    if "ef" not in _cache:
        _cache["ef"] = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _cache["client"] = chromadb.PersistentClient(path=DB_DIR)
    return _cache["ef"], _cache["client"]


def _load_qa():
    if "qa_model" not in _cache:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        _cache["qa_tok"] = AutoTokenizer.from_pretrained(QA_MODEL)
        _cache["qa_model"] = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval()
    return _cache["qa_tok"], _cache["qa_model"]


def warm_up():
    """Load both models now so the first question is as fast as the rest."""
    _embedder()
    _load_qa()


def get_collection(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """One collection per chunking config so configs can be compared fairly."""
    ef, client = _embedder()
    return client.get_or_create_collection(
        name=f"docs2_{chunk_size}_{overlap}", embedding_function=ef,
        metadata={"hnsw:space": "cosine"})


def ingest(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    ids, docs, metas = load_pdf_chunks(DATA_DIR, chunk_size, overlap)
    if not docs:
        raise SystemExit("No text found. Put text-based PDFs in data/ "
                         "(or run: python make_sample_pdf.py)")
    col = get_collection(chunk_size, overlap)
    for i in range(0, len(docs), 500):
        col.upsert(ids=ids[i:i + 500], documents=docs[i:i + 500], metadatas=metas[i:i + 500])
    return len(docs)


def reset_index(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Delete everything that was indexed (used when you upload a fresh set of PDFs)."""
    _, client = _embedder()
    try:
        client.delete_collection(f"docs2_{chunk_size}_{overlap}")
    except Exception:
        pass   # nothing to delete yet


def list_indexed_docs(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """File names that currently have chunks in the index."""
    metas = get_collection(chunk_size, overlap).get(include=["metadatas"])["metadatas"] or []
    return sorted({m["source"] for m in metas})


def retrieve(question, k=TOP_K, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    col = get_collection(chunk_size, overlap)
    n = min(k, col.count())          # never ask for more chunks than exist
    if n == 0:
        return []
    res = col.query(query_texts=[question], n_results=n)
    return [{"text": d, "source": m["source"], "page": m["page"], "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def best_span(start_p, end_p, ctx_idx, max_answer_tokens=30):
    """Pick the (start, end) token pair with the highest start_prob * end_prob.
    Pure Python on lists, so it is fast. Only tokens inside the context are allowed."""
    allowed = set(ctx_idx)
    best, best_score = None, 0.0
    for s in ctx_idx:
        for e in range(s, s + max_answer_tokens):
            if e not in allowed:
                break
            score = start_p[s] * end_p[e]
            if score > best_score:
                best, best_score = (s, e), score
    return best, best_score


def extract_answer(question, context):
    """Find the best answer span inside one chunk. Returns (text, confidence 0-1)."""
    import torch
    tok, model = _load_qa()
    enc = tok(question, context, return_tensors="pt", truncation="only_second",
              max_length=384, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    with torch.inference_mode():
        out = model(**enc)
    start_p = out.start_logits.softmax(-1)[0].tolist()   # tensors -> lists once, up front
    end_p = out.end_logits.softmax(-1)[0].tolist()
    ctx_idx = [i for i, s in enumerate(seq_ids) if s == 1]   # tokens that belong to the context
    best, score = best_span(start_p, end_p, ctx_idx)
    if best is None:
        return "", 0.0
    return context[offsets[best[0]][0]:offsets[best[1]][1]], score



def ask(question, k=TOP_K, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Returns (result_dict, retrieved_chunks)."""
    chunks = retrieve(question, k, chunk_size, overlap)
    # Handle email questions directly
    if "email" in question.lower():
        for c in chunks:
            match = re.search(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                c["text"]
            )
            if match:
                return {
                    "answer": match.group(0),
                    "score": 1.0,
                    "source": c["source"],
                    "page": c["page"]
                }, chunks

    best = {
        "answer": NO_ANSWER,
        "score": 0.0,
        "source": None,
        "page": None
    }

    print(f"\nQuestion: {question}")

    for i, c in enumerate(chunks, 1):
        text, score = extract_answer(question, c["text"])

        print(f"\n--- Chunk {i} ---")
        print(f"Extracted answer: {text!r}")
        print(f"Confidence score: {score:.4f}")
        print(f"Source: {c['source']}, Page: {c['page']}")

        if score > best["score"]:
            best = {
                "answer": text,
                "score": score,
                "source": c["source"],
                "page": c["page"]
            }

    if best["score"] < MIN_SCORE:
        best["answer"] = NO_ANSWER

    print(f"\n--- Final selected answer: {best['answer']!r} ---")

    return best, chunks


def answer(question, k=TOP_K, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    best, chunks = ask(question, k, chunk_size, overlap)
    if best["source"] is None or best["answer"] == NO_ANSWER:
        return NO_ANSWER, chunks
    return f"{best['answer']}  [{best['source']} p.{best['page']}]", chunks


# ---------------------------------------------------------------------------
# Optional better answer engine: a free local LLM through Ollama (https://ollama.com)
# Handles questions extractive QA cannot: lists, summaries, "what does he do?", etc.
# Needs Ollama installed and a model pulled (see SETUP_VSCODE.md). Still no API key, no cost.
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
LLM_TOP_K = 6
LLM_SYSTEM = (
    "You answer questions using ONLY the provided context. Be concise. "
    "If the answer is not in the context, reply exactly: \"I couldn't find that in the documents.\" "
    "Cite sources like [file.pdf p.2]."
)


def ollama_answer(question, chunks, model=OLLAMA_MODEL, url=OLLAMA_URL, timeout=180):
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)
    payload = {
        "model": model, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "system", "content": LLM_SYSTEM},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Model '{model}' is not installed. Run in a terminal: ollama pull {model}") from e
        raise RuntimeError(f"Ollama returned an error ({e.code}).") from e
    except urllib.error.URLError as e:
        raise RuntimeError("Could not reach Ollama. Install it from https://ollama.com and make sure "
                           "it is running (open the Ollama app, or run: ollama serve).") from e


def ask_llm(question, k=LLM_TOP_K, model=OLLAMA_MODEL, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Retrieve chunks, then let a local LLM write the answer. Returns (text, chunks)."""
    chunks = retrieve(question, k, chunk_size, overlap)
    if not chunks:
        return NO_ANSWER, chunks
    return ollama_answer(question, chunks, model=model), chunks