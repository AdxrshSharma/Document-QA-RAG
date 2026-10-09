"""Compare chunking configs on two metrics (both free):
 1. hit@k      : is the page that holds the answer among the top-k retrieved chunks?
 2. answer acc : does the extracted answer contain the expected text? (--answers)
For a stronger portfolio number, replace eval_set.json with 20-30 questions about YOUR PDFs."""
import argparse
import json

from rag_core import ask, ingest, retrieve

CONFIGS = [(300, 60), (500, 100), (800, 150)]   # (chunk_size, overlap)


def hit_rate(data, k, cs, ov):
    hits = 0
    for ex in data:
        res = retrieve(ex["question"], k, cs, ov)
        hits += any(r["source"] == ex["source"] and r["page"] == ex["page"] for r in res)
    return hits / len(data)


def answer_accuracy(data, k, cs, ov):
    ok = 0
    for ex in data:
        best, _ = ask(ex["question"], k, cs, ov)
        ok += ex["expected"].lower() in best["answer"].lower()
    return ok / len(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--answers", action="store_true", help="also score the extracted answers")
    a = ap.parse_args()
    data = json.load(open("eval_set.json"))
    print(f"{'chunk':>6} {'overlap':>8} {'hit@k':>7} {'answer_acc':>11}")
    for cs, ov in CONFIGS:
        ingest(cs, ov)
        h = hit_rate(data, a.k, cs, ov)
        acc = answer_accuracy(data, a.k, cs, ov) if a.answers else float("nan")
        print(f"{cs:>6} {ov:>8} {h:>7.2f} {acc:>11.2f}")