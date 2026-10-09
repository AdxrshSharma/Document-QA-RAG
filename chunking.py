"""PDF loading and chunking (pure Python, no ML libraries)."""
import re
from pathlib import Path

from pypdf import PdfReader

# Split on sentence ends and on bullet symbols, so a bullet point or sentence is never cut in half.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\s*[●•▪◦]\s*")


def split_units(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [u.strip() for u in _SPLIT.split(text) if u and u.strip()]


def chunk_text(text, size=500, overlap=100):
    """Pack whole sentences/bullets into chunks of about `size` characters.
    The last few units of each chunk are repeated at the start of the next one (overlap),
    so a fact that sits on a boundary still appears intact in at least one chunk."""
    chunks, cur, cur_len = [], [], 0
    for u in split_units(text):
        if len(u) > size:                                  # one huge unit: hard-split it
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            step = max(size - overlap, 1)
            chunks += [u[i:i + size] for i in range(0, len(u), step)]
            continue
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            carry, n = [], 0
            for prev in reversed(cur):                     # carry trailing units as overlap
                if n + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                n += len(prev) + 1
            cur, cur_len = carry, n
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def load_pdf_chunks(data_dir="data", size=500, overlap=100):
    """Chunk page by page so every chunk keeps an exact page number for citations."""
    ids, docs, metas = [], [], []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for page_no, page in enumerate(reader.pages, start=1):
            for i, chunk in enumerate(chunk_text(page.extract_text() or "", size, overlap)):
                ids.append(f"{pdf.name}-p{page_no}-c{i}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
    return ids, docs, metas