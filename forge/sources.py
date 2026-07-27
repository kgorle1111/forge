"""Local source-document ingestion for Forge lessons."""
import base64
import binascii
import hashlib
import os
import re
import shutil
import subprocess
import tempfile


SOURCE_LIMIT = 8000
UPLOAD_LIMIT_BYTES = 2_000_000
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between",
    "could", "every", "first", "from", "have", "into", "more", "must", "only",
    "other", "should", "source", "that", "their", "there", "these", "this",
    "through", "under", "using", "where", "which", "while", "with", "would",
}


class SourceError(ValueError):
    """Raised when a source document cannot be read safely."""


def clamp_source(text: str, limit: int = SOURCE_LIMIT) -> str:
    """Normalize and bound user-provided source text before prompt injection."""
    text = text.replace("\x00", "").strip()
    return text[:limit]


def source_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def chunk_source(text: str, size: int = 900, overlap: int = 120) -> list[dict]:
    """Split source text into overlapping chunks with stable local citations."""
    text = clamp_source(text, SOURCE_LIMIT)
    if not text:
        return []
    chunks, start, idx = [], 0, 1
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        body = text[start:end].strip()
        if body:
            chunks.append({"id": idx, "label": f"S{idx}", "text": body})
            idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def source_manifest(docs: list[tuple[str, str]]) -> dict:
    out, all_text, next_id = [], [], 1
    for name, text in docs:
        text = clamp_source(text)
        if not text:
            continue
        chunks = chunk_source(text)
        for chunk in chunks:
            chunk["id"] = next_id
            chunk["label"] = f"S{next_id}"
            next_id += 1
        out.append({"name": name or "source", "digest": source_digest(text),
                    "text": text, "chunks": chunks})
        all_text.append(text)
    return {"docs": out, "digest": source_digest("\n".join(all_text))}


def manifest_text(manifest: dict) -> str:
    return "\n".join(doc.get("text", "") for doc in manifest.get("docs", []))


def relevant_chunks(manifest: dict, query: str, limit: int = 4) -> list[dict]:
    q = set(_terms(query))
    scored = []
    for doc in manifest.get("docs", []):
        for chunk in doc.get("chunks", []):
            terms = set(_terms(chunk.get("text", "")))
            score = len(q & terms) / max(1, len(q))
            scored.append((score, doc, chunk))
    scored.sort(key=lambda x: (-x[0], x[1].get("name", ""), x[2].get("id", 0)))
    rows = []
    for score, doc, chunk in scored[:limit]:
        rows.append({"source": doc.get("name", "source"),
                     "digest": doc.get("digest", ""),
                     "label": chunk.get("label", ""),
                     "chunk_id": chunk.get("id", 0),
                     "score": score,
                     "text": chunk.get("text", "")})
    return rows


def source_citation_block(manifest: dict, query: str, limit: int = 4) -> str:
    chunks = relevant_chunks(manifest, query, limit)
    if not chunks:
        return ""
    lines = ["Use and cite these source snippets when relevant:"]
    for c in chunks:
        excerpt = c["text"].replace("\n", " ")[:700]
        lines.append(f"[{c['label']}] {c['source']}: {excerpt}")
    return "\n" + "\n".join(lines)


def glossary_terms(text: str, limit: int = 20) -> list[dict]:
    counts: dict[str, int] = {}
    for term in _terms(text):
        counts[term] = counts.get(term, 0) + 1
    rows = [{"term": term, "count": count} for term, count in counts.items()]
    rows.sort(key=lambda r: (-r["count"], r["term"]))
    return rows[:limit]


def unsupported_terms(text: str, source_text: str, limit: int = 20) -> list[str]:
    source_terms = set(_terms(source_text))
    out = []
    for term in _terms(text):
        if term not in source_terms and term not in out:
            out.append(term)
        if len(out) >= limit:
            break
    return out


def _terms(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)
            if w.lower() not in STOPWORDS]


def read_source(path: str, limit: int = SOURCE_LIMIT) -> str:
    """Read a txt/md/pdf source document for `forge learn --from`.

    PDFs stay local: Forge shells out to `pdftotext` when available. No cloud OCR,
    no package dependency, and no attempt to silently ignore extraction failures.
    """
    if not path:
        return ""
    if not os.path.exists(path):
        raise SourceError(f"source file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXTENSIONS:
        return clamp_source(_read_pdf(path), limit)
    if ext and ext not in TEXT_EXTENSIONS:
        raise SourceError("source must be .txt, .md, .markdown, .text, or .pdf")
    with open(path, encoding="utf-8", errors="replace") as f:
        return clamp_source(f.read(), limit)


def read_uploaded_source(filename: str, data: str, limit: int = SOURCE_LIMIT) -> str:
    """Read a browser-uploaded source sent as base64 or a data URL."""
    filename = filename or "upload.txt"
    data = data or ""
    if "," in data and data[:40].lower().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as e:
        raise SourceError("uploaded source is not valid base64") from e
    if len(raw) > UPLOAD_LIMIT_BYTES:
        raise SourceError("uploaded source is too large (2 MB max)")
    ext = os.path.splitext(filename)[1].lower()
    if ext in PDF_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(raw)
            f.flush()
            return read_source(f.name, limit)
    if ext and ext not in TEXT_EXTENSIONS:
        raise SourceError("uploaded source must be .txt, .md, .markdown, .text, or .pdf")
    return clamp_source(raw.decode("utf-8", errors="replace"), limit)


def _read_pdf(path: str) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise SourceError("PDF ingestion requires pdftotext (Poppler) on PATH")
    try:
        proc = subprocess.run(
            [exe, "-layout", path, "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except OSError as e:
        raise SourceError(f"could not run pdftotext: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SourceError("pdftotext timed out while reading the PDF") from e
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[:1]
        suffix = f": {detail[0]}" if detail else ""
        raise SourceError(f"pdftotext failed{suffix}")
    return proc.stdout
