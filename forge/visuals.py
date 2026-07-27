"""Deterministic local visual lesson artifacts."""
import html
import math
import os
import re


STOPWORDS = {
    "about", "after", "again", "alpha", "because", "before", "being", "concept",
    "could", "first", "forge", "gamma", "lesson", "memory", "remember", "should",
    "their", "there", "these", "thing", "through", "under", "where", "which",
    "while", "would",
}


def render_visual_lesson(topic: str, concept: str, lesson: str,
                         missed: list[str] | None = None,
                         outdir: str | None = None) -> str:
    """Write a compact SVG concept map and return its path."""
    outdir = outdir or os.environ.get("FORGE_ARTIFACT_DIR",
                                      os.path.expanduser("~/.forge/visuals"))
    os.makedirs(outdir, exist_ok=True)
    stem = _slug(f"{topic}-{concept}")[:80] or "lesson"
    path = os.path.join(outdir, f"{stem}.svg")
    terms = _terms(" ".join([concept, lesson] + (missed or [])))[:6]
    if not terms:
        terms = ["definition", "mechanism", "example", "contrast"]
    missed_words = {w for m in missed or [] for w in re.findall(r"[a-z0-9]+", m.lower())}
    svg = _svg(topic, concept, terms, missed_words)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", text.lower()).strip("-")


def _terms(text: str) -> list[str]:
    seen, out = set(), []
    for w in re.findall(r"[a-z][a-z0-9]{3,}", text.lower()):
        if w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _svg(topic: str, concept: str, terms: list[str], missed_words: set[str]) -> str:
    cx, cy, radius = 480, 270, 160
    center = html.escape(concept[:64])
    title = html.escape(topic[:72])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#10131a"/>',
        f'<text x="480" y="44" text-anchor="middle" fill="#e8ebf1" '
        f'font-family="Georgia,serif" font-size="26">{title}</text>',
        '<circle cx="480" cy="270" r="78" fill="#171d28" stroke="#c9a35c" stroke-width="3"/>',
        f'<text x="480" y="264" text-anchor="middle" fill="#e8ebf1" '
        f'font-family="Arial,sans-serif" font-size="18" font-weight="700">{center}</text>',
        '<text x="480" y="290" text-anchor="middle" fill="#79839a" '
        'font-family="Arial,sans-serif" font-size="13">reconstruct from memory</text>',
    ]
    for i, term in enumerate(terms):
        a = 2 * math.pi * i / max(1, len(terms)) - math.pi / 2
        x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
        focus = term in missed_words
        fill = "#2a1820" if focus else "#151b24"
        stroke = "#d66b72" if focus else "#7fa6d9"
        label = html.escape(term)
        parts += [
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            'stroke="#2e3648" stroke-width="2"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="48" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>',
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" fill="#d7dde6" '
            'font-family="Arial,sans-serif" font-size="15" font-weight="700">'
            f'{label}</text>',
        ]
    parts.append("</svg>")
    return "\n".join(parts)
