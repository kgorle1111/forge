"""Export the knowledge graph as a self-contained HTML file (offline, no CDN).

Topics form an inner ring, their concepts orbit them. Node color encodes
strength: green = high ease / few lapses, red = struggling.
"""
import json
import math
import time

from .memory import Memory

TEMPLATE = """<!doctype html><meta charset="utf-8"><title>Forge knowledge graph</title>
<body style="margin:0;background:#111;color:#eee;font:14px sans-serif">
<div style="padding:8px">The Forge — knowledge graph ({n} concepts). Green=strong, red=weak, ring=due for review.</div>
<canvas id="c" width="1200" height="800"></canvas>
<script>
const data = {data};
const ctx = document.getElementById("c").getContext("2d");
ctx.strokeStyle = "#444";
for (const e of data.edges) {{
  ctx.beginPath(); ctx.moveTo(e[0], e[1]); ctx.lineTo(e[2], e[3]); ctx.stroke();
}}
for (const n of data.nodes) {{
  ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 7);
  ctx.fillStyle = n.color; ctx.fill();
  if (n.due) {{ ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }}
  ctx.fillStyle = "#eee"; ctx.fillText(n.label, n.x + n.r + 4, n.y + 4);
}}
</script>"""


def _color(ease: float, lapses: int) -> str:
    strength = max(0.0, min(1.0, (ease - 1.3) / 1.5 - 0.15 * lapses))
    return f"rgb({int(220 * (1 - strength))},{int(200 * strength)},80)"


def layout(memory: Memory) -> dict:
    """Radial layout of the knowledge graph, shared by HTML export and dashboard."""
    rows = memory.stats()
    topics = sorted({r["topic"] for r in rows})
    cx, cy, now = 600, 400, time.time()
    nodes, edges = [], []
    for ti, t in enumerate(topics):
        ta = 2 * math.pi * ti / max(1, len(topics))
        tx, ty = cx + 220 * math.cos(ta), cy + 220 * math.sin(ta)
        nodes.append({"x": tx, "y": ty, "r": 14, "label": t, "color": "#4a90d9", "due": False})
        mine = [r for r in rows if r["topic"] == t]
        for ci, r in enumerate(mine):
            ca = ta + 2 * math.pi * (ci + 1) / (len(mine) + 1)
            x, y = tx + 110 * math.cos(ca), ty + 110 * math.sin(ca)
            edges.append([tx, ty, x, y])
            nodes.append({"x": x, "y": y, "r": 8, "label": r["concept"],
                          "color": _color(r["ease"], r["lapses"]),
                          "due": r["due"] <= now})
    return {"nodes": nodes, "edges": edges}


def export(path: str = "forge_graph.html", memory: Memory | None = None) -> str:
    memory = memory or Memory()
    data = layout(memory)
    n = sum(1 for node in data["nodes"] if node["r"] == 8)
    html = TEMPLATE.format(n=n, data=json.dumps(data))
    with open(path, "w") as f:
        f.write(html)
    return path
