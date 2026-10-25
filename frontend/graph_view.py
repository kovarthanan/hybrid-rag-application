import re

SOURCE_COLORS = {"vector": "#60a5fa", "graph": "#fb923c", "both": "#4ade80"}
TYPE_COLORS = {
    "service": "#c4b5fd", "table": "#fde68a", "technology": "#a5f3fc",
    "feature": "#fbcfe8", "concept": "#e5e7eb",
}


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _nid(node_id: str) -> str:
    return "n_" + re.sub(r"\W", "_", node_id)


def build_dot(trace: dict, question: str) -> str:
    """Query -> chunks (vector side) and Query -> entities -> related entities (KG side)."""
    ctx = set(trace.get("context_chunk_ids", []))
    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent";',
        'node [fontname="Helvetica", fontsize=10, style=filled];',
        'edge [fontname="Helvetica", fontsize=8, color="#888888", fontcolor="#888888"];',
        f'q [label="{_esc(question[:40])}", shape=diamond, fillcolor="#7c3aed", fontcolor=white];',
    ]

    # Vector side: chunks (thick border = sent to the LLM)
    for h in trace.get("vector_hits", []):
        cid = f'c{h["chunk_id"]}'
        in_ctx = h["chunk_id"] in ctx
        lines.append(
            f'{cid} [label="chunk {h["chunk_id"]}\\n{h["score"]:.2f}", shape=box, '
            f'fillcolor="{SOURCE_COLORS[h["source"]]}", penwidth={3 if in_ctx else 1}, '
            f'color="{"#111111" if in_ctx else "#999999"}"];'
        )
        style = "dashed" if h["source"] == "graph" else "solid"
        lines.append(f"q -> {cid} [style={style}];")

    # KG side: entities (yellow = named in the question)
    for n in trace.get("graph", {}).get("nodes", []):
        fill = "#facc15" if n["matched"] else TYPE_COLORS.get(n["type"], "#e5e7eb")
        lines.append(f'{_nid(n["id"])} [label="{_esc(n["label"])}", shape=ellipse, fillcolor="{fill}"];')
        if n["matched"]:
            lines.append(f'q -> {_nid(n["id"])} [style=dotted, label="mentions"];')

    for e in trace.get("graph", {}).get("edges", []):
        lines.append(f'{_nid(e["source"])} -> {_nid(e["target"])} [label="{_esc(e["relation"])}"];')

    lines.append("}")
    return "\n".join(lines)