import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from .config import get_settings

NODE_TYPES = {"service", "table", "technology", "feature", "concept"}

# Words too generic to identify a node on their own
GENERIC = {"service", "servic", "data", "system", "management", "the", "and",
           "for", "with", "api", "user", "core", "app"}

EXTRACT_PROMPT = """You extract a knowledge graph from software architecture notes.
Return JSON only:
{"entities":[{"name":str,"type":str}],
 "relations":[{"source":str,"relation":str,"target":str}]}
Rules:
- type is one of: service, table, technology, feature, concept
- relation is UPPER_SNAKE_CASE, e.g. USES, OWNS_TABLE, STORES_IN,
  PUBLISHES_EVENT, INTEGRATES_WITH, PROVIDES, PART_OF
- use short canonical names ("Playback & Session Service", "PostgreSQL", "playlists")
- every relation source/target must also appear in entities
- only include facts stated in the text"""


def _key(name: str) -> str:
    """Normalize a name into a node id: 'PostgreSQL ' -> 'postgresql'."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _tokens(text: str) -> set[str]:
    """Crude stemming (strip trailing 's') so 'payments' matches 'payment'."""
    return {t.rstrip("s") for t in _key(text).split() if len(t) > 2}


class GraphStore:
    def __init__(self):
        s = get_settings()
        self.client = OpenAI(
            api_key=s.openai_api_key.get_secret_value(),
            timeout=s.llm_timeout_s,
        )
        self.model = s.llm_model
        self.path = Path(s.data_dir) / "graph.json"
        self.nodes: dict[str, dict] = {}   # id -> {label, type, chunk_ids}
        self.edges: list[dict] = []        # {source, target, relation, chunk_id}
        self.adj: dict[str, list[int]] = defaultdict(list)
        self.node_tokens: dict[str, set[str]] = {}

    # ---------------- Build ----------------
    def _extract(self, chunk: dict) -> tuple[int, dict]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": chunk["text"]},
            ],
        )
        try:
            return chunk["id"], json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            return chunk["id"], {}

    def _add_node(self, name: str, ntype: str, cid: int) -> str | None:
        k = _key(name)
        if not k:
            return None
        ntype = ntype if ntype in NODE_TYPES else "concept"
        node = self.nodes.setdefault(k, {"label": name.strip(), "type": ntype, "chunk_ids": []})
        if cid not in node["chunk_ids"]:
            node["chunk_ids"].append(cid)
        return k

    def _merge(self, results: list[tuple[int, dict]]) -> None:
        self.nodes, self.edges, seen = {}, [], set()
        for cid, data in results:
            for e in data.get("entities", []):
                self._add_node(str(e.get("name", "")), str(e.get("type", "")), cid)
            for r in data.get("relations", []):
                s = self._add_node(str(r.get("source", "")), "concept", cid)
                t = self._add_node(str(r.get("target", "")), "concept", cid)
                rel = re.sub(r"[^A-Z_]", "", str(r.get("relation", "")).upper().replace(" ", "_"))
                rel = rel or "RELATED_TO"
                if s and t and s != t and (s, rel, t) not in seen:
                    seen.add((s, rel, t))
                    self.edges.append({"source": s, "target": t, "relation": rel, "chunk_id": cid})

    def _index(self) -> None:
        self.adj = defaultdict(list)
        for i, e in enumerate(self.edges):
            self.adj[e["source"]].append(i)
            self.adj[e["target"]].append(i)
        self.node_tokens = {k: _tokens(n["label"]) - GENERIC for k, n in self.nodes.items()}

    def load_or_build(self, chunks: list[dict]) -> None:
        sig = hashlib.sha256("".join(c["text"] for c in chunks).encode()).hexdigest()
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("sig") == sig:
                self.nodes, self.edges = saved["nodes"], saved["edges"]
                self._index()
                return

        # 16 chunks -> 16 LLM calls, run in parallel (~5-10 s, one time only)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(self._extract, chunks))
        self._merge(results)
        self.path.write_text(
            json.dumps({"sig": sig, "nodes": self.nodes, "edges": self.edges}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self._index()

    # ---------------- Query ----------------
    def match_nodes(self, query: str, limit: int = 8) -> list[str]:
        """Nodes named in the question (exact phrase, or all key tokens present)."""
        q_key, q_tok = f" {_key(query)} ", _tokens(query)
        hits = [
            k for k, toks in self.node_tokens.items()
            if (len(k) > 2 and f" {k} " in q_key) or (toks and toks <= q_tok)
        ]
        # Prefer well-connected nodes when the question matches many
        hits.sort(key=lambda k: len(self.adj[k]), reverse=True)
        return hits[:limit]

    def subgraph(self, seeds: list[str], max_nodes: int = 25) -> tuple[dict, list[int]]:
        """1-hop neighborhood of the seed nodes + chunk ids ranked by relevance."""
        node_ids = list(dict.fromkeys(seeds))
        edge_idx = []
        for s in seeds:
            for i in self.adj[s]:
                e = self.edges[i]
                other = e["target"] if e["source"] == s else e["source"]
                if other not in node_ids:
                    if len(node_ids) >= max_nodes:
                        continue
                    node_ids.append(other)
                edge_idx.append(i)
        edge_idx = list(dict.fromkeys(edge_idx))

        # Rank chunks: matched nodes count double, connecting edges count once
        score = Counter()
        for k in seeds:
            for cid in self.nodes[k]["chunk_ids"]:
                score[cid] += 2
        for i in edge_idx:
            score[self.edges[i]["chunk_id"]] += 1

        graph = {
            "nodes": [
                {"id": k, "label": self.nodes[k]["label"], "type": self.nodes[k]["type"], "matched": k in seeds}
                for k in node_ids
            ],
            "edges": [
                {"source": self.edges[i]["source"], "target": self.edges[i]["target"], "relation": self.edges[i]["relation"]}
                for i in edge_idx
            ],
        }
        return graph, [cid for cid, _ in score.most_common()]