from collections import Counter
from dataclasses import dataclass

import numpy as np

from .graph_store import GraphStore
from .schemas import ChunkHit, GraphTrace, QueryTrace
from .vector_store import VectorStore

RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class Retrieval:
    context: str          # text sent to the LLM
    top_score: float      # best vector score (used by the relevance guardrail)
    trace: QueryTrace     # everything the frontend visualizes


class HybridRetriever:
    def __init__(self, vs: VectorStore, gs: GraphStore, top_k: int = 5):
        self.vs, self.gs, self.top_k = vs, gs, top_k

    def _seeds_from_chunk(self, cid: int, limit: int = 5) -> list[str]:
        """Fallback when the question names no entity: use the best vector chunk's entities."""
        ids = [k for k, n in self.gs.nodes.items() if cid in n["chunk_ids"]]
        ids.sort(key=lambda k: len(self.gs.adj[k]), reverse=True)
        return ids[:limit]

    def retrieve(self, question: str) -> Retrieval:
        # 1) Vector side: one embedding call, one matrix multiply
        scores = self.vs.query_scores(question)
        vec_ranked = [int(i) for i in np.argsort(-scores)[: self.top_k]]

        # 2) Graph side: pure Python, ~1 ms
        matched = self.gs.match_nodes(question)
        seeds = matched or self._seeds_from_chunk(vec_ranked[0])
        graph, graph_ranked = self.gs.subgraph(seeds)
        graph_ranked = graph_ranked[: self.top_k]
        for n in graph["nodes"]:
            n["matched"] = n["id"] in matched   # highlight only what the question named

        # 3) Fuse with Reciprocal Rank Fusion
        fused = Counter()
        for ranked in (vec_ranked, graph_ranked):
            for rank, cid in enumerate(ranked):
                fused[cid] += 1 / (RRF_K + rank + 1)
        context_ids = [cid for cid, _ in fused.most_common(self.top_k)]

        # 4) Trace: every chunk either side touched, labeled by who found it
        vec_set, graph_set = set(vec_ranked), set(graph_ranked)
        hits = [
            ChunkHit(
                chunk_id=cid,
                score=round(float(scores[cid]), 4),
                preview=self.vs.chunks[cid]["text"][:150],
                source="both" if cid in vec_set and cid in graph_set
                else "vector" if cid in vec_set else "graph",
            )
            for cid in dict.fromkeys(vec_ranked + graph_ranked)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)

        context = "\n\n".join(
            f"[chunk {cid}]\n{self.vs.chunks[cid]['text']}" for cid in context_ids
        )
        return Retrieval(
            context=context,
            top_score=float(scores.max()),
            trace=QueryTrace(
                vector_hits=hits,
                graph=GraphTrace(**graph),
                context_chunk_ids=context_ids,
            ),
        )