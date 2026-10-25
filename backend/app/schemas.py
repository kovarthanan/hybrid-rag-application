from typing import Literal
from pydantic import BaseModel, Field


# ---------- Request ----------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


# ---------- Vector trace ----------
class ChunkHit(BaseModel):
    chunk_id: int
    score: float                                  # cosine similarity (0..1)
    preview: str                                  # first ~150 chars
    source: Literal["vector", "graph", "both"]    # who found this chunk


# ---------- Knowledge-graph trace ----------
class GraphNode(BaseModel):
    id: str
    label: str
    type: str          # service | table | technology | feature | concept
    matched: bool      # True = matched directly from the question


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str      # e.g. USES, OWNS_TABLE, STORES_IN, PUBLISHES


class GraphTrace(BaseModel):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


# ---------- Guardrails ----------
class GuardrailResult(BaseModel):
    stage: Literal["input", "output"]
    passed: bool
    reasons: list[str] = []


# ---------- Full trace + response ----------
class QueryTrace(BaseModel):
    vector_hits: list[ChunkHit] = []
    graph: GraphTrace = GraphTrace()
    context_chunk_ids: list[int] = []   # chunks actually sent to the LLM


class QueryResponse(BaseModel):
    answer: str
    blocked: bool = False
    guardrails: list[GuardrailResult] = []
    trace: QueryTrace = QueryTrace()
    timings_ms: dict[str, float] = {}   # retrieval, llm, guardrails, total