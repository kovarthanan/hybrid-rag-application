import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import APIError

from .config import get_settings
from .graph_store import GraphStore
from .guardrails import FALLBACK_ANSWER, Guardrails
from .ingest import build_chunks
from .llm import LLM, LLMError
from .retriever import HybridRetriever
from .schemas import QueryRequest, QueryResponse
from .vector_store import VectorStore

state: dict = {}                              # loaded once, shared by all requests
pool = ThreadPoolExecutor(max_workers=16)     # for parallel guardrail + retrieval


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    chunks = build_chunks()
    vs, gs = VectorStore(), GraphStore()
    vs.load_or_build(chunks)                  # instant if cached in data/
    gs.load_or_build(chunks)
    state.update(
        vs=vs, gs=gs,
        retriever=HybridRetriever(vs, gs, s.top_k),
        guard=Guardrails(),
        llm=LLM(),
    )
    # Warm up connections so the first real query isn't slow
    try:
        vs.query_scores("warm up")
        state["guard"]._moderate("warm up")
    except Exception:
        pass
    yield
    pool.shutdown(wait=False)


app = FastAPI(title="Hybrid RAG API", version="1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Never leak stack traces or internals to the client
    return JSONResponse(status_code=500, content={"detail": "internal error"})


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(state["vs"].chunks),
        "nodes": len(state["gs"].nodes),
        "edges": len(state["gs"].edges),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    guard, retriever, llm = state["guard"], state["retriever"], state["llm"]
    t0, timings = time.perf_counter(), {}

    try:
        # 1) Input guardrail and retrieval run in parallel
        f_in = pool.submit(guard.check_input, req.question)
        f_ret = pool.submit(retriever.retrieve, req.question.strip())
        q, g_in = f_in.result()
        if not g_in.passed:
            f_ret.cancel()
            return QueryResponse(
                answer="Your question was blocked by the input guardrail.",
                blocked=True, guardrails=[g_in], timings_ms={"total": _ms(t0)},
            )
        r = f_ret.result()
        timings["guard_and_retrieval"] = _ms(t0)

        # 2) Relevance guardrail (free: uses the retrieval score)
        g_rel = guard.check_relevance(r.top_score)
        if not g_rel.passed:
            return QueryResponse(
                answer=FALLBACK_ANSWER, blocked=True, guardrails=[g_in, g_rel],
                trace=r.trace, timings_ms={**timings, "total": _ms(t0)},
            )

        # 3) LLM answer
        t1 = time.perf_counter()
        raw = llm.answer(q, r.context)
        timings["llm"] = _ms(t1)

    except (LLMError, APIError):
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")

    # 4) Output guardrail
    t2 = time.perf_counter()
    answer, g_out = guard.check_output(raw, r.context)
    timings["output_guard"] = _ms(t2)
    timings["total"] = _ms(t0)

    return QueryResponse(
        answer=answer,
        blocked=not g_out.passed,
        guardrails=[g_in, g_rel, g_out],
        trace=r.trace,
        timings_ms=timings,
    )


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int):
    chunks = state["vs"].chunks
    if not 0 <= chunk_id < len(chunks):
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunks[chunk_id]


@app.get("/graph")
def full_graph(limit: int = 300):
    """Whole knowledge graph, for an 'explore the KG' view in the frontend."""
    gs = state["gs"]
    limit = max(1, min(limit, 1000))
    return {
        "nodes": [
            {"id": k, "label": n["label"], "type": n["type"], "degree": len(gs.adj[k])}
            for k, n in gs.nodes.items()
        ],
        "edges": [
            {"source": e["source"], "target": e["target"], "relation": e["relation"]}
            for e in gs.edges[:limit]
        ],
    }