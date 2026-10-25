import os

import requests
import streamlit as st

from graph_view import build_dot

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG", page_icon="🎧", layout="wide")


@st.cache_resource
def http() -> requests.Session:
    """One keep-alive session = no new TCP/TLS handshake per question."""
    return requests.Session()


@st.cache_data(ttl=30)
def health():
    try:
        return http().get(f"{BACKEND_URL}/health", timeout=3).json()
    except Exception:
        return None

def client_ip() -> str:
    """Visitor's real IP, from the header Traefik adds."""
    try:
        xff = st.context.headers.get("X-Forwarded-For", "")
    except Exception:
        xff = ""
    return xff.split(",")[0].strip() or "local"


def ask(question: str) -> dict:
    try:
        r = http().post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            headers={"X-Client-IP": client_ip()},
            timeout=60,
        )
    except requests.RequestException:
        return {"error": "Backend is not reachable."}
    if r.status_code == 422:
        return {"error": "Question must be between 3 and 500 characters."}
    if r.status_code != 200:
        try:
            return {"error": r.json().get("detail", f"Error {r.status_code}")}
        except ValueError:
            return {"error": f"Error {r.status_code}"}
    return r.json()


def get_chunk(cid: int) -> str:
    try:
        return http().get(f"{BACKEND_URL}/chunks/{cid}", timeout=5).json()["text"]
    except Exception:
        return "Could not load chunk."


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎧 Hybrid RAG")
    st.caption("Spotify-like architecture knowledge base")
    h = health()
    if h:
        st.success(f"Backend online · {h['chunks']} chunks · {h['nodes']} nodes · {h['edges']} edges")
    else:
        st.error("Backend offline")

    st.markdown(
        "**Graph legend**\n"
        "- 🟪 Query\n"
        "- 🟦 Chunk found by vector\n"
        "- 🟧 Chunk found by graph\n"
        "- 🟩 Chunk found by both\n"
        "- ⬛ Thick border = sent to LLM\n"
        "- 🟨 Entity named in question"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Layout ----------------
chat_col, trace_col = st.columns([5, 6], gap="large")

with chat_col:
    st.subheader("Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask about the architecture…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Searching vector store + knowledge graph…"):
            resp = ask(question)
        content = f"⚠️ {resp['error']}" if "error" in resp else resp["answer"]
        if resp.get("blocked"):
            content = f"🛡️ {content}"
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "resp": resp, "question": question}
        )
        st.rerun()

# ---------------- Trace panel (latest answer) ----------------
with trace_col:
    st.subheader("Query trace")
    last = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant" and "error" not in m["resp"]),
        None,
    )
    if not last:
        st.info("Ask a question to see which chunks and graph nodes it touched.")
    else:
        resp, trace = last["resp"], last["resp"].get("trace", {})
        tab_graph, tab_vec, tab_guard, tab_time = st.tabs(
            ["🕸️ Graph view", "📄 Vector hits", "🛡️ Guardrails", "⏱️ Timings"]
        )

        with tab_graph:
            if trace.get("vector_hits") or trace.get("graph", {}).get("nodes"):
                st.graphviz_chart(build_dot(trace, last["question"]))
            else:
                st.info("Blocked before retrieval, so there's nothing to show.")

        with tab_vec:
            ctx = set(trace.get("context_chunk_ids", []))
            rows = [
                {"chunk": h["chunk_id"], "score": h["score"], "found by": h["source"],
                 "sent to LLM": h["chunk_id"] in ctx, "preview": h["preview"]}
                for h in trace.get("vector_hits", [])
            ]
            if rows:
                st.dataframe(
                    rows, hide_index=True,
                    column_config={"score": st.column_config.ProgressColumn(
                        "score", min_value=0.0, max_value=1.0, format="%.3f")},
                )
                cid = st.selectbox("Read a full chunk", [r["chunk"] for r in rows])
                st.code(get_chunk(cid), language=None)
            else:
                st.info("No chunks retrieved.")

        with tab_guard:
            for g in resp.get("guardrails", []):
                label = f"**{g['stage'].upper()}**"
                if g["passed"]:
                    st.success(f"{label} passed" + (f" · {', '.join(g['reasons'])}" if g["reasons"] else ""))
                else:
                    st.error(f"{label} blocked · {', '.join(g['reasons'])}")

        with tab_time:
            t = resp.get("timings_ms", {})
            cols = st.columns(max(len(t), 1))
            for col, (k, v) in zip(cols, t.items()):
                col.metric(k.replace("_", " "), f"{v:.0f} ms")