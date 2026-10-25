import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import get_settings


class VectorStore:
    """
    Tiny in-memory vector store.
    - Embeds chunks once with OpenAI, saves to disk (embeddings.npy + chunks.json).
    - On restart, loads from disk instantly (no API cost).
    - Search = one matrix multiplication -> milliseconds.
    """

    def __init__(self):
        s = get_settings()
        self.client = OpenAI(
            api_key=s.openai_api_key.get_secret_value(),
            timeout=s.llm_timeout_s,
        )
        self.model = s.embed_model
        data_dir = Path(s.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = data_dir / "embeddings.npy"
        self.chunks_path = data_dir / "chunks.json"
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize so dot product == cosine similarity
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def load_or_build(self, chunks: list[dict]) -> None:
        """Reuse saved embeddings if the chunks haven't changed; otherwise rebuild."""
        if self.emb_path.exists() and self.chunks_path.exists():
            saved = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            if saved == chunks:
                self.chunks = saved
                self.matrix = np.load(self.emb_path)
                return

        self.chunks = chunks
        self.matrix = self._embed([c["text"] for c in chunks])
        np.save(self.emb_path, self.matrix)
        self.chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
        )

    def query_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every chunk (index = chunk id)."""
        return self.matrix @ self._embed([query])[0]

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.query_scores(query)
        top = np.argsort(-scores)[:k]
        return [
            {"id": int(i), "text": self.chunks[i]["text"], "score": float(scores[i])}
            for i in top
        ]