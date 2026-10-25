import re

from openai import OpenAI

from .config import get_settings
from .schemas import GuardrailResult

# A secret marker placed inside the system prompt (llm.py will use it).
# If it ever shows up in an answer, the prompt has leaked.
PROMPT_CANARY = "rag-canary-7f3a"

FALLBACK_ANSWER = "I couldn't find that in the knowledge base."

INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|earlier)",
    r"(reveal|show|print|repeat|output)\s+(your|the)\s+(system|hidden|initial)\s+(prompt|instructions?)",
    r"\byou\s+are\s+now\b",
    r"\b(jailbreak|developer\s+mode|dan\s+mode)\b",
    r"</?\s*(system|assistant)\s*>",
]]

SECRET_PATTERNS = [re.compile(p) for p in [
    r"sk-[A-Za-z0-9_\-]{20,}",                    # OpenAI-style keys
    r"AKIA[0-9A-Z]{16}",                          # AWS access keys
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]]


def _clean(text: str) -> str:
    """Strip control characters (keeps newlines/tabs)."""
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text).strip()


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


class Guardrails:
    def __init__(self):
        s = get_settings()
        self.s = s
        self.client = OpenAI(api_key=s.openai_api_key.get_secret_value(), timeout=10)

    # ---------- shared ----------
    def _moderate(self, text: str) -> list[str]:
        """Returns flagged categories ([] = clean). Fails open, but says so."""
        if not self.s.enable_moderation:
            return []
        try:
            r = self.client.moderations.create(model="omni-moderation-latest", input=text).results[0]
            if r.flagged:
                return [f"moderation: {k}" for k, v in r.categories.model_dump().items() if v]
            return []
        except Exception:
            return ["moderation unavailable (skipped)"]  # informational, not a block

    # ---------- input ----------
    def check_input(self, question: str) -> tuple[str, GuardrailResult]:
        q = _clean(question)
        reasons = []

        if len(q) < 3:
            reasons.append("question too short")
        if len(q) > self.s.max_query_chars:
            reasons.append(f"question longer than {self.s.max_query_chars} chars")
        if any(p.search(q) for p in INJECTION_PATTERNS):
            reasons.append("possible prompt injection")
        if any(p.search(q) for p in SECRET_PATTERNS):
            reasons.append("question contains what looks like a secret key")

        if not reasons:  # only spend the API call if cheap checks passed
            reasons += self._moderate(q)

        blocking = [r for r in reasons if "unavailable" not in r]
        return q, GuardrailResult(stage="input", passed=not blocking, reasons=reasons)

    def check_relevance(self, top_vector_score: float) -> GuardrailResult:
        """Off-topic guard: uses the score retrieval already computed (free)."""
        ok = top_vector_score >= self.s.relevance_threshold
        reasons = [] if ok else [
            f"off-topic: best match {top_vector_score:.2f} < {self.s.relevance_threshold}"
        ]
        return GuardrailResult(stage="input", passed=ok, reasons=reasons)

    # ---------- output ----------
    def check_output(self, answer: str, context: str) -> tuple[str, GuardrailResult]:
        reasons = []

        if PROMPT_CANARY in answer:
            reasons.append("system prompt leak detected")
        if any(p.search(answer) for p in SECRET_PATTERNS):
            reasons.append("answer contains a secret-like string")

        # Grounding: most meaningful words in the answer should come from the context
        if not answer.startswith(FALLBACK_ANSWER):
            words = _content_words(answer)
            if words:
                overlap = len(words & _content_words(context)) / len(words)
                if overlap < self.s.grounding_threshold:
                    reasons.append(f"weak grounding: {overlap:.0%} of answer terms found in context")

        if not reasons:
            reasons += self._moderate(answer)

        blocking = [r for r in reasons if "unavailable" not in r]
        if blocking:
            return FALLBACK_ANSWER, GuardrailResult(stage="output", passed=False, reasons=reasons)
        return answer, GuardrailResult(stage="output", passed=True, reasons=reasons)