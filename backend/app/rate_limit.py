import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """
    In-memory sliding-window limiter (fine for a single uvicorn process).
    rules: list of (name, limit, window_seconds, per_client)
    A request is recorded only if it passes ALL rules.
    """

    def __init__(self, rules: list[tuple[str, int, int, bool]]):
        self.rules = rules
        self.hits: dict[tuple, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, str, int]:
        """Returns (allowed, rule_name, retry_after_seconds)."""
        now = time.monotonic()
        with self.lock:
            # 1) Check every rule without recording anything yet
            for name, limit, window, per_client in self.rules:
                q = self.hits[(name, client if per_client else "*")]
                while q and now - q[0] >= window:
                    q.popleft()
                if len(q) >= limit:
                    return False, name, int(window - (now - q[0])) + 1

            # 2) All passed: record the hit everywhere
            for name, _, _, per_client in self.rules:
                self.hits[(name, client if per_client else "*")].append(now)

            # 3) Housekeeping: drop empty entries so memory stays small
            if len(self.hits) > 10_000:
                for k in [k for k, q in self.hits.items() if not q]:
                    del self.hits[k]

        return True, "", 0