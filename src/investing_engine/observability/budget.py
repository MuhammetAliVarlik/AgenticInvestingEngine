"""Usage budgets: per-user daily quotas and a provider circuit breaker.

Free LLM tiers are rate-limited per organisation, so one user (or one bug)
can exhaust the whole demo. Two controls keep usage predictable:

* :class:`UsageStore` - per-principal daily counters (analyses and tokens)
  in SQLite. A request is refused up front once either budget is spent.
* :class:`CircuitBreaker` - after the provider reports a rate limit, new
  analyses fail fast for a cool-down period instead of queueing doomed
  requests.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from investing_engine.observability.tracing import hash_principal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    principal TEXT NOT NULL,
    day TEXT NOT NULL,
    analyses INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (principal, day)
);
"""


class BudgetExceededError(RuntimeError):
    def __init__(self, message: str, *, retry_after: int, reason: str = "budget") -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Usage:
    analyses: int
    tokens: int


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    return 86_400 - (now.hour * 3600 + now.minute * 60 + now.second)


class UsageStore:
    """Daily per-user counters. Principals are stored hashed, never in clear."""

    def __init__(
        self, db_path: str | Path, *, salt: str, max_analyses: int, max_tokens: int
    ) -> None:
        self._db_path = Path(db_path)
        self._salt = salt
        self.max_analyses = max_analyses
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _key(self, principal: str) -> str:
        return hash_principal(principal, self._salt)

    def get(self, principal: str) -> Usage:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT analyses, tokens FROM usage WHERE principal = ? AND day = ?",
                (self._key(principal), _today()),
            ).fetchone()
        return Usage(*row) if row else Usage(0, 0)

    def check(self, principal: str) -> Usage:
        """Raise :class:`BudgetExceededError` if today's budget is spent."""
        usage = self.get(principal)
        if usage.analyses >= self.max_analyses:
            raise BudgetExceededError(
                f"Daily limit of {self.max_analyses} analyses reached",
                retry_after=_seconds_until_midnight_utc(),
            )
        if usage.tokens >= self.max_tokens:
            raise BudgetExceededError(
                "Daily token budget reached", retry_after=_seconds_until_midnight_utc()
            )
        return usage

    def record(self, principal: str, *, tokens: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage (principal, day, analyses, tokens) VALUES (?, ?, 1, ?)
                ON CONFLICT(principal, day)
                DO UPDATE SET analyses = analyses + 1, tokens = tokens + excluded.tokens
                """,
                (self._key(principal), _today(), max(tokens, 0)),
            )


class CircuitBreaker:
    """Opens for ``cooldown`` seconds after the LLM provider rate-limits us."""

    def __init__(self, cooldown: float = 60.0) -> None:
        self._cooldown = cooldown
        self._open_until = 0.0

    def check(self) -> None:
        remaining = self._open_until - time.monotonic()
        if remaining > 0:
            raise BudgetExceededError(
                "The language model is rate-limited; please retry shortly",
                retry_after=int(remaining) + 1,
                reason="provider",
            )

    def trip(self) -> None:
        self._open_until = time.monotonic() + self._cooldown


RATE_LIMIT_ERRORS = frozenset({"RateLimitError", "ResponseError429", "TooManyRequests"})


def total_tokens(usage: dict[str, dict[str, int]]) -> int:
    return sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage.values())
