"""Anonymous access codes: signed, time-limited, quota-bound and bound to one device.

A code carries its own id, expiry and quota, signed with ``ACCESS_CODE_SECRET``
(HMAC-SHA256), so validating a code needs no database and codes survive a
restart. A small SQLite store keeps what must be remembered:

* which device each code is bound to (first use claims the code);
* analyses started per code and per day (quotas, burst detection);
* revoked code ids;
* failed redemptions and alert bookkeeping.

No personal data is stored: no names, e-mail addresses or IP addresses. The
owner hands out codes without recording who received which one.

Command line (run where the gateway runs, with ``ACCESS_CODE_SECRET`` set)::

    python -m gateway.codes create --days 7 --quota 10 --count 3
    python -m gateway.codes list
    python -m gateway.codes revoke <code or code id>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import struct
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PREFIX = "IE-"
_ID_BYTES = 8
_SIG_BYTES = 16
_PAYLOAD = struct.Struct(">8sIH")  # id, expiry (unix seconds), quota
DEFAULT_DB_PATH = "/tmp/gateway/gateway.db"  # noqa: S108 - container scratch volume

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bindings (code_id TEXT PRIMARY KEY, device_hash TEXT NOT NULL,
                                     bound_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS analyses (code_id TEXT NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_analyses_code ON analyses(code_id, at);
CREATE INDEX IF NOT EXISTS idx_analyses_at ON analyses(at);
CREATE TABLE IF NOT EXISTS revoked (code_id TEXT PRIMARY KEY, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS failures (at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS alerts (key TEXT PRIMARY KEY, at REAL NOT NULL);
"""


class InvalidCodeError(Exception):
    """The code is malformed, forged, expired or revoked."""


@dataclass(frozen=True)
class AccessCode:
    code_id: str
    expires_at: int
    quota: int

    @property
    def identity(self) -> str:
        return f"code:{self.code_id}"

    @property
    def short_id(self) -> str:
        return self.code_id[:8]


class CodeSigner:
    def __init__(self, secret: str) -> None:
        if len(secret) < 32:
            raise RuntimeError("ACCESS_CODE_SECRET must be at least 32 characters")
        self._key = hashlib.sha256(secret.encode()).digest()

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()[:_SIG_BYTES]

    def create(self, *, days: float, quota: int) -> tuple[str, AccessCode]:
        if not 1 <= quota <= 1000:
            raise ValueError("quota must be between 1 and 1000")
        raw_id = secrets.token_bytes(_ID_BYTES)
        expires = int(time.time() + days * 86400)
        payload = _PAYLOAD.pack(raw_id, expires, quota)
        token = base64.urlsafe_b64encode(payload + self._sign(payload)).decode().rstrip("=")
        return PREFIX + token, AccessCode(raw_id.hex(), expires, quota)

    def parse(self, code: str, *, now: float | None = None) -> AccessCode:
        """Verify the signature and expiry. Raises :class:`InvalidCodeError`."""
        code = code.strip()
        if not code.startswith(PREFIX) or len(code) > 80:
            raise InvalidCodeError
        body = code[len(PREFIX) :]
        try:
            blob = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        except (ValueError, TypeError):
            raise InvalidCodeError from None
        if len(blob) != _PAYLOAD.size + _SIG_BYTES:
            raise InvalidCodeError
        payload, signature = blob[: _PAYLOAD.size], blob[_PAYLOAD.size :]
        if not hmac.compare_digest(signature, self._sign(payload)):
            raise InvalidCodeError
        raw_id, expires, quota = _PAYLOAD.unpack(payload)
        if expires <= (now if now is not None else time.time()):
            raise InvalidCodeError
        return AccessCode(raw_id.hex(), expires, quota)


def device_hash(device_secret: str) -> str:
    return hashlib.sha256(device_secret.encode()).hexdigest()


def _day_start(now: float) -> float:
    day = datetime.fromtimestamp(now, tz=timezone.utc).replace(hour=0, minute=0, second=0)
    return day.replace(microsecond=0).timestamp()


class CodeStore:
    def __init__(self, path: str | Path, *, revoked_ids: frozenset[str] = frozenset()) -> None:
        self._path = Path(path)
        self._static_revoked = revoked_ids

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=5)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- Devices ----------------------------------------------------------------------

    def bind(self, code_id: str, device: str, *, now: float | None = None) -> bool:
        """Bind ``code_id`` to ``device`` on first use. True if the device may use it."""
        digest = device_hash(device)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO bindings (code_id, device_hash, bound_at) VALUES (?, ?, ?)",
                (code_id, digest, now or time.time()),
            )
            row = conn.execute(
                "SELECT device_hash FROM bindings WHERE code_id = ?", (code_id,)
            ).fetchone()
        return bool(row) and hmac.compare_digest(row[0], digest)

    # --- Revocation -------------------------------------------------------------------

    def revoke(self, code_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO revoked (code_id, at) VALUES (?, ?)", (code_id, time.time())
            )

    def is_revoked(self, code_id: str) -> bool:
        if code_id in self._static_revoked or code_id[:8] in self._static_revoked:
            return True
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM revoked WHERE code_id = ?", (code_id,)).fetchone()
        return row is not None

    # --- Usage ------------------------------------------------------------------------

    def record_analysis(self, code_id: str, *, now: float | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (code_id, at) VALUES (?, ?)", (code_id, now or time.time())
            )

    def used(self, code_id: str, *, since: float = 0.0) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE code_id = ? AND at >= ?", (code_id, since)
            ).fetchone()
        return int(row[0])

    def analyses_today(self, *, now: float | None = None) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE at >= ?", (_day_start(now or time.time()),)
            ).fetchone()
        return int(row[0])

    def active_codes(self, *, since: float) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT code_id) FROM analyses WHERE at >= ?", (since,)
            ).fetchone()
        return int(row[0])

    # --- Failed redemptions -----------------------------------------------------------

    def record_failure(self, *, now: float | None = None) -> int:
        """Record a failed redemption; returns the number in the last hour."""
        now = now or time.time()
        with self._connect() as conn:
            conn.execute("INSERT INTO failures (at) VALUES (?)", (now,))
            conn.execute("DELETE FROM failures WHERE at < ?", (now - 86400,))
            row = conn.execute(
                "SELECT COUNT(*) FROM failures WHERE at >= ?", (now - 3600,)
            ).fetchone()
        return int(row[0])

    def failures(self, *, since: float) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM failures WHERE at >= ?", (since,)).fetchone()
        return int(row[0])

    # --- Alert bookkeeping ------------------------------------------------------------

    def claim_alert(self, key: str, *, every: float, now: float | None = None) -> bool:
        """True if an alert with ``key`` may be sent now (at most once per ``every`` s)."""
        now = now or time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT at FROM alerts WHERE key = ?", (key,)).fetchone()
            if row is not None and now - row[0] < every:
                return False
            conn.execute(
                "INSERT INTO alerts (key, at) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET at = excluded.at",
                (key, now),
            )
        return True

    def alerts_since(self, since: float) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE at >= ? AND key NOT LIKE 'digest:%'", (since,)
            ).fetchone()
        return int(row[0])

    def summary(self) -> list[tuple[str, int, float | None, bool, bool]]:
        """(code id, analyses, last analysis, bound, revoked) for every code seen."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ids.code_id,
                       (SELECT COUNT(*) FROM analyses a WHERE a.code_id = ids.code_id),
                       (SELECT MAX(at) FROM analyses a WHERE a.code_id = ids.code_id),
                       EXISTS (SELECT 1 FROM bindings b WHERE b.code_id = ids.code_id),
                       EXISTS (SELECT 1 FROM revoked r WHERE r.code_id = ids.code_id)
                FROM (SELECT code_id FROM bindings UNION SELECT code_id FROM analyses
                      UNION SELECT code_id FROM revoked) AS ids
                ORDER BY 3 DESC
                """
            ).fetchall()
        return [(r[0], int(r[1]), r[2], bool(r[3]), bool(r[4])) for r in rows]


def store_from_env() -> CodeStore:
    revoked = frozenset(
        x.strip().lower() for x in os.getenv("REVOKED_CODE_IDS", "").split(",") if x.strip()
    )
    store = CodeStore(os.getenv("GATEWAY_DB_PATH", DEFAULT_DB_PATH), revoked_ids=revoked)
    store.init()
    return store


# --- Command line ---------------------------------------------------------------------


def _code_id(value: str) -> str:
    """Accept a full code or a code id (or its 8-character prefix)."""
    value = value.strip()
    if value.startswith(PREFIX):
        return CodeSigner(os.environ["ACCESS_CODE_SECRET"]).parse(value, now=0).code_id
    return value.lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gateway.codes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="make new access codes")
    create.add_argument("--days", type=float, default=7.0)
    create.add_argument("--quota", type=int, default=10, help="analyses per code")
    create.add_argument("--count", type=int, default=1)
    sub.add_parser("list", help="show usage of codes seen by this gateway")
    revoke = sub.add_parser("revoke", help="make a code unusable immediately")
    revoke.add_argument("code", help="the code, its id or the 8-character id prefix")
    args = parser.parse_args(argv)

    if args.command == "create":
        signer = CodeSigner(os.environ.get("ACCESS_CODE_SECRET", ""))
        for _ in range(max(1, min(args.count, 100))):
            code, info = signer.create(days=args.days, quota=args.quota)
            expires = datetime.fromtimestamp(info.expires_at, tz=timezone.utc).isoformat()
            print(f"{code}  id={info.short_id}  quota={info.quota}  expires={expires}")
        return 0

    store = store_from_env()
    if args.command == "revoke":
        target = _code_id(args.code)
        matches = [row[0] for row in store.summary() if row[0].startswith(target)]
        for code_id in matches or [target]:
            store.revoke(code_id)
            print(f"revoked {code_id[:8]}")
        return 0

    print(f"{'id':10} {'used':>5}  {'last analysis (UTC)':20} {'bound':6} revoked")
    for code_id, used, last, bound, revoked in store.summary():
        when = (
            datetime.fromtimestamp(last, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            if last
            else "-"
        )
        print(
            f"{code_id[:8]:10} {used:>5}  {when:20} {'yes' if bound else 'no':6} "
            f"{'yes' if revoked else 'no'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
