"""Short-lived, owner-scoped storage for user uploads.

Uploads (price CSVs today, disclosure documents later) live in memory only,
expire automatically, and are bound to the principal that created them: a
caller that guesses another user's upload id still gets "not found".
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from cachetools import TTLCache

T = TypeVar("T")

DEFAULT_TTL_SECONDS = 2 * 60 * 60
DEFAULT_MAX_ITEMS = 512


@dataclass(frozen=True, slots=True)
class Upload(Generic[T]):
    id: str
    owner: str
    kind: str
    label: str
    payload: T
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UploadNotFoundError(KeyError):
    """The upload does not exist, has expired, or belongs to someone else."""


class UploadStore:
    def __init__(
        self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_items: int = DEFAULT_MAX_ITEMS
    ) -> None:
        self._items: TTLCache[str, Upload[Any]] = TTLCache(maxsize=max_items, ttl=ttl_seconds)
        self._lock = threading.Lock()

    def put(self, *, owner: str, kind: str, label: str, payload: T) -> Upload[T]:
        upload = Upload(
            id=secrets.token_urlsafe(16), owner=owner, kind=kind, label=label, payload=payload
        )
        with self._lock:
            self._items[upload.id] = upload
        return upload

    def get(self, upload_id: str, *, owner: str, kind: str) -> Upload[Any]:
        with self._lock:
            upload = self._items.get(upload_id)
        # Constant-time owner comparison; identical error for every failure
        # mode so ids cannot be probed.
        if upload is None or upload.kind != kind or not secrets.compare_digest(upload.owner, owner):
            raise UploadNotFoundError(upload_id)
        return upload

    def delete(self, upload_id: str, *, owner: str) -> None:
        with self._lock:
            upload = self._items.get(upload_id)
            if upload is not None and secrets.compare_digest(upload.owner, owner):
                del self._items[upload_id]
