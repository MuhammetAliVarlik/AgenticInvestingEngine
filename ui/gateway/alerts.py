"""Owner alerts by e-mail: anomalies as they happen and a daily digest.

Alerts never contain personal data: no IP addresses, no codes, no content.
They name a code only by the first eight characters of its random id, so the
owner can revoke it. Each alert kind is sent at most once per hour; every
alert is also written to the log, so nothing is lost without SMTP settings.

Settings (environment variables):

* ``ALERT_EMAIL_TO`` - the owner's address; alerts are off when it is empty.
* ``SMTP_HOST``, ``SMTP_PORT`` (default 587, STARTTLS; 465 uses implicit TLS),
  ``SMTP_USER``, ``SMTP_PASSWORD``, ``ALERT_EMAIL_FROM`` (default ``SMTP_USER``).
* ``DIGEST_HOUR_UTC`` - hour of the daily digest (default 6; empty disables it).
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage

from gateway.codes import CodeStore

logger = logging.getLogger("gateway.alerts")

ALERT_INTERVAL_SECONDS = 3600.0


@dataclass(frozen=True)
class SmtpSettings:
    to: str
    host: str
    port: int
    user: str
    password: str
    sender: str

    @classmethod
    def from_env(cls) -> SmtpSettings | None:
        to = os.getenv("ALERT_EMAIL_TO", "").strip()
        host = os.getenv("SMTP_HOST", "").strip()
        if not to or not host:
            return None
        user = os.getenv("SMTP_USER", "").strip()
        return cls(
            to=to,
            host=host,
            port=int(os.getenv("SMTP_PORT", "587")),
            user=user,
            password=os.getenv("SMTP_PASSWORD", ""),
            sender=os.getenv("ALERT_EMAIL_FROM", user).strip() or user,
        )


class Alerter:
    def __init__(self, store: CodeStore, smtp: SmtpSettings | None) -> None:
        self._store = store
        self._smtp = smtp

    @property
    def enabled(self) -> bool:
        return self._smtp is not None

    async def alert(self, kind: str, message: str, *, key: str | None = None) -> None:
        """Log the event and e-mail it, at most once per hour for each ``key``."""
        logger.warning("alert %s: %s", kind, message)
        if not self._store.claim_alert(key or kind, every=ALERT_INTERVAL_SECONDS):
            return
        await self._send(f"[Investing Engine] Alert: {kind}", message)

    async def _send(self, subject: str, body: str) -> None:
        if self._smtp is None:
            return
        try:
            await asyncio.to_thread(_send_mail, self._smtp, subject, body)
        except (OSError, smtplib.SMTPException):
            logger.exception("Could not send the alert e-mail")

    # --- Daily digest -----------------------------------------------------------------

    def digest_text(self, *, now: float | None = None) -> str:
        now = now or time.time()
        day_ago = now - 86400
        return "\n".join(
            [
                "Daily summary for the last 24 hours (UTC).",
                "",
                f"Analyses started today: {self._store.analyses_today(now=now)}",
                f"Codes used in the last 24 hours: {self._store.active_codes(since=day_ago)}",
                f"Failed code entries: {self._store.failures(since=day_ago)}",
                f"Alerts: {self._store.alerts_since(day_ago)}",
                "",
                "This message contains no personal data.",
            ]
        )

    async def run_digest(self, hour_utc: int, *, poll_seconds: float = 300.0) -> None:
        """Send the digest once a day at ``hour_utc``. Runs until cancelled."""
        while True:
            now = datetime.now(timezone.utc)
            key = f"digest:{now.date().isoformat()}"
            if now.hour >= hour_utc and self._store.claim_alert(key, every=86400):
                await self._send("[Investing Engine] Daily summary", self.digest_text())
            await asyncio.sleep(poll_seconds)


def _send_mail(smtp: SmtpSettings, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = smtp.sender
    message["To"] = smtp.to
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    if smtp.port == 465:
        with smtplib.SMTP_SSL(smtp.host, smtp.port, context=context, timeout=15) as client:
            if smtp.user:
                client.login(smtp.user, smtp.password)
            client.send_message(message)
        return
    with smtplib.SMTP(smtp.host, smtp.port, timeout=15) as client:
        client.starttls(context=context)
        if smtp.user:
            client.login(smtp.user, smtp.password)
        client.send_message(message)


def digest_hour_from_env() -> int | None:
    raw = os.getenv("DIGEST_HOUR_UTC", "6").strip()
    if not raw:
        return None
    hour = int(raw)
    if not 0 <= hour <= 23:
        raise RuntimeError("DIGEST_HOUR_UTC must be between 0 and 23")
    return hour
