"""
Deployment concerns: configuration, rate limiting, and admin control.

Kept apart from server.py so the protocol layer stays about the game and this
stays about running it somewhere real. Nothing here knows the rules of the
commons.

Environment:

    OVERGRAZE_DB          path to the SQLite file (put it on a persistent disk)
    OVERGRAZE_ADMIN_TOKEN secret for /admin/* -- unset means the routes refuse
    PORT                  what the platform tells you to bind
    OVERGRAZE_RATE        max tool calls per token per minute (default 120)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path

DEFAULT_RATE = 120          # tool calls per token per minute
WINDOW = 60.0


def db_path() -> Path:
    """Where state lives. On a platform, point this at the mounted volume.

    Test the string, not the Path: Path("") is Path(".") and therefore truthy,
    so an `or` on the Path silently resolves an unset variable to the working
    directory instead of falling back.
    """
    configured = os.environ.get("OVERGRAZE_DB", "").strip()
    return Path(configured) if configured else Path(__file__).with_name("overgraze.db")


def port() -> int:
    return int(os.environ.get("PORT", "8000"))


def host() -> str:
    # containers must bind all interfaces; local runs should not
    return os.environ.get("OVERGRAZE_HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")


def admin_token() -> str | None:
    tok = os.environ.get("OVERGRAZE_ADMIN_TOKEN", "").strip()
    return tok or None


class RateLimiter:
    """A sliding window per token. Somebody will loop harvest() a thousand times.

    Kept in memory on purpose: it protects this process from a runaway client,
    which is what the plan asks for. It is not a billing meter and does not
    survive a restart -- and with more than one instance each would enforce its
    own share, which is one more reason the barrier already pins us to one.
    """

    def __init__(self, per_minute: int | None = None):
        self.limit = per_minute or int(os.environ.get("OVERGRAZE_RATE", DEFAULT_RATE))
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Returns (allowed, seconds_until_a_slot_frees)."""
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        while q and now - q[0] > WINDOW:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(0.0, WINDOW - (now - q[0]))
        q.append(now)
        return True, 0.0

    def forget(self, key: str) -> None:
        self.hits.pop(key, None)
