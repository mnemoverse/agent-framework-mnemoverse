"""Shared plumbing: which Mnemoverse domain a session reads and writes, the client every call
goes through, time-boxed calls, and SDK errors turned into sentences a person can act on."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Mapping
from typing import Any, TypeVar

from mnemoverse import (
    AsyncMnemoClient,
    MnemoAuthError,
    MnemoError,
    MnemoRateLimitError,
    MnemoUnavailableError,
)

logger = logging.getLogger("agent_framework_mnemoverse")

T = TypeVar("T")

DEFAULT_SOURCE_ID = "mnemoverse"
"""The provider's ``source_id``; also the key under ``session.state`` holding the domain binding."""

DOMAIN_KEY = "domain"
LAST_IDS_KEY = "last_memory_ids"

# API limits (https://mnemoverse.com/docs/api/reference): a read query of 1-5,000 characters,
# a memory of 1-10,000 characters, a domain of up to 100 characters.
MAX_QUERY = 5000
MAX_CONTENT = 10000
MAX_DOMAIN = 100

KEYS_PAGE = "https://console.mnemoverse.com/dashboard/keys"


def _check_domain(domain: Any) -> str:
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    name: str = domain.strip()
    if len(name) > MAX_DOMAIN:
        raise ValueError(f"domain must be at most {MAX_DOMAIN} characters")
    return name


def bind_domain(session: Any, domain: str, *, source_id: str = DEFAULT_SOURCE_ID) -> None:
    """Record which Mnemoverse domain a session reads and writes, for example ``user:42``.

    Stored as ``session.state[source_id]["domain"]``, so it is saved with the session. A domain
    keeps one person's memories apart from another's inside one account; it is a namespace, not
    a permission boundary, because the key can read every domain of its account.
    """
    session.state.setdefault(source_id, {})[DOMAIN_KEY] = _check_domain(domain)


def bind_user(session: Any, user_id: str, *, source_id: str = DEFAULT_SOURCE_ID) -> None:
    """Shorthand for ``bind_domain(session, f"user:{user_id}")``.

    Take ``user_id`` from your app's signed-in user on the server, never from client input or
    model output: the key can read every domain of its account.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    bind_domain(session, f"user:{user_id.strip()}", source_id=source_id)


def domain_of(session: Any, source_id: str = DEFAULT_SOURCE_ID) -> str | None:
    """The domain bound to ``session`` with :func:`bind_domain`, if any."""
    slot = _slot(session, source_id)
    if slot is None:
        return None
    value = slot.get(DOMAIN_KEY)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _slot(session: Any, source_id: str) -> Mapping[str, Any] | None:
    state = getattr(session, "state", None)
    if not isinstance(state, Mapping):
        return None
    slot = state.get(source_id)
    return slot if isinstance(slot, Mapping) else None


def remember_ids(session: Any, ids: list[str], source_id: str = DEFAULT_SOURCE_ID) -> None:
    """Keep the ids of the memories last shown to the model in this session, for the rate tool."""
    state = getattr(session, "state", None)
    if isinstance(state, dict):
        state.setdefault(source_id, {})[LAST_IDS_KEY] = list(ids)


def last_ids(session: Any, source_id: str = DEFAULT_SOURCE_ID) -> list[str]:
    """The ids :func:`remember_ids` kept for this session, latest recall or search only."""
    slot = _slot(session, source_id)
    if slot is None:
        return []
    ids = slot.get(LAST_IDS_KEY)
    return [str(i) for i in ids] if isinstance(ids, (list, tuple)) else []


class NoDomain(Exception):
    """The provider has no fixed domain and the session names none."""


class DomainConflict(NoDomain):
    """The component has a fixed domain and the session is bound to a different one."""


class Scope:
    """One client for every call, and the domain each session reads and writes."""

    def __init__(
        self,
        client: AsyncMnemoClient | None = None,
        *,
        api_key: str | None = None,
        domain: str | None = None,
        source_id: str = DEFAULT_SOURCE_ID,
    ) -> None:
        if client is not None and api_key is not None:
            raise ValueError("pass either client= or api_key=, not both")
        self.source_id = source_id
        self.domain = _check_domain(domain) if domain is not None else None
        self._owned = client is None
        # MNEMOVERSE_API_KEY applies when no key is passed (the SDK reads it, and raises
        # ValueError right here when neither is set).
        self.client: AsyncMnemoClient = client if client is not None else AsyncMnemoClient(api_key=api_key)

    def domain_for(self, session: Any) -> str:
        """The fixed domain, or the one bound to this session. Never another session's.

        When both are there and they differ, :class:`DomainConflict`: the provider and the tools
        each have their own ``domain=``, and picking either could read one user's memory and
        write another's.
        """
        bound = domain_of(session, self.source_id)
        if self.domain is not None:
            if bound is not None and bound != self.domain:
                raise DomainConflict(
                    "this session is bound to a different Mnemoverse domain than the domain= this "
                    "component was given; pass domain= only in a single-user app, or leave it out "
                    "and bind each session with bind_user(session, user_id)"
                )
            return self.domain
        if bound is None:
            raise NoDomain(
                "no Mnemoverse domain is bound to this session; pass domain= to the provider "
                f"for a single-user app, or call bind_user(session, user_id) "
                f"(session.state[{self.source_id!r}]['domain'])"
            )
        return bound

    async def close(self) -> None:
        if self._owned:
            await self.client.close()


async def within(call: Awaitable[T], timeout: float, what: str) -> T:
    """Wait for an SDK call at most ``timeout`` seconds without cancelling it.

    The call runs on to the SDK's own timeout and reports its outcome to the client's circuit
    breaker; cancelling it instead could leave the breaker stuck half-open, after which every
    call on that client fails. A late failure is logged.
    """
    task = asyncio.ensure_future(call)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        task.add_done_callback(functools.partial(_log_late_failure, what))
        raise


def _log_late_failure(what: str, task: asyncio.Future[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "Mnemoverse: the %s that ran past its budget later failed: %s", what, describe_error(exc)
        )


def _server_says(exc: MnemoError) -> str:
    return (getattr(exc, "message", "") or str(exc)).strip().rstrip(".")


def describe_error(exc: BaseException) -> str:
    """One actionable message for an SDK failure, with the server's own reason when it gave one."""
    if isinstance(exc, MnemoAuthError):
        return (
            f"Mnemoverse refused the call (401 or 403): {_server_says(exc)}. A missing, mistyped or "
            f"revoked key is fixed at {KEYS_PAGE}; a 403 can also mean this key may not use this "
            "domain, such as a room it is not a member of."
        )
    if isinstance(exc, MnemoRateLimitError):
        if exc.retryable is False:
            return f"Mnemoverse refused the call (429): {_server_says(exc)}."
        wait = f" Try again in {exc.retry_after:.0f} s." if exc.retry_after else ""
        return f"Mnemoverse is rate limiting this account right now (429).{wait}"
    if isinstance(exc, MnemoUnavailableError):
        return "Mnemoverse could not be reached; the run goes on without memory. Try again shortly."
    if isinstance(exc, MnemoError):
        return f"Mnemoverse error: {_server_says(exc)}."
    return f"Mnemoverse call failed: {exc!r}"


# "<" and ">" become the single guillemets U+2039 and U+203A.
_ANGLE_BRACKETS = str.maketrans({"<": chr(0x2039), ">": chr(0x203A)})


def one_line(text: str) -> str:
    """Memory text as it is shown to a model: one line, angle brackets replaced by single
    guillemets (U+2039, U+203A).

    Memory text is written by other people and other tools (shared rooms, every AI tool on the
    account, capture). Kept on one line, it cannot start a line of its own that looks like
    another memory (``- [<id>] ...``) or a heading; without ``<`` and ``>`` it cannot close
    the ``<mnemoverse_memory>`` block it is shown in, in any spelling.
    """
    return " ".join(text.split()).translate(_ANGLE_BRACKETS)


def format_items(items: list[Any]) -> tuple[list[str], list[str]]:
    """``(lines, ids)``: one line per recalled memory with its id, so the rate tool can name it.

    Each memory's text goes through :func:`one_line`, so the ids on these lines are exactly the
    ids the server returned, one per line."""
    lines: list[str] = []
    ids: list[str] = []
    for item in items:
        content = one_line(getattr(item, "content", "") or "")
        if not content:
            continue
        atom_id = str(getattr(item, "atom_id", "") or "")
        ids.append(atom_id)
        lines.append(f"- [{atom_id}] {content}")
    return lines, ids
