"""``mnemoverse_tools``: search, save and rate as Agent Framework function tools."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from agent_framework import AgentSession, FunctionInvocationContext, FunctionTool, tool
from mnemoverse import AsyncMnemoClient
from pydantic import Field

from ._common import (
    DEFAULT_SOURCE_ID,
    MAX_CONTENT,
    MAX_QUERY,
    NoDomain,
    Scope,
    describe_error,
    format_items,
    last_ids,
    remember_ids,
)

SEARCH_DESCRIPTION = "Search the user's long-term Mnemoverse memory. Returns memories, each with its id."
SEARCH_RATE_HINT = " Pass an id to mnemoverse_rate once you know whether the memory helped."


def mnemoverse_tools(
    client: AsyncMnemoClient | None = None,
    *,
    api_key: str | None = None,
    domain: str | None = None,
    source_id: str = DEFAULT_SOURCE_ID,
    save: bool = True,
    rate: bool = True,
) -> list[FunctionTool]:
    """Mnemoverse as Agent Framework function tools: ``mnemoverse_search``, ``mnemoverse_save``,
    ``mnemoverse_rate``.

    Pass the ``AsyncMnemoClient`` you also give :class:`MnemoverseContextProvider`, and close it
    yourself when the app stops; with ``api_key=`` (or ``MNEMOVERSE_API_KEY``) the tools open a
    connection pool of their own that stays open until the process ends. Each session uses a
    fixed ``domain`` or the domain bound to it with :func:`bind_user` or :func:`bind_domain`
    (the same binding the provider reads).

    ``save=False`` leaves out ``mnemoverse_save``; ``rate=False`` leaves out ``mnemoverse_rate``.
    There is no delete tool.
    """
    scope = Scope(client, api_key=api_key, domain=domain, source_id=source_id)

    def _session(ctx: FunctionInvocationContext | None) -> AgentSession | None:
        return ctx.session if ctx is not None else None

    @tool(
        name="mnemoverse_search",
        description=SEARCH_DESCRIPTION + (SEARCH_RATE_HINT if rate else ""),
        approval_mode="never_require",
    )
    async def mnemoverse_search(
        query: Annotated[str, Field(description="What to look for, in natural language.")],
        limit: Annotated[int, Field(description="How many memories, 1-50.", ge=1, le=50)] = 8,
        ctx: FunctionInvocationContext | None = None,
    ) -> str:
        session = _session(ctx)
        try:
            result = await scope.client.read(
                query[-MAX_QUERY:], top_k=limit, domain=scope.domain_for(session)
            )
        except NoDomain as exc:
            return f"Mnemoverse is not available: {exc}."
        except Exception as exc:  # noqa: BLE001 - the tool answers the model with the error
            return describe_error(exc)
        lines, ids = format_items(list(result.items))
        remember_ids(session, ids, source_id)
        if not lines:
            return "No memories in the user's Mnemoverse memory matched."
        return "\n".join(["Memories from the user's Mnemoverse memory:", *lines])

    @tool(name="mnemoverse_save", approval_mode="never_require")
    async def mnemoverse_save(
        content: Annotated[
            str, Field(description="The fact to keep, in the user's own words. One fact per call.")
        ],
        ctx: FunctionInvocationContext | None = None,
    ) -> str:
        """Save something to the user's Mnemoverse memory. Call this only when the user asks you
        to remember something or plainly means it; never save your own conclusions."""
        if len(content) > MAX_CONTENT:
            return (
                f"Mnemoverse did not store it: a memory holds up to {MAX_CONTENT:,} characters and "
                f"this one has {len(content):,}. Save the facts that matter, one per call."
            )
        try:
            result = await scope.client.write(
                content,
                domain=scope.domain_for(_session(ctx)),
                metadata={"platform": "agent-framework", "role": "user"},
            )
        except NoDomain as exc:
            return f"Mnemoverse is not available: {exc}."
        except Exception as exc:  # noqa: BLE001 - the tool answers the model with the error
            return describe_error(exc)
        if not getattr(result, "stored", False):
            reason = (getattr(result, "reason", "") or "").strip()
            return "Mnemoverse did not store it" + (f": {reason}." if reason else ".")
        return f"Saved to Mnemoverse (id {result.atom_id}); it can be recalled now."

    @tool(name="mnemoverse_rate", approval_mode="never_require")
    async def mnemoverse_rate(
        helpful: Annotated[
            bool, Field(description="True if the memories helped answer, False if they misled.")
        ],
        memory_ids: Annotated[
            list[str] | None,
            Field(description="Ids of the memories to rate. Leave out to rate the last ones recalled."),
        ] = None,
        ctx: FunctionInvocationContext | None = None,
    ) -> str:
        """Rate recalled memories as helpful or misleading. The rating changes what Mnemoverse
        recalls next for similar questions."""
        session = _session(ctx)
        ids: list[UUID | str] = list(memory_ids or last_ids(session, source_id))
        if not ids:
            return "No memories to rate: search first, or pass memory_ids."
        try:
            result = await scope.client.feedback(
                ids, 1.0 if helpful else -1.0, domain=scope.domain_for(session)
            )
        except NoDomain as exc:
            return f"Mnemoverse is not available: {exc}."
        except Exception as exc:  # noqa: BLE001 - the tool answers the model with the error
            return describe_error(exc)
        word = "helpful" if helpful else "misleading"
        return f"Rated {result.updated_count} of {len(ids)} memories as {word}."

    tools: list[FunctionTool] = [mnemoverse_search]
    if save:
        tools.append(mnemoverse_save)
    if rate:
        tools.append(mnemoverse_rate)
    return tools


__all__ = ["mnemoverse_tools"]
