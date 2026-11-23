"""Offline fixtures: a fake ``AsyncMnemoClient`` that records every call, and a scripted chat
client that the real ``agent_framework.Agent`` drives (tool calls included)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import MutableSequence
from typing import Any

import pytest
from agent_framework import (
    BaseChatClient,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from mnemoverse import FeedbackResponse, MemoryItem, MnemoAuthError, MnemoError, ReadResponse, WriteResponse

LEDGER_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
WINDOW_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")


def item(atom_id: uuid.UUID, content: str, domain: str = "user:ana") -> MemoryItem:
    return MemoryItem(
        atom_id=atom_id,
        content=content,
        relevance=0.8,
        similarity=0.8,
        valence=0.0,
        importance=0.5,
        source="test",
        concepts=[],
        domain=domain,
    )


def numbered(n: int) -> list[MemoryItem]:
    """``n`` memories in server order: ``memory 0``, ``memory 1``, ..., each with its own id."""
    return [item(uuid.UUID(int=i + 1, version=4), f"memory {i}") for i in range(n)]


class FakeMnemo:
    """Stands in for ``AsyncMnemoClient``: same method names and keywords, no network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.items: list[MemoryItem] = [
            item(LEDGER_ID, "We settled on Postgres for the ledger."),
            item(WINDOW_ID, "Ana takes the window seat."),
        ]
        self.fail: dict[str, BaseException] = {}
        # (call name, domain) -> error: a refusal that belongs to one domain only
        self.fail_in: dict[tuple[str, str], BaseException] = {}
        self.delay: float = 0.0
        self.stored = True
        self.closed = False
        # calls that ran to their end (not cancelled), in order
        self.completed: list[str] = []
        # when set, each read answers with the next list instead of self.items
        self.read_queue: list[list[MemoryItem]] = []

    async def _enter(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        self.completed.append(name)
        if name in self.fail:
            raise self.fail[name]
        key = (name, kwargs.get("domain"))
        if key in self.fail_in:
            raise self.fail_in[key]

    async def read(self, query: str, **kwargs: Any) -> ReadResponse:
        await self._enter("read", query=query, **kwargs)
        items = self.read_queue.pop(0) if self.read_queue else list(self.items)
        return ReadResponse(
            items=items, episodic_hit=False, query_concepts=[], expanded_concepts=[], search_time_ms=1.0
        )

    async def write(self, content: str, **kwargs: Any) -> WriteResponse:
        await self._enter("write", content=content, **kwargs)
        if not self.stored:
            return WriteResponse(stored=False, reason="duplicate of an existing memory")
        return WriteResponse(
            stored=True, atom_id=uuid.UUID("33333333-3333-4333-8333-333333333333"), importance=0.5
        )

    async def feedback(self, atom_ids: list[Any], outcome: float, **kwargs: Any) -> FeedbackResponse:
        await self._enter("feedback", atom_ids=list(atom_ids), outcome=outcome, **kwargs)
        return FeedbackResponse(updated_count=len(atom_ids), avg_valence=outcome)

    async def close(self) -> None:
        self.closed = True

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]


AUTH = MnemoAuthError("Invalid API key")
SERVER = MnemoError("server error", status=500)


class ScriptedChatClient(FunctionInvocationLayer[Any], BaseChatClient[Any]):
    """A chat client with no model: replays scripted responses and records every request."""

    def __init__(self, script: list[ChatResponse] | None = None) -> None:
        super().__init__()
        self.script = list(script or [])
        self.seen: list[dict[str, Any]] = []

    def _inner_get_response(  # type: ignore[override]
        self, *, messages: MutableSequence[Message], stream: bool, options: dict[str, Any], **kwargs: Any
    ) -> Any:
        async def _get() -> ChatResponse:
            self.seen.append({"messages": list(messages), "options": dict(options)})
            if self.script:
                return self.script.pop(0)
            return ChatResponse(messages=Message(role="assistant", contents=["ok"]))

        return _get()


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> ChatResponse:
    return ChatResponse(
        messages=Message(
            role="assistant",
            contents=[Content.from_function_call(call_id=call_id, name=name, arguments=args)],
        )
    )


def say(text: str) -> ChatResponse:
    return ChatResponse(messages=Message(role="assistant", contents=[text]))


def tool_results(chat: ScriptedChatClient) -> list[str]:
    """Text of every function result the model was shown, in order."""
    out: list[str] = []
    for message in chat.seen[-1]["messages"]:
        for c in message.contents:
            if c.type == "function_result":
                result = c.result
                if isinstance(result, list):
                    result = "".join(getattr(x, "text", "") or "" for x in result)
                out.append(str(result))
    return out


def instructions(chat: ScriptedChatClient, i: int = 0) -> str:
    return str(chat.seen[i]["options"].get("instructions") or "")


@pytest.fixture
def mem() -> FakeMnemo:
    return FakeMnemo()
