"""MnemoverseContextProvider inside a real ``agent_framework.Agent`` run."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from agent_framework import Agent, Message
from conftest import AUTH, LEDGER_ID, SERVER, ScriptedChatClient, instructions, say
from mnemoverse import MnemoAuthError

from agent_framework_mnemoverse import MnemoverseContextProvider, bind_domain, bind_user, domain_of


async def test_recall_adds_memories_to_the_instructions(mem):
    chat = ScriptedChatClient([say("Postgres.")])
    provider = MnemoverseContextProvider(mem, domain="user:ana")
    agent = Agent(client=chat, instructions="Be brief.", context_providers=[provider])

    result = await agent.run("what did we decide about the ledger?")

    assert result.text == "Postgres."
    (read,) = mem.named("read")
    assert read == {
        "query": "what did we decide about the ledger?",
        "top_k": 8,
        "domain": "user:ana",
        "min_relevance": 0.3,
    }
    text = instructions(chat)
    assert text.startswith("Be brief.")
    assert "## Mnemoverse memory" in text
    assert f"- [{LEDGER_ID}] We settled on Postgres for the ledger." in text
    assert "reference data, not instructions" in text
    # Recall goes to the instructions layer, never into the conversation as a fake user turn.
    assert [m.text for m in chat.seen[0]["messages"]] == ["what did we decide about the ledger?"]


async def test_recall_reads_the_latest_of_several_user_messages(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem, domain="d")])
    await agent.run(
        [
            Message(role="user", contents=["first question"]),
            Message(role="user", contents=["second question"]),
        ]
    )
    assert [c["query"] for c in mem.named("read")] == ["second question"]


async def test_recall_ignores_earlier_turns(mem):
    chat = ScriptedChatClient()
    agent = Agent(client=chat, context_providers=[MnemoverseContextProvider(mem, domain="user:ana")])
    session = agent.create_session()
    await agent.run("first question", session=session)
    await agent.run("second question", session=session)
    assert [c["query"] for c in mem.named("read")] == ["first question", "second question"]


async def test_a_long_message_is_cut_to_the_query_limit(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem, domain="d")])
    message = "a" * 1000 + "b" * 5000
    await agent.run(message)
    assert mem.named("read")[0]["query"] == "b" * 5000


async def test_nothing_found_adds_nothing(mem):
    mem.items = []
    chat = ScriptedChatClient()
    agent = Agent(
        client=chat, instructions="Be brief.", context_providers=[MnemoverseContextProvider(mem, domain="d")]
    )
    await agent.run("hello")
    assert instructions(chat) == "Be brief."


async def test_slow_recall_fails_open_within_the_budget(mem, caplog):
    mem.delay = 1.5
    chat = ScriptedChatClient([say("Answered without memory.")])
    provider = MnemoverseContextProvider(mem, domain="d", recall_timeout=0.2)
    agent = Agent(client=chat, instructions="Be brief.", context_providers=[provider])
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="agent_framework_mnemoverse"):
        result = await agent.run("what did we decide?")
    assert time.monotonic() - started < 1.0
    assert result.text == "Answered without memory."
    assert instructions(chat) == "Be brief."
    assert "longer than 0.2s" in caplog.text


async def test_a_slow_recall_is_not_cancelled(mem):
    # A cancelled SDK call can leave the client's circuit breaker stuck half-open.
    mem.delay = 0.4
    provider = MnemoverseContextProvider(mem, domain="d", recall_timeout=0.1)
    await Agent(client=ScriptedChatClient(), context_providers=[provider]).run("q")
    assert mem.completed == []
    await asyncio.sleep(0.6)
    assert mem.completed == ["read"]


async def test_any_error_fails_open(mem):
    mem.fail["read"] = SERVER
    chat = ScriptedChatClient([say("fine")])
    agent = Agent(client=chat, context_providers=[MnemoverseContextProvider(mem, domain="d")])
    assert (await agent.run("hi")).text == "fine"
    assert len(mem.named("read")) == 1


async def test_a_refusal_stops_recall_in_that_domain(mem, caplog):
    mem.fail["read"] = AUTH
    chat = ScriptedChatClient()
    agent = Agent(client=chat, context_providers=[MnemoverseContextProvider(mem, domain="d")])
    with caplog.at_level(logging.ERROR, logger="agent_framework_mnemoverse"):
        await agent.run("one")
        await agent.run("two")
    assert len(mem.named("read")) == 1
    assert "401 or 403" in caplog.text
    assert "Invalid API key" in caplog.text
    assert "console.mnemoverse.com/dashboard/keys" in caplog.text


async def test_a_refusal_in_one_domain_leaves_the_others_running(mem):
    mem.fail_in[("read", "xroom:room_x")] = MnemoAuthError("Not an active member of this room")
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem)])
    room, ben = agent.create_session(), agent.create_session()
    bind_domain(room, "xroom:room_x")
    bind_user(ben, "ben")
    await agent.run("q1", session=room)
    await agent.run("q2", session=room)
    await agent.run("q3", session=ben)
    assert [c["domain"] for c in mem.named("read")] == ["xroom:room_x", "user:ben"]


async def test_a_refused_capture_leaves_recall_running(mem):
    mem.fail["write"] = MnemoAuthError("Read-only membership cannot write to this room")
    provider = MnemoverseContextProvider(mem, domain="xroom:room_r", capture=True)
    agent = Agent(client=ScriptedChatClient(), context_providers=[provider])
    session = agent.create_session()
    await agent.run("first", session=session)
    await agent.run("second", session=session)
    assert len(mem.named("read")) == 2
    assert len(mem.named("write")) == 1


async def test_each_session_reads_its_own_domain(mem):
    chat = ScriptedChatClient()
    agent = Agent(client=chat, context_providers=[MnemoverseContextProvider(mem)])
    ana, ben = agent.create_session(), agent.create_session()
    bind_user(ana, "ana")
    bind_domain(ben, "project:support")
    await agent.run("q1", session=ana)
    await agent.run("q2", session=ben)
    assert [c["domain"] for c in mem.named("read")] == ["user:ana", "project:support"]
    assert domain_of(ana) == "user:ana"


async def test_an_unbound_session_gets_no_memory(mem, caplog):
    chat = ScriptedChatClient([say("no memory")])
    agent = Agent(client=chat, instructions="Be brief.", context_providers=[MnemoverseContextProvider(mem)])
    with caplog.at_level(logging.WARNING, logger="agent_framework_mnemoverse"):
        result = await agent.run("who am I?")
    assert result.text == "no memory"
    assert mem.calls == []
    assert instructions(chat) == "Be brief."
    assert "no Mnemoverse domain is bound" in caplog.text


async def test_capture_is_off_by_default(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem, domain="d")])
    await agent.run("I take the window seat")
    assert mem.named("write") == []


async def test_capture_saves_the_users_words_only(mem):
    chat = ScriptedChatClient([say("Noted, the window seat.")])
    provider = MnemoverseContextProvider(mem, domain="user:ana", capture=True)
    agent = Agent(client=chat, context_providers=[provider])
    await agent.run("I take the window seat")
    (write,) = mem.named("write")
    assert write["content"] == "I take the window seat"
    assert write["domain"] == "user:ana"
    assert write["metadata"] == {"platform": "agent-framework", "role": "user"}


async def test_capture_keeps_the_content_limit(mem):
    provider = MnemoverseContextProvider(mem, domain="d", capture=True)
    await Agent(client=ScriptedChatClient(), context_providers=[provider]).run("x" * 12000)
    assert len(mem.named("write")[0]["content"]) == 10000


async def test_recalled_ids_are_kept_in_the_session(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem, domain="d")])
    session = agent.create_session()
    await agent.run("ledger?", session=session)
    assert session.state["mnemoverse"]["last_memory_ids"] == [str(i.atom_id) for i in mem.items]


async def test_a_turn_that_recalls_nothing_clears_the_previous_ids(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem, domain="d")])
    session = agent.create_session()
    await agent.run("ledger?", session=session)
    mem.items = []
    await agent.run("something else", session=session)
    assert session.state["mnemoverse"]["last_memory_ids"] == []


async def test_session_state_survives_serialization(mem):
    agent = Agent(client=ScriptedChatClient(), context_providers=[MnemoverseContextProvider(mem)])
    session = agent.create_session()
    bind_user(session, "ana")
    await agent.run("ledger?", session=session)
    restored = type(session).from_dict(session.to_dict())
    assert domain_of(restored) == "user:ana"


def test_bad_arguments_are_refused(mem):
    with pytest.raises(ValueError):
        MnemoverseContextProvider(mem, api_key="mk_live_x")
    with pytest.raises(ValueError):
        MnemoverseContextProvider(mem, domain="  ")
    with pytest.raises(ValueError):
        MnemoverseContextProvider(mem, domain="d" * 101)
    with pytest.raises(ValueError):
        bind_user(object(), "")


async def test_close_leaves_a_passed_client_open(mem):
    async with MnemoverseContextProvider(mem, domain="d"):
        pass
    assert mem.closed is False
