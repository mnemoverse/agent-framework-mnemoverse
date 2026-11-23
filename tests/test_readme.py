"""The README quickstart and examples/quickstart.py, built offline exactly as written.

Only ``Agent.run`` is replaced, so no request leaves the process; every object the example
builds (the Mnemoverse client, the OpenAI chat client, the provider, the tools, the sessions)
is constructed for real, with only the API keys in the environment, as a reader starts out.
"""

from __future__ import annotations

import asyncio
import re
import runpy
from pathlib import Path
from typing import Any

import agent_framework_openai  # noqa: F401  (in the dev extra; a missing package must fail, not skip)
import pytest
from agent_framework import Agent

from agent_framework_mnemoverse import MnemoverseContextProvider

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EXAMPLE = ROOT / "examples" / "quickstart.py"

# Variables that would let a client find a model or another endpoint without the example saying so.
UNSET = (
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_MODEL",
    "AZURE_OPENAI_MODEL",
)


def _readme_quickstart() -> str:
    blocks = re.findall(r"^```python\n(.*?)^```", README.read_text(encoding="utf-8"), re.S | re.M)
    (block,) = [b for b in blocks if "OpenAIChatClient" in b]
    return str(block)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[Agent, Any, dict[str, Any]]]:
    """Only the variables the README tells a reader to set; ``Agent.run`` recorded, never sent."""
    monkeypatch.chdir(tmp_path)  # no stray .env file can supply a model
    for name in UNSET:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_live_offline_test_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "offline-test-model")
    seen: list[tuple[Agent, Any, dict[str, Any]]] = []

    async def fake_run(self: Agent, messages: Any = None, **kwargs: Any) -> str:
        seen.append((self, messages, kwargs))
        return "ok"

    monkeypatch.setattr(Agent, "run", fake_run)
    return seen


def _check(runs: list[tuple[Agent, Any, dict[str, Any]]], turns: int) -> None:
    from agent_framework.openai import OpenAIChatClient

    assert len(runs) == turns
    agent = runs[0][0]
    assert isinstance(agent.client, OpenAIChatClient)
    assert any(isinstance(p, MnemoverseContextProvider) for p in agent.context_providers)
    for _, _, kwargs in runs:
        assert kwargs["session"].state["mnemoverse"]["domain"] == "user:ana"


def test_the_readme_quickstart_builds_with_only_the_documented_variables_set(runs, capsys):
    exec(compile(_readme_quickstart(), str(README), "exec"), {"__name__": "readme_quickstart"})
    _check(runs, turns=2)
    assert capsys.readouterr().out == "ok\nok\n"


def test_the_quickstart_example_builds_with_only_the_documented_variables_set(runs):
    namespace = runpy.run_path(str(EXAMPLE), run_name="quickstart")
    asyncio.run(namespace["main"]())
    _check(runs, turns=2)


def test_without_a_model_the_chat_client_refuses_as_the_readme_says(monkeypatch, tmp_path):
    from agent_framework.openai import OpenAIChatClient

    monkeypatch.chdir(tmp_path)
    for name in (*UNSET, "OPENAI_CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test-key")
    with pytest.raises(Exception, match=r"(?i)model"):
        OpenAIChatClient()
