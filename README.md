# agent-framework-mnemoverse

Long-term memory for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (Python) agents, backed by [Mnemoverse](https://mnemoverse.com). The memory lives in your Mnemoverse account, so what an agent saves here is also there in Claude Code, Cursor, ChatGPT and any other tool connected to the same account.

- **`MnemoverseContextProvider`** recalls the memories relevant to the user's latest message before every run and adds them to the run's instructions, within a time budget. If recall is slow or fails, the run goes on without memory.
- **`mnemoverse_tools()`** gives the agent three function tools: `mnemoverse_search`, `mnemoverse_save` and `mnemoverse_rate`. Rating a recalled memory as helpful or misleading changes what comes back next for similar questions.

## Install

```bash
pip install agent-framework-mnemoverse
```

Get a free API key at [console.mnemoverse.com](https://console.mnemoverse.com) and set it as `MNEMOVERSE_API_KEY`, or pass `api_key=`. With neither, building the provider or the tools raises `ValueError`.

The example below uses `OpenAIChatClient`, which comes in a separate package (`pip install agent-framework-openai`) and reads `OPENAI_API_KEY`; any Agent Framework chat client works the same way.

## Use

```python
import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from mnemoverse import AsyncMnemoClient

from agent_framework_mnemoverse import MnemoverseContextProvider, bind_user, mnemoverse_tools


async def main() -> None:
    client = AsyncMnemoClient()  # reads MNEMOVERSE_API_KEY
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[MnemoverseContextProvider(client)],
        tools=mnemoverse_tools(client),
    )
    session = agent.create_session()
    bind_user(session, "ana")  # this session reads and writes the domain user:ana

    print(await agent.run("Remember that I always take the window seat.", session=session))
    print(await agent.run("Which seat do I take on flights?", session=session))
    await client.close()


asyncio.run(main())
```

## Which memory a session uses

Every call reads and writes one Mnemoverse **domain**, a namespace inside your account such as `user:ana` or `project:support`.

- **One user:** pass `domain="user:ana"` to the provider and to `mnemoverse_tools()`; every session uses it.
- **Several users in one app:** leave `domain` out and bind each session with `bind_user(session, user_id)` (domain `user:<user_id>`) or `bind_domain(session, domain)`. Take `user_id` from your app's signed-in user on the server, never from client input or model output. The binding is kept in `session.state`, so it is saved with the session. A session with no binding gets no memory: it is never served from another session's domain.

A domain separates how memory is used, not who may read it: the API key can read every domain of its account. Give each app its own key, so you can revoke one without the others.

## Settings

| `MnemoverseContextProvider` argument | Default | What it does |
|---|---|---|
| `top_k` | 8 | Memories per recall |
| `min_relevance` | 0.3 | Lowest relevance, 0 to 1, of a recalled memory. A broad message ("what do you know about me?") scores lower than a specific one; lower this if broad messages recall nothing |
| `recall_timeout` | 8.0 | Seconds the run waits for recall before it goes on without it; also how long it waits for a capture. A call that runs longer is not cancelled, it finishes in the background |
| `context_prompt` | a heading and one line | Text above the recalled memories. The default line tells the model the memories are reference data, not instructions; keep it if you write your own |
| `capture` | `False` | Save the user's own messages after each turn. Prefer the `mnemoverse_save` tool, which saves only what the user asks to keep |

`mnemoverse_tools(save=False)` leaves out the save tool, and `rate=False` leaves out the rate tool. There is no delete tool.

## Errors

Memory never breaks a run. A timeout or a failed call is logged under the `agent_framework_mnemoverse` logger and the run continues without memory. A refusal (401 or 403) is logged with the server's reason, and the provider stops recall, or capture, for that domain only. Tool calls return the error to the model as a short message with the server's reason, so the agent can tell the user what happened. Long input is kept within the API limits: recall reads the last 5,000 characters of the message, capture saves the first 10,000, and the save tool refuses a memory over 10,000 characters.

## Links

- Python SDK: [mnemoverse on PyPI](https://pypi.org/project/mnemoverse/), [docs](https://mnemoverse.com/docs/api/python-sdk)
- Pricing and limits: [mnemoverse.com/pricing](https://mnemoverse.com/pricing)
- Issues: [github.com/mnemoverse/mnemoverse-sdk-python/issues](https://github.com/mnemoverse/mnemoverse-sdk-python/issues/new)

## License

MIT
