"""An agent that remembers a user across sessions.

Needs MNEMOVERSE_API_KEY (free at https://console.mnemoverse.com). The chat client here is
OpenAIChatClient from the agent-framework-openai package, which reads OPENAI_API_KEY; any
Agent Framework chat client works the same. From a clone of the repository:

    pip install agent-framework-mnemoverse agent-framework-openai
    python examples/quickstart.py
"""

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from mnemoverse import AsyncMnemoClient

from agent_framework_mnemoverse import MnemoverseContextProvider, bind_user, mnemoverse_tools


async def main() -> None:
    client = AsyncMnemoClient()
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a travel assistant. Save what the user asks you to remember.",
        context_providers=[MnemoverseContextProvider(client)],
        tools=mnemoverse_tools(client),
    )

    first = agent.create_session()
    bind_user(first, "ana")
    print(await agent.run("Please remember that I always take the window seat.", session=first))

    # A new session for the same user: before the run, the provider reads the memories
    # relevant to the message and adds them to the run's instructions.
    second = agent.create_session()
    bind_user(second, "ana")
    print(await agent.run("Which seat do I take on flights?", session=second))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
