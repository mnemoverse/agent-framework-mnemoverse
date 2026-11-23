# Changelog

## [Unreleased]

### Fixed

- `mnemoverse_search(limit=N)` and `MnemoverseContextProvider(top_k=N)` show the model at most N memories, the first N in the server's order, even when the server returns more: `top_k` on a read is a requested count, not a hard bound.
- The README quickstart and `examples/quickstart.py` name `OPENAI_CHAT_MODEL` next to `OPENAI_API_KEY`: `OpenAIChatClient` raises `SettingNotFoundError` without a model. A test builds both offline.
- `MnemoverseContextProvider(top_k=...)` refuses a value that is not an int from 1 to 500 when it is created, instead of failing later inside a run.

## [0.1.0] - 2026-11-21

- `MnemoverseContextProvider`: recall from a Mnemoverse domain before each run, within a time budget, into the run's instructions; optional capture of the user's messages after each turn.
- `mnemoverse_tools()`: `mnemoverse_search`, `mnemoverse_save` and `mnemoverse_rate` function tools.
- `bind_user()` and `bind_domain()`: the domain a session reads and writes, kept in `session.state`.
- Requires `mnemoverse>=0.5.2,<0.6.0`.
- Recalled memory is shown to the model one line per memory, with `<` and `>` replaced by `‹` and `›`, so its text cannot close the memory block or pose as another memory.
- A session bound to a domain other than a fixed `domain=` is refused by every component given that `domain=`: the provider runs without memory or capture and logs an error, and such a tool answers the model with an error. One run never reads one user's memory and writes another's.
- A recall or capture that runs past `recall_timeout` goes on only while its event loop runs; when `asyncio.run()` cancels it on return, a warning says it did not complete.
