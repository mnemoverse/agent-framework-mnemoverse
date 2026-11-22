# Changelog

## [0.1.0] - 2026-11-21

- `MnemoverseContextProvider`: recall from a Mnemoverse domain before each run, within a time budget, into the run's instructions; optional capture of the user's messages after each turn.
- `mnemoverse_tools()`: `mnemoverse_search`, `mnemoverse_save` and `mnemoverse_rate` function tools.
- `bind_user()` and `bind_domain()`: the domain a session reads and writes, kept in `session.state`.
- Requires `mnemoverse>=0.5.2,<0.6.0`.
- Recalled memory is shown to the model one line per memory, with `<` and `>` replaced by `‹` and `›`, so its text cannot close the memory block or pose as another memory.
