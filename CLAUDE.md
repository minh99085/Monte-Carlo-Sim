# Project instructions for Claude

## Git workflow (user preference, 2026-07-19)

- When coding is finished, push directly to `main` on this repo
  (`minh99085/Monte-Carlo-Sim`). Do **not** create a new feature branch or
  open a PR for completed work — the owner has explicitly authorized direct
  pushes to main.
- Keep commits signed with author/committer `noreply@anthropic.com`.
- Run `python -m pytest -q` and make sure the full suite passes before any
  push.

## Communication style (user preference, 2026-07-19)

The owner is driving this project but is not a developer. From now on:

- **Explain in plain, easy-to-understand language.** Avoid jargon; when a
  technical term is unavoidable, say what it means in one short phrase.
- **Give step-by-step instructions**, numbered, one action per step, in the
  order they should be done.
- **Always end with a clear "What YOU need to do manually" section** listing
  anything the owner must do on their own machine / accounts / hardware
  (things Claude cannot do from here: edit secret files, point DNS, click in
  TradingView, buy a VPS, etc.). If nothing is needed, say so explicitly.
- **Act as the guide**: proactively tell the owner the next sensible step,
  don't wait to be asked.
