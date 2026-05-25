# FT_ASSUMPTIONS

Moved alongside the fine-tuning engine code:
[`src/ft/ASSUMPTIONS.md`](src/ft/ASSUMPTIONS.md)

Rationale: the assumptions doc and the engine that implements them must stay
in sync. Co-locating them under `src/ft/` makes drift much harder — anyone
editing `src/ft/methods.ts` or `src/ft/cost.ts` sees the doc in the same
folder.
