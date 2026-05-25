# Should I self-host my LLM?

**[Live calculator →](https://artvandelay.github.io/should-i-self-host-llm/)**

Given your query volume and the API price you'd pay (GPT, Claude, Gemini, DeepSeek), find the largest open-weight model (Llama, Qwen, Mistral) you can self-host on Modal, Lambda, or Runpod for the same cost or less. Live pricing from [models.dev](https://models.dev), GPU rates refreshed nightly via a GitHub Action.

## Quick start

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). No API key needed.

## How live data works

Three data lanes keep the calculator current:

```
1. BUNDLED JSON
   src/pricing.json + src/knownModels.json are committed in the repo
   and included in every build. Always available, used as cold-start
   data and offline fallback.

2. LIVE models.dev FETCH
   Happens client-side on page load and on the "Refresh prices" click.
   Updates closed-API prices and the open-weight model catalog directly
   from models.dev/api.json (public, no key needed).

3. GITHUB ACTION NIGHTLY
   Runs scripts/refresh-prices.ts at 3 AM UTC. Pulls models.dev for
   APIs/models, scrapes GPU provider pages via Firecrawl, writes updated
   pricing.json and knownModels.json, commits back. Your deploy host
   picks up the change and rebuilds automatically.

4. LMARENA ELO (bundled, refreshed nightly)
   scripts/refresh-elo.ts pulls the LMArena leaderboard from the
   api.wulong.dev mirror and writes src/elo.json. The same nightly Action
   runs both refresh scripts. ELO is matched into both closed APIs and
   open-weight models with a version-strict fuzzy matcher; unmatched
   entries degrade gracefully (the UI shows "no LMArena score for this
   size class" rather than guessing).
```

## Modeling assumptions

The engine optimises for honest defaults that approximate measured behaviour without forcing every user to fill in 20 fields:

- **VRAM overhead** auto-scales with active model size (4 GB for ≤ 13B, 8 GB for 14–34B, 12 GB for 35–80B, 18 GB for 81–200B, 24 GB for 200B+). For MoE models the active-params count drives the tier, since KV cache scales with active params, not total. The Advanced settings "VRAM overhead floor" is the **minimum** — raise it if you have a measured KV-cache figure for your context length and concurrency.
- **Cold-start penalty** defaults to 30s and only matters under `cold_per_query` and `bursty` traffic patterns. Override per your vendor (Modal ~15–30s, Runpod serverless ~30–90s).
- **Throughput estimate** is calibrated to H100-class hardware: ~120 tok/s per 8B active params on an 80GB unit, scaling linearly with GPU count.
- **Inference only.** Engineering time, evals, monitoring, on-call, and any fine-tuning compute beyond what you enter in Advanced are **not** included. Use the one-time fine-tuning cost field to amortise that yourself.

The math is shown step-by-step under "Show the math" so every recommendation is auditable.

## Updating GPU prices

GPU rates are scraped via Firecrawl. To run locally:

1. Copy `.env.example` to `.env` and add your Firecrawl API key:
   ```
   FIRECRAWL_API_KEY=fc-...
   ```

2. Run the refresh script:
   ```bash
   npm run refresh
   ```

To enable the nightly GitHub Action, add `FIRECRAWL_API_KEY` as a repository secret:

`Settings → Secrets and variables → Actions → New repository secret`

Name: `FIRECRAWL_API_KEY`. The app works without it (GPU prices stay at their last-updated values).

## Project layout

```
.
├── .github/workflows/refresh-prices.yml  # Nightly + manual refresh Action
├── index.html                            # Vite entry point
├── package.json
├── scripts/refresh-prices.ts             # Node CLI: models.dev + Firecrawl
├── src/
│   ├── App.tsx                           # Main UI + live data wiring
│   ├── engine.ts                         # Deterministic cost engine
│   ├── modelsDev.ts                      # models.dev fetcher + extractors
│   ├── useLiveData.ts                    # React hook: live fetch + merge
│   ├── useUrlState.ts                    # Query-string-backed state
│   ├── main.tsx                          # React entry point
│   ├── index.css                         # Tailwind base
│   ├── pricing.json                      # GPU tiers + API rates (bundled)
│   ├── knownModels.json                  # Open-weight model catalog (bundled)
│   ├── elo.json                          # LMArena ELO snapshot (bundled)
│   └── eloMatch.ts                       # Version-strict ELO matcher
├── scripts/refresh-elo.ts                # Node CLI: pull LMArena ELO
├── tests/
│   ├── modelsDev.test.ts                 # Vitest tests for modelsDev.ts
│   └── elo.test.ts                       # Vitest tests for eloMatch.ts
└── README.md
```

## Credits

- **[models.dev](https://models.dev)** — the open, community-maintained database of LLM specs and pricing that powers this calculator's API rates and open-weight catalog. Huge thanks to the maintainers; without their work this tool would be a stale spreadsheet.
- **[Firecrawl](https://firecrawl.dev)** — used to refresh GPU hourly rates from cloud provider pricing pages.
- **[Hugging Face](https://huggingface.co)** — fallback for resolving model parameter counts.
- **[LMArena](https://lmarena.ai/leaderboard)** (formerly LMSYS Chatbot Arena) — Arena ELO scores that power the "Minimum quality" filter and the cost-vs-quality view. LMArena does not publish an official API; we read the daily-updated mirror at [api.wulong.dev/arena-ai-leaderboards](https://api.wulong.dev/arena-ai-leaderboards) (source repo: [oolong-tea-2026/arena-ai-leaderboards](https://github.com/oolong-tea-2026/arena-ai-leaderboards)). All ELO scores remain the work of LMArena users and the LMArena team.

## Maintainer

Jigar Doshi — [GitHub](https://github.com/artvandelay) · [Twitter/X @jigarkdoshi](https://twitter.com/jigarkdoshi)

## License

MIT