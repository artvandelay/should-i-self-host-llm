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
```

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
│   └── knownModels.json                  # Open-weight model catalog (bundled)
├── tests/
│   └── modelsDev.test.ts                 # Vitest tests for modelsDev.ts
└── README.md
```

## License

MIT