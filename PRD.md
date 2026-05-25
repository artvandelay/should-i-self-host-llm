# PRD: API-vs-Self-Host LLM Cost Calculator

**Owner:** TBD
**Status:** Draft for hand-off
**Last updated:** 2026-05-20

---

## 1. One-line summary

A web calculator that, given a user's traffic profile and the API price they currently pay (or would pay), tells them **the largest open-weight model they can self-host for the same or less money** — and shows the full derivation so they can trust the answer.

## 2. Problem

Teams comparing closed APIs (OpenAI/Anthropic/etc.) vs self-hosting open-weight models have to assemble the math by hand: traffic shape, GPU $/hr, VRAM ceilings, quantization trade-offs, billing modes, cold starts. Existing online calculators are either tied to one vendor, hardcoded to one model size tier, or hide their math. The user wants a tool that is **generic, transparent, current, and answers a single sharp question**.

## 3. Target user

- Engineering leads / founders deciding "API or self-host?"
- Open to any cloud GPU vendor and any open-weight model family.
- Wants to drop in their own numbers and get a defensible recommendation in under a minute.

## 4. The question the tool answers

> **"If I'm willing to spend at most what the API would cost me, what is the largest open-weight model I can self-host instead, on what hardware, with what billing strategy?"**

Bigger model = better quality, so the recommender picks the **largest** model that fits under the API budget — not the cheapest.

## 5. Hard requirements (explicit from user)

### 5.1 Engine

- **Deterministic.** No LLM call in the recommendation hot path. User explicitly rejected an LLM-driven engine because it could go wonky and would require internet access.
- **Continuous model-size axis.** The tool must reason about any params count (e.g. 71B, 72B, 83B-MoE, 235B-A22B, 405B, 671B) — not snap to a small set of discrete tiers.
- **VRAM-based hardware selection.** Hardware tier is derived from `params_B × bytes_per_param + overhead_GB`, not from a hand-mapped table.
- **Quantization aware.** Support at minimum FP16 (2 B/param), INT8 (1 B/param), INT4 (0.5 B/param). Lower precision unlocks larger models on the same GPU.
- **Billing-mode aware.** Evaluate at least three billing modes per candidate config and pick the cheapest: always-on, hourly (warm during active hours only), per-second (scale-to-zero, billed only while serving).
- **Traffic-pattern aware.** Take a weekly traffic shape (uniform, business hours, bursty, custom) into account when computing billed hours.
- **Show the full math.** The headline answer must be accompanied by a step-by-step derivation: `API $/query → weekly API budget → cheapest billing mode → billed GPU hours → max affordable $/hr → eligible GPU tiers → chosen GPU → VRAM available → max params at chosen quant → recommendation`. Headline number and derivation number must always agree.

### 5.2 UI

- **Minimal default view.** Four to five top-level inputs maximum. Everything else lives in an "Advanced" expander.
- **Single headline recommendation block.** Plain English. Names the model size, an example named model, the hardware, the billing mode, the hot hours, the per-second hours, the weekly cost, the savings vs API.
- **Layperson-readable.** A non-engineer must be able to read the recommendation cold. No raw enum values (`per_second` → "scale-to-zero, pay only while serving"). No exposed LaTeX-trap characters.
- **Warning state.** When no self-host config beats the API, the box switches to a warning style and tells the user to stick with the API.

### 5.3 Pricing data

- **Versioned, not hardcoded.** All GPU rates and API per-query rates live in a single editable config file (`pricing.json` or equivalent), with a `last_updated` ISO date surfaced in the UI footer.
- **BYOK refresh.** A standalone CLI script must be able to pull the latest rates from vendor pricing pages and rewrite the config file. The user supplies their own API key (Firecrawl, currently) via a `.env` file at the repo root. No keys in code, no keys in UI fields, no hosted secrets.
- **Graceful degradation.** If a scrape fails or a regex misses, the old rate is preserved and the user is warned on stderr. The app never breaks because a vendor page changed layout.

### 5.4 Operational

- Local-first: runs as a single-command app on the user's machine. No required cloud account.
- API keys live in `.env` at the repo root (project convention).
- Python venvs live under `~/pyenv`; package manager is `uv` (project convention).
- Long-running scripts are runnable in a tmux session (project convention).

## 6. Soft requirements / nice-to-have

- One-time fine-tuning cost amortized over user-supplied N weeks, added to weekly self-host cost.
- Charts: cost-vs-volume (log-log), break-even table, "at this volume" comparison bar, reverse $/hr ceiling. These already exist in the prototype and should be preserved behind a "Detailed charts" expander.
- Display of example known-model names near the recommended params count (e.g. "158B → similar to Mistral Large 2, Llama 3.2 90B").
- Cold-start warmup penalty applied in per-second billing mode.

## 7. Out of scope (until requested)

- Multi-region / data residency selection.
- User accounts, saved scenarios, multi-tenant.
- Exporting results to PDF/CSV.
- Token-level pricing decomposition (input vs output tokens). Currently flattened to one $/query figure.
- Quality benchmarks (MMLU, HELM scores). The tool assumes "bigger = better" without a benchmark layer.
- Latency SLO modelling beyond an exponential mean.
- Training-cost modelling beyond a single one-time number.

## 8. Open questions (user has not specified — flag before building)

- **Tech stack.** Current prototype is Streamlit. User has not committed to staying on Streamlit. Open: Streamlit / Next.js / Gradio / pure-CLI / API-only?
- **Hosting target.** Local-only forever, or eventually a hosted SaaS? Affects auth, secrets handling, persistence.
- **Currency.** USD-only or USD/INR/EUR toggle? Prototype currently has a USD→INR conversion path; user has not confirmed whether non-USD users are a real audience.
- **Token-vs-query pricing.** API vendors charge per token; the prototype simplifies to per-query with an assumed average token count. Acceptable, or do we need full token-level decomposition?
- **MoE handling.** For mixture-of-experts (Qwen3-235B-A22B, DeepSeek-V3), should the recommender treat them by **total params** (memory bound) or **active params** (compute bound)? Both? Different answer per billing mode?
- **Multi-GPU sharding.** Should the tool consider tensor/pipeline parallelism beyond the prebuilt 2×/4×/8× H100 tiers? Or stop at "one node"?
- **Concurrency model.** Currently uses Poisson P95 to size a single GPU. Should the recommender suggest a replica count + horizontal scaling cost, or stay single-replica?
- **Quality threshold.** Right now any params count is fair game. Should the user be able to set a minimum (e.g. "≥ 30B equivalent") to filter out tiny models even if they fit cheaper?
- **Refresh cadence.** BYOK refresh is on-demand via CLI. Should there also be a scheduled/cron mode, or a GitHub Action that maintains a community pricing.json?
- **Vendor coverage.** Currently Modal, Lambda, Runpod, OpenAI, Anthropic. Should we also cover Together, Fireworks, Groq, DeepInfra, OpenRouter, Replicate, AWS/GCP/Azure on-demand and spot rates?
- **Closed-API self-host hybrid.** Some users want to route easy queries to a small self-host and hard queries to the API. Is that a future feature or out of scope?
- **Output format for downstream sharing.** A "copy this recommendation" or markdown export for putting in a design doc — wanted?
- **Persistence.** Should the app remember the user's last inputs between sessions?

## 9. Recommended success metrics

- Time to first usable recommendation < 30 seconds from cold open.
- Default view fits on a 13" laptop screen without scrolling.
- `pricing.json` `last_updated` never older than 30 days for the canonical hosted instance (if hosted).
- Zero crashes when a vendor page changes layout.
- A first-time visitor can correctly explain the recommendation to a colleague after reading only the headline and the "show the math" expander.

## 10. Existing prototype (reference, not a constraint)

A working Streamlit prototype lives at `/Users/jigar/projects/messing-around/AI-PH-FT-calculation` with:

- `app.py` — UI (minimal default view + Advanced expander + Detailed charts).
- `engine.py` — deterministic cost engine and `recommend_continuous()` that scans the params axis.
- `contracts.py` — frozen dataclasses for inputs, cost breakdown, recommendation, derivation.
- `models.py` — GPU tier list (now driven by `pricing.json`), known-model name database, legacy discrete model classes for the detailed charts.
- `pricing.py` / `pricing.json` — versioned price store.
- `refresh_prices.py` — BYOK Firecrawl-based price refresher CLI.
- `patterns.py` — traffic pattern shapes.

The next implementer is free to adopt, rewrite, or discard any of this. The PRD is the source of truth, not the prototype.
