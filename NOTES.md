# How the math got right

This is a postmortem of the substantive bugs and assumption fixes that shaped the calculator. It's not a changelog of features — it's a tour of *what we got wrong on the way to getting it right*, narrated from the git log. If you're building anything that estimates LLM inference or fine-tuning costs, the mistakes here are likely the same ones lurking in your spreadsheet.

The repo was built across ~13 collaborative sessions with Claude (Anthropic) running in Cursor on May 25–26, 2026. **57 commits.** Almost every important commit message starts with `fix` or `replace`.

## The headline lessons

If you skim nothing else:

1. **LoRA/QLoRA save memory, not compute.** A common spreadsheet treats them as cheaper to run. The FLOPs are roughly the same — sometimes worse because of recompute. ([5051425](https://github.com/artvandelay/should-i-self-host-llm/commit/5051425))
2. **MoE active params drive compute. MoE total params drive VRAM.** Conflating them is the single most expensive modeling error for Mixtral / Qwen-MoE / DeepSeek-V3 — you can be off by 4–10× either way. ([5cffcf6](https://github.com/artvandelay/should-i-self-host-llm/commit/5cffcf6), [a857f04](https://github.com/artvandelay/should-i-self-host-llm/commit/a857f04))
3. **Continuous batching changes the throughput model.** Naive QPS / single-stream-throughput math gives wildly inflated replica counts. vLLM-style continuous batching is closer to "GPU sees one stream's worth of cost serving many concurrent users." ([120ad66](https://github.com/artvandelay/should-i-self-host-llm/commit/120ad66))
4. **Pick GPUs by total weekly cost, not lowest `$/hr`.** A "cheaper" GPU that needs 2× the replicas to hit your SLO costs more, not less. ([5e92e4f](https://github.com/artvandelay/should-i-self-host-llm/commit/5e92e4f))
5. **Don't scrape LMArena. Use the official dataset.** Saves you from breakage *and* from quietly publishing wrong Elo numbers. ([d8c23fb](https://github.com/artvandelay/should-i-self-host-llm/commit/d8c23fb), [de81621](https://github.com/artvandelay/should-i-self-host-llm/commit/de81621))

The rest of this doc walks through how those landed.

## The arc

### Day 1 (May 25): scaffolding + first stress

[`15dc2d4`](https://github.com/artvandelay/should-i-self-host-llm/commit/15dc2d4) → [`2735f40`](https://github.com/artvandelay/should-i-self-host-llm/commit/2735f40). React + Vite + TypeScript, models.dev as the live source of API prices. The first non-cosmetic commit already hints at the theme:

> [`54ab4bb`](https://github.com/artvandelay/should-i-self-host-llm/commit/54ab4bb) Auto-scale VRAM overhead with active model size; document assumptions

A fixed 20% overhead is wrong for a 7B model and wrong for a 405B model in different directions. The pattern of "every constant becomes a function of something else" repeated through the rest of the project.

The catalog was full of noise on day one — multimodal models pretending to be text-LLMs, multiple revisions of the same family. [`507cea0`](https://github.com/artvandelay/should-i-self-host-llm/commit/507cea0) and [`2735f40`](https://github.com/artvandelay/should-i-self-host-llm/commit/2735f40) cleaned this up with a family blocklist + auto-deprecate-by-(family,size) rule. Lesson: model catalog hygiene is invisible work but every downstream comparison depends on it.

### Day 2 morning: quality-aware recommendations

> [`184f663`](https://github.com/artvandelay/should-i-self-host-llm/commit/184f663) feat(engine): quality-aware recommendations alongside size

Cost-only recommendations lie. A 1B model is "cheaper" than a 70B model, but if the 1B can't do your task, you save nothing. Quality (Elo from LMArena) became a first-class axis next to cost. This was the first time the calculator started giving non-trivially useful answers.

Then immediately:

> [`de81621`](https://github.com/artvandelay/should-i-self-host-llm/commit/de81621) fix(elo): replace shady arena scraper with official LMArena dataset

The Elo scraper was returning numbers that "looked right" but came from a third-party mirror with stale data. The fix swapped to LMArena's published dataset. This bug recurred — [`d8c23fb`](https://github.com/artvandelay/should-i-self-host-llm/commit/d8c23fb) two days later applied the same fix in the FT engine after the rearchitect.

**Generalizable lesson:** if your cost model depends on a third-party quality signal, *own the source*. A shady scraper that returns plausible numbers is worse than no quality signal.

### Day 2 midday: the FT capex panel and what it exposed

> [`701963b`](https://github.com/artvandelay/should-i-self-host-llm/commit/701963b) Per-card fine-tuning capex panel with cumulative break-even chart

The fine-tune panel turned out to be where most bugs lived. The first round looked plausible. Then a sequence of fixes:

- [`9f7c945`](https://github.com/artvandelay/should-i-self-host-llm/commit/9f7c945) — docstring lied about what `FLOPS_PER_TOKEN_PER_PARAM` meant (6, not 2 — forward+backward+activations).
- [`5051425`](https://github.com/artvandelay/should-i-self-host-llm/commit/5051425) — **LoRA/QLoRA compute multipliers were wrong.** PEFT methods reduce *trainable parameters* and therefore *memory* and *optimizer state*; the forward + backward FLOPs through the base model are essentially unchanged. The fix: `compute_mult ≈ 1.0` for LoRA, ~1.0 for QLoRA (4-bit math has its own overhead). The earlier code treated LoRA as a `0.4× compute` discount, which was wishful thinking.
- [`44c794f`](https://github.com/artvandelay/should-i-self-host-llm/commit/44c794f) — added `experiments_multiplier` because nobody fine-tunes once and ships. A realistic project runs 2–5 experiments before the final.
- [`1f82307`](https://github.com/artvandelay/should-i-self-host-llm/commit/1f82307) + [`bb228e1`](https://github.com/artvandelay/should-i-self-host-llm/commit/bb228e1) — inference throughput and GPU selection both became `bf16_tflops`-aware. Before this, an L4 looked competitive with an H100 on `$/hr` alone.
- [`9e401cb`](https://github.com/artvandelay/should-i-self-host-llm/commit/9e401cb) — cluster overhead (single-GPU 1.10× → multi-GPU 1.35× → multi-node 1.70×) auto-picked from the FT VRAM footprint. NCCL is not free.

> [`9b63a43`](https://github.com/artvandelay/should-i-self-host-llm/commit/9b63a43) add FT_ASSUMPTIONS.md — full methodology, citations, and limitations

After enough fixes, the assumptions warranted their own document. [`FT_ASSUMPTIONS.md`](FT_ASSUMPTIONS.md) became the canonical reference — every constant in the engine has a citation. (The api-vs-selfhost-skill sister repo fetches this file at runtime.)

### Day 2 afternoon: MoE is its own genre of bug

The single most expensive class of error.

> [`5cffcf6`](https://github.com/artvandelay/should-i-self-host-llm/commit/5cffcf6) fix MoE fine-tuning cost — compute uses active params, VRAM uses total

The earlier code used `active_params_b` for everything. For Mixtral 8x7B (47B total, 13B active), this *undercounted VRAM by 3.6×*. The fix: every formula in the engine now explicitly says which one it uses.

> [`a857f04`](https://github.com/artvandelay/should-i-self-host-llm/commit/a857f04) fix multi-GPU + MoE catalog data; never default MoE active to total

The catalog had been silently filling in `active = total` for models where active wasn't listed. The fix forces `null` and surfaces it as a UI warning.

> [`cd30550`](https://github.com/artvandelay/should-i-self-host-llm/commit/cd30550) replace Mixtral hand-edits with a calibrated NxYB regex

The hand-edited Mixtral overrides were getting overwritten by nightly price refreshes (and were also a maintenance burden). A regex that parses model names like "Mixtral-8x7B" / "Qwen3-235B-A22B" / "DeepSeek-V3-A37B" replaced the special cases.

> [`50e0899`](https://github.com/artvandelay/should-i-self-host-llm/commit/50e0899) add inference regression tests for MoE, multi-GPU, billing math

After three MoE bugs, regression tests. Sequence matters: tests followed the bugs, not the other way around. That's normal for a research-grade calculator; the cost is that the tests now lock in the *right* answers, not just *some* answer.

### Day 2 evening: the billing fix that changed verdicts

> [`52326ee`](https://github.com/artvandelay/should-i-self-host-llm/commit/52326ee) fix self-host billing: input tokens, replica double-count, per-hour scaling

Three bugs in one commit:

1. Input tokens were being billed at the output rate (the model produces them; you don't pay for them at output cost on most APIs).
2. Replicas were being counted twice — once in throughput, once in the `$/hr` line.
3. Per-hour scaling was applying a non-linearity meant for cluster overhead at the wrong layer.

Together these meant some recommendations had been flipping the wrong way. The fix moved several scenarios from `selfhost_wins` → `api_wins` and vice versa.

> [`5e92e4f`](https://github.com/artvandelay/should-i-self-host-llm/commit/5e92e4f) pick GPU by total weekly cost, not lowest $/hr

A second-order effect of the billing fix. With the right throughput model, an L4 at $0.40/hr that needs 4 replicas costs more than an L40S at $0.86/hr that needs 1. Selecting on `$/hr` would have kept recommending the L4.

### Day 2 night: continuous batching

The most subtle bug in the project.

> [`120ad66`](https://github.com/artvandelay/should-i-self-host-llm/commit/120ad66) fix wildly inflated replicas at high QPS — model continuous batching

The naive throughput model: `tokens_per_sec_per_replica × replicas ≥ qps × tokens_per_query`. At 1k QPS, this estimator demanded dozens of replicas. In reality, **vLLM (and equivalent) batches concurrent requests through the same GPU** — throughput per GPU is roughly the same at 1 user and at 100 users until you hit memory or batch-size limits. The fix: replicas scale sub-linearly with QPS until a saturation threshold.

This commit alone moved high-QPS scenarios from "infeasibly expensive to self-host" to "obviously self-host wins".

### Day 3 (May 26 morning): the FT rearchitect

The FT panel had accumulated enough patches that the math was hard to verify. A focused refactor:

> [`1d5c6c4`](https://github.com/artvandelay/should-i-self-host-llm/commit/1d5c6c4) rebuild FT cost engine into clean src/ft/ module

Same math, isolated. Tests + assumptions doc move alongside the code. Then a tighter pass:

- [`23ca3a4`](https://github.com/artvandelay/should-i-self-host-llm/commit/23ca3a4) — cluster overhead got a tolerance band (don't switch topology at exactly `vram_gb`) + a utilization derate (real fine-tunes hit 30–40% MFU, not the datasheet's peak).
- [`6a3c6cd`](https://github.com/artvandelay/should-i-self-host-llm/commit/6a3c6cd) — added a `FtWarning` channel + a floor on cluster overhead. The engine now surfaces *why* a number might be off, instead of pretending precision it doesn't have.
- [`3c632e1`](https://github.com/artvandelay/should-i-self-host-llm/commit/3c632e1) — UI shows the warnings banner.

> [`174d8fc`](https://github.com/artvandelay/should-i-self-host-llm/commit/174d8fc) refresh-prices: auto-discover GPU SKUs from vendor markdown

Earlier, the GPU catalog was hand-maintained. Hand-maintained data drifts. The refresh script now parses Runpod / Lambda / Modal pricing pages and surfaces new SKUs automatically. The same pattern showed up in the skill sibling — fetch live, don't memorize.

### Day 3 afternoon: the skill sibling

> [`39bf3a8`](https://github.com/artvandelay/should-i-self-host-llm/commit/39bf3a8) Link to api-vs-selfhost-skill (agent-native sibling)

By this point the math was solid enough to bet on. The natural next move: most users with this question are already inside an agent (Claude Code, Cursor, Codex). Make the math callable from there instead of forcing a context switch to a website.

That work happened in [`api-vs-selfhost-skill`](https://github.com/artvandelay/api-vs-selfhost-skill) — separate repo, separate story, [its own NOTES.md](https://github.com/artvandelay/api-vs-selfhost-skill/blob/main/NOTES.md).

## What I'd warn anyone doing similar work about

- **MoE specs are wrong in half the catalogs you'll find.** Always check active vs total against the model card or paper, not a wrapper API.
- **GPU pricing pages change layout monthly.** Auto-refresh is brittle but hand-curation is worse. Bias toward auto-refresh with a "last reviewed" timestamp and a `FtWarning` for staleness.
- **LMArena Elo is the only widely-trusted quality signal that's machine-readable. Use their dataset, not a scraper.**
- **Replicas-from-QPS is not a divide. It's a curve.** If you assume linear, you'll over-estimate self-host cost at high QPS and recommend API wrong.
- **PEFT (LoRA/QLoRA) saves memory and optimizer state. Not compute.** Most calculators get this wrong.
- **`$/hr` is not a comparable unit across GPUs.** Use `$/TFLOP-hr` or `$/weekly-job` to compare.
- **An assumption you can't cite is a bug waiting.** [`FT_ASSUMPTIONS.md`](FT_ASSUMPTIONS.md) exists because every number in the engine needed to be traceable to a paper, a vendor doc, or a measured calibration.

## Acknowledgements

Built collaboratively with Claude (Anthropic), running in Cursor — ~13 sessions on May 25–26, 2026. Jigar Gosar drove product decisions, the pushbacks, and the taste calls; Claude wrote most of the code, found most of the bugs, and patched them. The git log is the literal record.

The agent-native sibling skill is at [api-vs-selfhost-skill](https://github.com/artvandelay/api-vs-selfhost-skill), with [its own NOTES.md](https://github.com/artvandelay/api-vs-selfhost-skill/blob/main/NOTES.md).
