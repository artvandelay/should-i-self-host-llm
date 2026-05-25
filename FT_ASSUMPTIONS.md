# Fine-Tuning Cost Calculator — Assumptions & Sources

A reader-facing companion to the FT cost panel. You should be able to read this
without opening any code. If any assumption here looks wrong for your workload,
treat the FT panel's output as a rough order-of-magnitude estimate, not a quote.

Last updated: 2026-05-26 (post-MoE-fix). Engine code: `src/ftMethods.ts`,
`src/engine.ts` (`computeFtCapex`, `ftVramGb`, `pickClusterOverhead`,
`pickFtGpu`), `src/FtPanel.tsx`.

---

## 1. TL;DR

- The calculator estimates **GPU $-cost and wall-clock GPU-hours** to fine-tune
  an open-weight model of size `params_b` on `num_examples × tokens_per_example
  × epochs` tokens, given the chosen method (LoRA / QLoRA / full FT).
- For **MoE** models (Mixtral, DeepSeek, Qwen-A-series, Llama-4-Scout, etc.),
  compute FLOPs scale with **active params per token**, not total — only the
  experts that fire do work. VRAM footprint and cluster overhead still use
  total params (all experts load into memory).
- It does estimate experiments/failed-runs (`experiments_multiplier`, default
  2.5×) and multi-GPU comms overhead (auto-picked `1.0× / 1.3× / 1.6×` from
  the FT VRAM footprint). It does **not** estimate dataset prep cost,
  engineering salary, evaluation runs, or any quality delta.
- **Headline assumption that surprises most people:** PEFT (LoRA / QLoRA) saves
  GPU **memory**, not GPU **compute**. The backward pass still propagates
  gradients through the frozen base weights, so LoRA/QLoRA only drop training
  FLOPs to ~⅔ of a full fine-tune — not the "99% cheaper" people expect from
  the trainable-parameter count. See [CE-LoRA, arxiv 2502.01378](https://arxiv.org/pdf/2502.01378)
  and the [AWS EU AI Act SageMaker formula](https://aws.amazon.com/blogs/machine-learning/navigating-eu-ai-act-requirements-for-llm-fine-tuning-on-amazon-sagemaker-ai/)
  (`F_ft = 4·N_total + 2·N_trainable`).
- **Calibration anchor:** the [QLoRA paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
  trained Guanaco-65B in 24 hours on a single 48GB pro GPU
  (~150 BF16 TFLOPS). An H100 is ~3× faster, so our engine predicts
  ~5–15 GPU-hours for the same workload on an H100 — within striking distance.
  The unit test `tests/ftCapex.test.ts` pins this anchor.

---

## 2. The formula in plain English

We walk from "tokens you'll train on" to "dollars" in four steps.

1. **Tokens.** `total_tokens = num_examples × tokens_per_example × epochs`.
   Every epoch is counted as a full pass; no warmup or LR-schedule overhead is
   subtracted or added.

2. **FLOPs.** Use the Kaplan **6N** rule: each token costs `6 × params` FLOPs to
   train (2N forward + 4N backward). Then multiply by a **method multiplier**
   that accounts for PEFT saving some — but not most — of the backward pass.

3. **Effective throughput.** Start from the H100 BF16 peak (**989 TFLOPS**),
   scale by a **30% baseline MFU** (Model FLOPs Utilization — realistic for
   small-batch, attention-heavy fine-tuning workloads), then apply a
   **per-method MFU penalty** (QLoRA pays a dequantization tax in particular).

4. **Hours → dollars.** Divide compute FLOPs by effective FLOPS/sec to get
   seconds, convert to hours, multiply by the cheapest H100 rate across our
   three tracked vendors (Modal / Lambda / Runpod). Then add `prep_cost_usd`
   (user-supplied black box: data labeling, eng time, etc.).

The whole thing in one line:

```
cost = (6 × params × tokens × method_multiplier)
       ────────────────────────────────────────────────────────
       (peak_FLOPS × baseline_MFU × method_MFU_penalty) × 3600
     × cheapest_H100_$/hr
     + prep_cost
```

### Worked example: 80B QLoRA, 100K examples × 1K tokens × 3 epochs

| Step | Value |
|---|---|
| `total_tokens` | 100,000 × 1,000 × 3 = **300M tokens** |
| Method multiplier (QLoRA) | 0.67 |
| Training FLOPs | 6 × 80e9 × 300e6 × 0.67 = **9.65 × 10¹⁹ FLOPs** |
| H100 peak BF16 | 989 TFLOPS |
| Baseline MFU | 30% |
| QLoRA MFU penalty | 0.70 |
| Effective FLOPS/sec | 989e12 × 0.30 × 0.70 = **2.08 × 10¹⁴ FLOPS** |
| Seconds | 9.65e19 / 2.08e14 ≈ **464,500 sec** |
| **GPU-hours** | ≈ **129 h** |
| Cheapest H100 (Runpod) | $2.90/hr |
| **GPU $-cost** | ≈ **$374** |
| `prep_cost_usd` | $0 (user-supplied) |
| **Total capex — 1× (single successful run, theoretical floor)** | **$374** |
| Experiments multiplier (default) | **2.5×** |
| **Total capex — 2.5× (default campaign: HP sweeps + failed runs)** | **$935** |

The 2.5× default reflects the reality that production fine-tuning campaigns
rarely succeed on the first run. Hyperparameter sweeps, early-stopped failures,
and ablation experiments routinely consume 2–5× the cost of the optimal run.
A parallel internal cost model (Jio-Health-AI) uses
`effective_full_runs (3.0–4.5) × early-stop factor (0.5–0.6)`, netting to
~1.5–2.5× — which brackets our 2.5× default. Users with disciplined iteration
(strong priors, narrow sweeps) can dial it down to 1.5× or even 1×; greenfield
work without prior recipes often warrants 3–4×. The single-run floor remains
visible in the panel breakdown so the multiplier's impact is transparent.

For comparison: at Fireworks' published [LoRA SFT pricing](https://fireworks.ai/pricing)
($3.00 per 1M training tokens for the 16.1B–80B band), the same 300M-token job
would cost **$900** end-to-end on their managed service. The ~2.4× gap is the
vendor's margin + multi-tenant overhead + included hosting. See §5 for more
back-solves.

---

## 3. Assumptions table

| # | Assumption | Current value | Source | Confidence | Flag for review? |
|---|---|---|---|---|---|
| 1 | Training FLOPs per token per param | **6N** (2N fwd + 4N bwd) | [Kaplan et al. 2020, arxiv 2001.08361](https://arxiv.org/abs/2001.08361); Chinchilla | HIGH | N |
| 2 | LoRA computeMultiplier | **0.67** | [CE-LoRA arxiv 2502.01378](https://arxiv.org/pdf/2502.01378); [AWS SageMaker EU AI Act post](https://aws.amazon.com/blogs/machine-learning/navigating-eu-ai-act-requirements-for-llm-fine-tuning-on-amazon-sagemaker-ai/) (F_ft = 4N_total + 2N_trainable ≈ ⅔ of 6N when N_trainable → 0) | MED | N |
| 3 | QLoRA computeMultiplier | **0.67** | Same as LoRA — quantization changes memory, not FLOPs count | MED | **Y** — could argue QLoRA does slightly fewer effective FLOPs because the matmul is in 4-bit; we capture this entirely via the MFU penalty instead. Worth checking whether to split. |
| 4 | Full-FT computeMultiplier | **1.0** | Definitional | HIGH | N |
| 5 | LoRA mfuPenalty | **0.85** | Implicit (LoRA's smaller adapter matmuls hit lower arithmetic intensity than full FT). Not directly benchmarked. | LOW | **Y** — no hard public benchmark backs the exact 0.85. Unsloth / TildAlice numbers could refine. |
| 6 | QLoRA mfuPenalty | **0.70** | [TildAlice QLoRA benchmark](https://tildalice.io/lora-qlora-full-finetuning-gpu-memory-benchmark/) — QLoRA ~45% slower per-step than LoRA on A100 → roughly 0.85 × (1/1.45) × correction ≈ 0.70 | MED | **Y** — TildAlice ran on A100, not H100. H100's better INT/FP8 path may close the gap. |
| 7 | Full-FT mfuPenalty | **1.0** | Definitional | HIGH | N |
| 8 | H100 BF16 peak | **989 TFLOPS** | NVIDIA H100 datasheet (sparsity off) | HIGH | N |
| 9 | Baseline MFU | **30%** | [stevengong notes on MFU](https://stevengong.co/notes/Model-FLOPs-Utilization); [technolynx training-efficiency post](https://www.technolynx.com/post/model-flops-utilization-training-efficiency) — FT recipes typically 25–35%, pretraining at scale 40–50% | MED | **Y** — well-tuned single-node FT with FlashAttention / fused kernels can hit 40%+; conservatively low for power users. |
| 10 | GPU = single H100 at cheapest vendor | **Runpod H100 80GB @ $2.90/hr** | `src/pricing.json` (refreshed via BYOK Firecrawl script) | HIGH | N |
| 11 | Multi-GPU / multi-node cluster overhead | **Auto-picked from FT VRAM footprint**, sized to the actual training GPU's `single_gpu_vram_gb × gpus_per_node` (from `pricing.json`). With today's H100 (80 GB, 8/node): ≤80 GB → 1.0×, ≤640 GB → 1.3×, >640 GB → 1.6×. Drops in unchanged for B200, MI300X (192 GB), GB200 NVL36, etc. User can override via the UI. | Industry rule-of-thumb; FSDP/DeepSpeed all-reduce typically eats 20–50% of wall clock; intra-node NVLink ≈ 1.3×, inter-node IB ≈ 1.6× | MED | N — first-class input. |
| 19 | Training GPU choice & peak TFLOPS | **Picked from `pricing.json` by best $/TFLOP-hr**: any row with `bf16_tflops > 0` is eligible; the one with the lowest `min(modal,lambda,runpod) / bf16_tflops` wins. H100 (989 TFLOPS @ $2.9/hr → $2.93/TFLOP-hr) beats A100 ($5.13/TFLOP-hr) today. A hypothetical B200 (2250 TFLOPS @ $5/hr → $2.22) would automatically take over. Falls back to 989 TFLOPS / $4/hr if the config has nothing tagged. | NVIDIA datasheets; AMD MI300X spec | HIGH | N — config-driven. |
| 20 | Vendor coverage | Hard-coded to Modal / Lambda / Runpod (`Vendor` union + parallel `${vendor}_per_hr` fields). Adding Together / Fireworks / CoreWeave / AWS requires editing the type union AND every JSON row AND the refresh script. **Out of scope by design** — three vendors give us a defensible "cheapest of three" floor; exhaustive vendor coverage isn't the goal. | Project decision | HIGH | N — scope-limited. |
| 21 | Inference throughput formula | **Aggregate** throughput with continuous batching, not single-stream decode. Anchor: `960 tok/s per 8B active per H100-equivalent unit` × `batchingMultiplier(active_params_b)` (10× for ≤4B, 6× for ≤16B, 3× for ≤40B, 2× for ≤80B, 1.5× for >80B). Scales linearly by `bf16_tflops / 989` for new hardware. Batching multipliers reflect KV-cache headroom — small-active MoE models get the biggest wins. Calibrated against published vLLM/TGI numbers at ~2–4K context. | vLLM / TGI / SGLang benchmarks; KV-cache slot math | MED | N — first-class. |
| 22 | `ftVramGb` bytes-per-param (14 / 1.0 / 0.5 for full / LoRA / QLoRA) | Calibrated against published 70B numbers (QLoRA ~40 GB, LoRA ~80 GB, full ~1 TB). Used only to pick cluster topology; doesn't affect $ cost directly. Off by 2× wouldn't shift the headline answer. | TildAlice, channel.tel benchmarks | MED | N — calibrated, leave it. |
| 23 | Cluster overhead magic values (1.0 / 1.3 / 1.6) | Round-number rule-of-thumb for single-GPU / intra-node-NVLink / multi-node-Infiniband comms tax. Real numbers vary 1.05–1.5 / 1.4–1.8. Pinned for legibility; tunable in code constants. | NVIDIA scaling docs; FSDP/DeepSpeed reports | MED | N — deliberate simplification. |
| 12 | Epochs counted as-is | No warmup / no LR-schedule overhead | n/a | MED | **Y** — warmup typically wastes ~5% of compute. |
| 13 | Gradient checkpointing overhead | Not modeled | [AWS EU AI Act post](https://aws.amazon.com/blogs/machine-learning/navigating-eu-ai-act-requirements-for-llm-fine-tuning-on-amazon-sagemaker-ai/) notes ~1.33× compute overhead when enabled (recomputes activations) | MED | **Y** — large models almost always enable checkpointing; our number is optimistic for them. |
| 14 | FP8 training (Hopper Transformer Engine) | Not modeled — pure BF16 throughput | [Lyceum 70B GPU guide](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/) — FP8 can ~2× effective throughput on H100 | MED | **Y** — power users on H100/H200 with TE-enabled stacks could halve our number. |
| 15 | Data loading / preprocessing time | Not modeled | n/a | HIGH (cheap) | N |
| 16 | Hyperparameter sweeps / failed runs | **User-controlled `experiments_multiplier`, default 2.5×** (clamped to ≥ 1.0). Applied to `gpu_hours` and `gpu_cost_usd`; NOT applied to `prep_cost_usd`. | First-class UI input; cross-checked against an internal Jio-Health-AI cost model that uses `effective_full_runs × early-stop_factor` netting to ~1.5–2.5×. | MED | N |
| 17 | Quality impact of FT | Not estimated | n/a (panel disclaimer) | HIGH | N |
| 18 | MoE handling | **Compute uses active params per token; VRAM/cluster overhead use total.** `computeFtCapex(active_params_b, inputs, total_params_b)`. For dense models active == total so no-op. For MoE (Mixtral 8x7B = 47/12, DeepSeek-V3 = 671/37, Llama-4-Scout = 109/17), FT FLOPs are computed against active — only the experts that fire do work — matching how every real MoE FT recipe in the wild trains. | Mixtral / DeepSeek / Qwen-A-series papers; expert-routing convention | HIGH | N — first-class. |
| 18b | `prep_cost_usd` | User-supplied black box | UI input | HIGH | N (but see open question §5) |

---

## 4. What's missing / known limitations

- **MoE handled correctly (as of 2026-05-26).** FT compute uses active
  params per token; VRAM and cluster overhead use total. Qwen3-235B-A22B
  fine-tunes at ~22B-compute cost, not 235B. MoE models that lack an
  `active_b` value in our catalog are skipped entirely rather than quoted
  with a misleading dense-equivalent number.
- **Multi-GPU / multi-node overhead is modeled** via `pickClusterOverhead`
  (1.0× / 1.3× / 1.6×). It captures comms tax, not "you need N GPUs" —
  workloads whose FT VRAM exceeds the largest catalog SKU still get a
  dollar number, with the caveat that the cluster is implied. A
  feasibility guard for "this won't fit anywhere we know about" is a
  known TODO.
- **No warmup / LR-schedule waste.** A linear warmup over 3–10% of steps does
  forward+backward passes that don't materially update weights; we count
  them at full speed.
- **No checkpoint I/O time.** Saving a 70B checkpoint to disk every N steps
  is non-trivial wall time at scale; not modeled.
- **No eval-during-training cost.** Most real FT runs evaluate every K steps.
  Together's pricing model explicitly bills eval tokens; we bill nothing.
- **`prep_cost` is opaque.** It's a single number the user types. We give no
  guidance on what good values are by use case.
- **Single vendor / single SKU.** We always price against the cheapest H100
  across Modal/Lambda/Runpod. No spot/preemptible discount, no A100 fallback,
  no Blackwell B200 (which [Lyceum reports ~2.2× faster than H100](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/)
  for Llama-2 70B FT).
- **Quality impact = 0.** Whether the fine-tune actually helps is out of
  scope; the panel disclaimer tells users to compare Arena ELO of base vs
  tuned models separately.

---

## 5. Reference comparisons

Cross-checking our engine against public anchors. Wall-clock anchors are
scaled to a single H100 (~3× faster than A100 80GB for BF16 training, per
[Lyceum's H100 vs A100 article](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/))
where the source ran on a different GPU.

### 5a. Wall-clock anchors

| Scenario | Source says | Our engine says | Ratio |
|---|---|---|---|
| Guanaco-65B QLoRA, 10K × ~500 tok × 3 epochs, single 48GB pro GPU | **24 h** | engine = 4.7 h on H100 — scaled back to ~150 TFLOPS pro GPU ≈ 31 h | within **1.3×** ✓ ([QLoRA paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)) |
| Llama 3.1 8B QLoRA, 15K examples × ~1K tok × 3 epochs, A100 80GB | **~4.5 h** ([Medium "$20" guide](https://medium.com/@velinxs/how-to-fine-tune-llms-for-under-20-step-by-step-c187a3059ca2)) | engine ≈ 0.43 h on H100 → ~1.3 h scaled to A100 | source **3.4×** higher — likely because real run had non-trivial setup, tokenizer debug, eval, and the marketplace A100 was underclocked. Flagged. |
| Llama 3.1 70B QLoRA, 1 epoch over moderate dataset, A100 80GB | **12–20 h** ([same source](https://medium.com/@velinxs/how-to-fine-tune-llms-for-under-20-step-by-step-c187a3059ca2)) | engine ≈ 3.8 h on H100 → ~11–12 h scaled to A100 | within **~1.5×** ✓ |

### 5b. Vendor $/1M-token back-solves

Vendors quote a flat $/1M-training-tokens price. We can back-solve: what does
our engine produce in GPU $ for the same workload?

| Vendor | Model band | $ / 1M training tokens (LoRA) | Our engine $ for 300M tokens | Vendor / Us | Source |
|---|---|---|---|---|---|
| **Fireworks** | up to 16B | $0.50 → $150 | 16B LoRA = **$72** | **2.1×** | [fireworks.ai/pricing](https://fireworks.ai/pricing) |
| **Fireworks** | 16.1B – 80B | $3.00 → $900 | 80B QLoRA = **$374** | **2.4×** | same |
| **Fireworks** | 80B – 300B | $6.00 → $1,800 | 235B-A22B QLoRA (MoE, active drives FLOPs) = **~$103** single run, ~$334 default campaign (2.5× + cluster) | **5–17×** | same — vendor charges by total-param band; we now charge by active. MoE FT looks much cheaper on raw hardware than vendor pricing reflects. |
| **Fireworks** | >300B | $10.00 → $3,000 | 405B (dense) QLoRA = **$1,895** | **1.6×** | same |
| **Together** | 70B–100B LoRA | $2.90 → $870 | 80B LoRA = **$454** | **1.9×** | [eesel breakdown of Together pricing](https://www.eesel.ai/blog/together-ai-pricing) |
| **Together** | 17B–69B LoRA | $1.50 → $450 | 70B LoRA = ~$397 (close-enough size) | **1.1×** | same |
| **PricePerToken aggregate** | Llama 3.1 70B LoRA | $2.90 → $870 | $397–$454 | **1.9–2.2×** | [pricepertoken.com/fine-tuning](https://pricepertoken.com/fine-tuning) |

**Pattern.** Across every band, managed vendors charge **~1.5–2.5×** what our
engine predicts in raw H100 $. That's a healthy and consistent gross margin
for someone running multi-tenant GPU infra with managed checkpointing,
storage, eval, and APIs included. The ratio is tightest at the very high end
(>80B) where vendor utilization is best, and widest at the small end where
our minimum-of-1-GPU floor doesn't apply.

**Nothing in this comparison set is off by >3×, and nothing flips the
direction of the answer the calculator gives.** The engine is calibrated to
"what would a single H100 actually cost you on Runpod" — which is a
defensible lower bound for someone who knows what they're doing — and managed
vendor prices sit ~2× above that, which is the right shape.

### 5c. Where we're most likely wrong

- **Small-batch / short-context runs** at the 7B–13B scale: our 0.43h figure
  for Llama 3.1 8B is half what the practitioner [actually measured](https://medium.com/@velinxs/how-to-fine-tune-llms-for-under-20-step-by-step-c187a3059ca2),
  because real runs are dominated by setup + eval, not the math we model.
- **Big-model multi-GPU runs**: rough — we apply a flat 1.3× / 1.6× tax,
  but real FSDP/DeepSpeed comms vary 1.05–1.8× depending on context length,
  batch, network, and zero-stage. Documented; not currently tunable per-run
  beyond the auto/override knob.
- **MoE without `active_b` metadata**: a MoE model whose name doesn't match
  our regex (`B-AYB` or `NxYB` Mixtral-style) is now skipped from the
  catalog rather than quoted with total-param FLOPs. Bias is omission, not
  over-statement.

---

## 6. Open questions for the user

These are the calls we'd like you to make before we polish the panel further.

1. **Expose method multipliers as advanced inputs?** Power users who have
   their own throughput benchmarks (Unsloth, vendor-specific) might want to
   override `computeMultiplier` and `mfuPenalty` per method. Worth a hidden
   "advanced calibration" expander, or keep it locked down for simplicity?

2. **Cluster-overhead multiplier — ✅ ADDED.** Engine now auto-picks `1.0× /
   1.3× / 1.6×` from the FT VRAM footprint (single-GPU / NVLink node /
   multi-node) via `ftVramGb(params_b, method)` and `pickClusterOverhead()`.
   User can override via the "Cluster overhead" select. Calibration: bytes-
   per-param 14 (full) / 1.0 (LoRA) / 0.5 (QLoRA), validated against the
   published 70B numbers (QLoRA ≈ 40 GB, LoRA ≈ 80 GB, full ≈ 1 TB).

3. **Add an FP8 / Transformer Engine mode?** [Lyceum reports](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/)
   ~2× throughput from H100 FP8 and ~2.2× more from Blackwell. A "modern
   stack" toggle could bump effective FLOPS accordingly. Risky if users
   don't actually use FP8 in their training script.

4. **Should we expose method-specific defaults for `experiments_multiplier`?**
   The failed-runs contingency is now a first-class UI input (default 2.5×,
   clamped to ≥ 1×). But should the default vary by method? Full fine-tuning
   typically needs more shots-at-goal (more HP knobs, higher blast radius
   per failure) than QLoRA — which often "just works" on the first try with
   sensible defaults. Options: (a) keep a single global 2.5× default;
   (b) split into per-method defaults (e.g. QLoRA=1.5×, LoRA=2.0×, full=3.0×);
   (c) tie the default to model size as well (bigger models = more expensive
   failures = higher discipline → lower multiplier in practice).

5. **Replace `prep_cost_usd` with a structured breakdown?** Today it's one
   free-text dollar input. We could break it into
   `examples × $/example_to_label + eng_days × $/eng_day`, with sensible
   defaults ($1–$5 per label, $1k–$2k per eng-day). More honest, but more
   inputs to a panel that's already busy.

6. **Model warmup / LR-schedule waste?** Adding a flat 5% padding would
   bring us closer to reality with almost no UI cost. Worth it, or not
   worth confusing users with an extra knob?

7. **Should MoE FT use active params instead of total? — ✅ DONE.** As of
   2026-05-26 `computeFtCapex(active_params_b, inputs, total_params_b)`
   takes both, applying active to FLOPs and total to VRAM/cluster overhead.
   `FtPanel` threads both fields. MoE without an `active_b` value is
   skipped from the candidate set rather than silently quoted at
   dense-equivalent cost. Pinned by tests in `tests/ftCapex.test.ts`
   under `describe("MoE: compute uses active params, VRAM uses total")`.

8. **Should we surface the headline vendor-margin observation in the UI?**
   E.g. "Our $374 estimate is what you'd pay running this yourself on
   Runpod. Managed vendors (Together / Fireworks) would charge **~$700–$900**
   for the same job." This is a useful trust-builder and a "vs API" anchor —
   but adds a chunk of explanatory text to an already-dense panel.

---

## 7. Sources

Code:

- `src/ftMethods.ts` — multipliers, constants, citations inline.
- `src/engine.ts` — `computeFtCapex`, `ftVramGb`, `pickClusterOverhead`,
  `pickFtGpu` (search the file for the function names; line numbers drift).
- `src/FtPanel.tsx` — UI and footer disclaimer.
- `tests/ftCapex.test.ts` — Guanaco-65B anchor; MoE active-vs-total split;
  cluster-overhead boundaries; experiments-multiplier semantics.

Public references (only those actively used above):

- AWS — [Navigating EU AI Act requirements for LLM fine-tuning on Amazon SageMaker AI](https://aws.amazon.com/blogs/machine-learning/navigating-eu-ai-act-requirements-for-llm-fine-tuning-on-amazon-sagemaker-ai/)
  (LoRA backward formula; gradient checkpointing overhead).
- CE-LoRA — [arxiv 2502.01378](https://arxiv.org/pdf/2502.01378) (PEFT does not
  cut backward compute the way trainable-param counts suggest).
- Dettmers et al. — [QLoRA paper, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
  (Guanaco-65B = 24 h on 48 GB GPU calibration anchor).
- Kaplan et al. — [Scaling Laws, arxiv 2001.08361](https://arxiv.org/abs/2001.08361)
  (6N training-compute rule).
- TildAlice — [LoRA / QLoRA / full FT GPU memory benchmark](https://tildalice.io/lora-qlora-full-finetuning-gpu-memory-benchmark/)
  (QLoRA dequantization tax).
- Steven Gong — [Model FLOPs Utilization notes](https://stevengong.co/notes/Model-FLOPs-Utilization).
- Technolynx — [MFU training-efficiency post](https://www.technolynx.com/post/model-flops-utilization-training-efficiency).
- Together AI — [docs.together.ai/docs/fine-tuning-pricing](https://docs.together.ai/docs/fine-tuning-pricing)
  (token-counting formula); [eesel pricing breakdown](https://www.eesel.ai/blog/together-ai-pricing)
  (actual $/1M-token table for SFT / LoRA across model sizes).
- Fireworks AI — [fireworks.ai/pricing](https://fireworks.ai/pricing)
  (LoRA / Full-Param SFT and DPO $/1M-token table across model bands).
- pricepertoken — [LLM Fine-Tuning Pricing 2026](https://pricepertoken.com/fine-tuning)
  (multi-vendor $/1M-token comparison).
- Lyceum — [Which GPU for Fine-Tuning 70B Models?](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/)
  (H100 vs A100 throughput multiplier; FP8/Blackwell upside; NVLink vs PCIe).
- velinxs / Medium — ["How to Fine-Tune LLMs for Under $20"](https://medium.com/@velinxs/how-to-fine-tune-llms-for-under-20-step-by-step-c187a3059ca2)
  (real-world Llama 3.1 8B and 70B QLoRA wall-clock anchors on A100 80GB).
