# Input contract and defaults

This file documents every field accepted by `skill/calc.py`, with units, sensible defaults, extraction hints for the LLM, and the confidence tag to attach when a value is defaulted rather than user-provided. The skill should never silently default a high-leverage field — ask the user instead, then record the answer here in the report's assumptions table.

## Legend

- **required?** — `yes` means `calc.py` raises `missing field <name>` if absent. `no` means the script applies a built-in default. "only if ..." means conditionally required.
- **sensible default** — what the LLM should propose if the user gave no signal. `(none — ask user)` means do not default; ask one clarifying question instead.
- **confidence-when-defaulted** — the tag the LLM must attach in the report's assumptions table whenever it filled this field without explicit user input.
- All `gpu.*` fields are nested under a single `gpu` object in the JSON payload; see the Frozen Interface Contract in the parent plan or the example payloads in `SKILL.md` Phase 4.
- Units are SI / decimal; no commas or unit suffixes inside the JSON values (e.g. `1000000`, not `"1M"`).

## Inference inputs

Fields consumed by `python3 skill/calc.py inference`. One row per top-level JSON field, plus one row per `gpu.*` sub-field.

| field | type | unit | required? | sensible default | extraction hints | confidence-when-defaulted |
| --- | --- | --- | --- | --- | --- | --- |
| params_b | float | billions of params | yes | (none — ask user) | Look for model name in code, configs, system prompts (e.g. `llama-3.1-70b`, `qwen2.5-32b`). Map name → param count via `knownModels.json` or the model card on Hugging Face. | low if guessed |
| active_params_b | float | billions of active params | yes | equal to `params_b` for dense models | For MoE models (Mixtral, Qwen3-MoE, DeepSeek-V3) use the active-experts count from the model card; for dense models reuse `params_b`. | med |
| quant | string | one of `fp16`, `int8`, `int4` | yes | `int4` for >30B, `fp16` for ≤7B | Look for `bnb_4bit`, `gguf` quant suffix (Q4_K_M → int4, Q8_0 → int8), `--load-in-4bit` flags, vLLM `--quantization` arg. | med |
| queries_per_week | int | requests/week | yes | (none — ask user) | Look in: access logs, billing pages, "<n> requests/day" mentions, Cloudflare/Vercel dashboards screenshots. Multiply daily by 7. | low if guessed |
| avg_tokens_per_query | int | tokens (input+output) | no | 800 | Sum a typical prompt + completion length. Chat apps ≈ 500–1500; RAG ≈ 2000–6000; agents ≈ 4000–20000. Drives default `context_len` and `avg_output_tokens`. | med |
| context_len | int | tokens | no | = `avg_tokens_per_query` | Max live context per request (drives KV cache size). Long-context apps (RAG, agents, 100k+ docs) make KV cache dominate VRAM. | med |
| avg_output_tokens | int | tokens | no | `0.6 × avg_tokens_per_query` | Generated (decode) tokens per query — this is what consumes throughput. Input/prefill tokens are cheap by comparison. | med |
| n_layers | int | transformer layers | no | heuristic from `params_b` | From the model config (`num_hidden_layers`). KV cache scales linearly with this. 32B≈64, 70B≈80. | low if guessed |
| d_model | int | hidden dim | no | heuristic from `params_b` | From the model config (`hidden_size`). KV cache scales linearly with this. 32B≈8192, 70B≈8192. | low if guessed |
| gqa_ratio | float | query heads per KV head | no | 8 | Grouped-Query-Attention ratio = `num_attention_heads / num_key_value_heads`. 1 = full multi-head (no KV reduction); 8 is typical for 30B+. | med |
| kv_bytes_per_elem | float | bytes | no | 2 (fp16 KV) | KV cache dtype size. 2 = fp16/bf16 (standard). Set 1.0 for fp8/int8 KV cache. | high |
| concurrency | int | concurrent streams (batch) | no | auto (VRAM- and roofline-capped) | Force a fixed batch size. If omitted the engine picks `min(VRAM-max, compute_intensity/2)`. Setting it above VRAM capacity flips `fits` to false. | low |
| api_cost_per_query_usd | float | USD per query | yes | (none — ask user or compute from token prices) | If only $/1M tokens is known, compute `(input_price * input_tokens + output_price * output_tokens) / 1e6`. Vendor pricing pages or models.dev. | med |
| traffic_pattern | string | one of `uniform`, `business`, `bursty`, `always_warm`, `cold_per_query` | yes | `business` | `uniform` = 24/7 steady; `business` = 9–6 weekdays (≈50h/wk); `bursty` = spiky internal tools (~20h/wk); `always_warm` = pinned GPU; `cold_per_query` = serverless cold-starts dominate. | med |
| hot_hours_per_week | int | hours/week | only if `traffic_pattern == "cold_per_query"` | 50 | Override used when traffic shape doesn't match a preset. Estimate from request timestamps. | low |
| cold_start_penalty_sec | int | seconds | no | 30 | Time the GPU stays warm after the last request, on serverless platforms (Modal, Replicate). Vendor docs or measurement. | low |
| gpu.name | string | label | yes | `H100 SXM 80GB` | From `references/GPU_SPECS.md`. Pick the smallest GPU whose VRAM ≥ `vram_needed_gb`. | high |
| gpu.vram_gb | int | gigabytes | yes | from `references/GPU_SPECS.md` | Static per-SKU. Do not scrape. | high |
| gpu.usd_per_hr | float | USD/hr | yes | (none — fetch live) | WebFetch Runpod / Lambda / Modal / Together pricing pages. Cite URL + timestamp. | low if guessed |
| gpu.bf16_tflops | int | TFLOPS | yes | from `references/GPU_SPECS.md` | Datasheet peak, no sparsity. Static per-SKU. Used for roofline-optimal batch size. | high |
| gpu.mem_bandwidth_tbs | float | TB/s | yes | from `references/GPU_SPECS.md` | HBM/global-memory bandwidth. Binding constraint for decode throughput. Static per-SKU. | high |

## Fine-tune inputs

Fields consumed by `python3 skill/calc.py finetune`. Defaults reflect realistic SFT practice (LoRA / QLoRA on instruction data); override aggressively for full fine-tunes or pretraining-style runs.

| field | type | unit | required? | sensible default | extraction hints | confidence-when-defaulted |
| --- | --- | --- | --- | --- | --- | --- |
| active_params_b | float | billions of active params | yes | (none — ask user) | Same as inference. For MoE, only the active experts are trained per token. | low if guessed |
| total_params_b | float | billions of total params | yes | equal to `active_params_b` for dense | For dense models reuse `active_params_b`. For MoE use the full param count (Mixtral 8x7B → 47B). Drives VRAM, not FLOPs. | med |
| method | string | one of `full`, `lora`, `qlora` | yes | `qlora` | Look for `peft`, `LoraConfig`, `bitsandbytes`, `--load-in-4bit`, `unsloth`, `axolotl` configs. Default to `qlora` for >7B if the user didn't say. | med |
| num_examples | int | training rows | yes | (none — ask user) | Count rows in the dataset (`wc -l data.jsonl`), or read the dataset card on Hugging Face. | low if guessed |
| tokens_per_example | int | tokens | yes | 1000 | Average prompt + response length. Instruction tuning ≈ 500–2000; long-context SFT ≈ 4000–16000. | med |
| epochs | int | passes over data | yes | 3 | Standard SFT default. Most papers use 1–5. | med |
| experiments_multiplier | float | runs (incl. failures, sweeps) | yes | 2.5 | Real projects do hyperparam sweeps + reruns. 1.0 = single perfect run (rare); 2.5 = realistic; 5+ = research. | med |
| prep_cost_usd | float | USD | yes | 0 | Data labeling, synthetic data generation, eval set construction. Often the dominant line item in production FT projects. Ask the user. | low |
| gpu.name | string | label | yes | `H100 SXM 80GB` | Same as inference. | high |
| gpu.vram_gb | int | gigabytes | yes | from `references/GPU_SPECS.md` | Static per-SKU. | high |
| gpu.usd_per_hr | float | USD/hr | yes | (none — fetch live) | Same as inference. | low if guessed |
| gpu.bf16_tflops | int | TFLOPS | yes | from `references/GPU_SPECS.md` | Static per-SKU. | high |
| gpu.gpus_per_node | int | GPUs | no | 8 | Cloud-standard 8-GPU node (HGX H100, DGX). Override only if the user has unusual hardware. | high |
