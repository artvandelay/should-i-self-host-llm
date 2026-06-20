#!/usr/bin/env python3
"""Deterministic math for the PH-FT agent skill.

Two subcommands: `inference` and `finetune`. Each reads a JSON object from
stdin and writes a JSON object to stdout. Pure arithmetic only — no network,
no file I/O beyond stdin/stdout, stdlib only. The LLM is responsible for
fetching live pricing/quality data and passing it in as numbers.

Exit codes: 0 on success, 2 on bad input (with `{"error": "..."}` on stdout),
1 on internal error.
"""

import json
import math
import sys

FLOPS_PER_TOKEN_PER_PARAM = 6
BASELINE_MFU = 0.30

FT_METHODS = {
    "full":  {"compute_mult": 1.0,  "mfu_penalty": 1.0,  "bytes_per_param": 14},
    "lora":  {"compute_mult": 0.67, "mfu_penalty": 0.85, "bytes_per_param": 1.0},
    "qlora": {"compute_mult": 0.67, "mfu_penalty": 0.70, "bytes_per_param": 0.5},
}

BYTES_PER_PARAM_INFERENCE = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}

# Fixed activation / runtime buffer on top of weights + KV cache (GB).
VRAM_ACTIVATION_OVERHEAD_GB = 2.0
# KV cache element size in bytes. Inference KV cache is conventionally fp16
# (2 bytes) even when the weights are quantized. Override via `kv_bytes_per_elem`
# (e.g. 1.0 to reproduce fp8/int8 KV napkin estimates).
KV_BYTES_PER_ELEM_DEFAULT = 2.0
# Grouped-Query-Attention ratio: query heads per KV head. Modern 30B+ models
# typically use 8 (cuts KV cache ~8x vs full multi-head). Override per model.
GQA_RATIO_DEFAULT = 8.0

TRAFFIC_PATTERN_HOURS = {
    "uniform": 168,
    "business": 50,
    "bursty": 20,
    "always_warm": 168,
}


def _estimate_arch(params_b: float):
    """Heuristic (d_model, n_layers) from dense param count.

    Anchored to published configs: 8B -> (4096, 32), 32B -> (8192, 64)
    [matches the injuly.in napkin-math example], 70B -> (8192, 80),
    405B -> (16384, 126). Used only when the caller does not supply
    `d_model` / `n_layers` explicitly.
    """
    if params_b <= 2:
        return 2048, 24
    if params_b <= 9:
        return 4096, 32
    if params_b <= 15:
        return 5120, 40
    if params_b <= 40:
        return 8192, 64
    if params_b <= 80:
        return 8192, 80
    return 16384, 126


def _require(d, key):
    if key not in d:
        raise KeyError(key)
    return d[key]


def compute_inference(inp: dict) -> dict:
    params_b = _require(inp, "params_b")
    quant = _require(inp, "quant")
    queries_per_week = _require(inp, "queries_per_week")
    api_cost_per_query_usd = _require(inp, "api_cost_per_query_usd")
    traffic_pattern = _require(inp, "traffic_pattern")
    gpu = _require(inp, "gpu")
    gpu_vram_gb = _require(gpu, "vram_gb")
    gpu_usd_per_hr = _require(gpu, "usd_per_hr")
    gpu_mem_bandwidth_tbs = _require(gpu, "mem_bandwidth_tbs")

    if quant not in BYTES_PER_PARAM_INFERENCE:
        raise ValueError(f"unknown quant {quant}")

    # --- Workload shape (optional, with sensible defaults) ---
    avg_tokens_per_query = inp.get("avg_tokens_per_query", 800)
    context_len = inp.get("context_len", avg_tokens_per_query)
    avg_output_tokens = inp.get("avg_output_tokens", max(1, round(avg_tokens_per_query * 0.6)))
    if context_len <= 0:
        raise ValueError("context_len must be > 0")

    # --- Model architecture (optional, heuristic fallback) ---
    d_model_default, n_layers_default = _estimate_arch(params_b)
    d_model = inp.get("d_model", d_model_default)
    n_layers = inp.get("n_layers", n_layers_default)
    gqa_ratio = inp.get("gqa_ratio", GQA_RATIO_DEFAULT)
    kv_bytes_per_elem = inp.get("kv_bytes_per_elem", KV_BYTES_PER_ELEM_DEFAULT)
    if gqa_ratio <= 0:
        raise ValueError("gqa_ratio must be > 0")

    # --- VRAM: weights + KV cache (the part the old weights-only model missed) ---
    bytes_per_param = BYTES_PER_PARAM_INFERENCE[quant]
    weights_gb = params_b * bytes_per_param
    # KV cache bytes per generated token, per concurrent stream.
    # 2 = one K and one V tensor; divided by GQA ratio.
    kv_per_token_bytes = 2.0 * n_layers * d_model * kv_bytes_per_elem / gqa_ratio
    kv_per_stream_gb = kv_per_token_bytes * context_len / 1e9
    activation_gb = VRAM_ACTIVATION_OVERHEAD_GB

    # Minimum footprint = weights + activations + 1 concurrent stream's KV cache.
    min_vram_gb = weights_gb + activation_gb + kv_per_stream_gb
    fits = min_vram_gb <= gpu_vram_gb

    # --- Concurrency (batch size B) the GPU can actually run ---
    avail_for_kv_gb = gpu_vram_gb - weights_gb - activation_gb
    if kv_per_stream_gb > 0:
        max_concurrency_vram = int(avail_for_kv_gb // kv_per_stream_gb)
    else:
        max_concurrency_vram = 1_000_000
    max_concurrency_vram = max(0, max_concurrency_vram)

    # Compute-optimal batch from arithmetic intensity: the roofline balance
    # point is reached at B = compute_intensity / 2 (decode does ~2*B ops per
    # byte loaded; compute_intensity = peak_flops / mem_bandwidth).
    bf16_tflops = gpu.get("bf16_tflops")
    if bf16_tflops:
        compute_intensity = (bf16_tflops * 1e12) / (gpu_mem_bandwidth_tbs * 1e12)
        batch_opt = max(1, int(compute_intensity / 2.0))
    else:
        batch_opt = None

    concurrency_override = inp.get("concurrency")
    if concurrency_override is not None:
        concurrency = max(1, int(concurrency_override))
        if concurrency > max_concurrency_vram:
            fits = False
    elif fits:
        cap = max_concurrency_vram if batch_opt is None else min(max_concurrency_vram, batch_opt)
        concurrency = max(1, cap)
    else:
        concurrency = 0

    kv_cache_gb = kv_per_stream_gb * concurrency
    vram_needed_gb = weights_gb + activation_gb + kv_cache_gb
    vram_headroom_gb = gpu_vram_gb - vram_needed_gb

    # --- Throughput: decode is memory-bandwidth bound ---
    # Each forward pass moves weights once + every active stream's KV cache,
    # then emits `concurrency` tokens. Aggregate tok/s = B / time_per_forward.
    bandwidth_bytes = gpu_mem_bandwidth_tbs * 1e12
    if fits and concurrency > 0:
        mem_moved_bytes = weights_gb * 1e9 + kv_per_stream_gb * 1e9 * concurrency
        time_per_forward_sec = mem_moved_bytes / bandwidth_bytes
        per_gpu_tokens_per_sec = concurrency / time_per_forward_sec
        per_user_tokens_per_sec = 1.0 / time_per_forward_sec
    else:
        per_gpu_tokens_per_sec = 0.0
        per_user_tokens_per_sec = 0.0

    if traffic_pattern in TRAFFIC_PATTERN_HOURS:
        billed_hours_per_week = TRAFFIC_PATTERN_HOURS[traffic_pattern]
    elif traffic_pattern == "cold_per_query":
        billed_hours_per_week = inp.get("hot_hours_per_week", 0)
    else:
        raise ValueError(f"unknown traffic_pattern {traffic_pattern}")

    # --- How many GPUs to keep up with output-token demand in the billed window ---
    output_tokens_per_week = queries_per_week * avg_output_tokens
    billed_seconds = billed_hours_per_week * 3600.0
    required_tokens_per_sec = (output_tokens_per_week / billed_seconds) if billed_seconds > 0 else 0.0
    if not fits:
        gpus_needed = 0
    elif per_gpu_tokens_per_sec > 0:
        gpus_needed = max(1, math.ceil(required_tokens_per_sec / per_gpu_tokens_per_sec))
    else:
        gpus_needed = 1

    selfhost_weekly_usd = gpus_needed * billed_hours_per_week * gpu_usd_per_hr
    api_weekly_usd = queries_per_week * api_cost_per_query_usd
    weekly_savings_usd = api_weekly_usd - selfhost_weekly_usd
    savings_pct = (weekly_savings_usd / api_weekly_usd * 100.0) if api_weekly_usd > 0 else 0.0

    if not fits:
        verdict = "infeasible"
    elif selfhost_weekly_usd < api_weekly_usd:
        verdict = "selfhost_wins"
    else:
        verdict = "api_wins"

    derivation = [
        {"step": "weights_gb", "formula": "params_b * bytes_per_param[quant]", "value": round(weights_gb, 4)},
        {"step": "kv_per_stream_gb", "formula": "2 * n_layers * d_model * kv_bytes_per_elem / gqa_ratio * context_len / 1e9", "value": round(kv_per_stream_gb, 4)},
        {"step": "max_concurrency_vram", "formula": "(vram_gb - weights_gb - activation_gb) // kv_per_stream_gb", "value": max_concurrency_vram},
        {"step": "concurrency", "formula": "min(max_concurrency_vram, compute_intensity/2)", "value": concurrency},
        {"step": "vram_needed_gb", "formula": "weights_gb + activation_gb + kv_per_stream_gb * concurrency", "value": round(vram_needed_gb, 4)},
        {"step": "per_gpu_tokens_per_sec", "formula": "concurrency / ((weights_gb + kv_per_stream_gb*concurrency) / mem_bandwidth)", "value": round(per_gpu_tokens_per_sec, 4)},
        {"step": "required_tokens_per_sec", "formula": "queries_per_week * avg_output_tokens / (billed_hours * 3600)", "value": round(required_tokens_per_sec, 4)},
        {"step": "gpus_needed", "formula": "ceil(required_tokens_per_sec / per_gpu_tokens_per_sec)", "value": gpus_needed},
        {"step": "billed_hours", "formula": "pattern -> hours/week", "value": billed_hours_per_week},
        {"step": "selfhost_weekly_usd", "formula": "gpus_needed * billed_hours * usd_per_hr", "value": round(selfhost_weekly_usd, 4)},
        {"step": "api_weekly_usd", "formula": "queries_per_week * api_cost_per_query_usd", "value": round(api_weekly_usd, 4)},
    ]

    return {
        "fits": fits,
        "weights_gb": round(weights_gb, 4),
        "kv_per_stream_gb": round(kv_per_stream_gb, 4),
        "kv_cache_gb": round(kv_cache_gb, 4),
        "concurrency": concurrency,
        "max_concurrency_vram": max_concurrency_vram,
        "vram_needed_gb": round(vram_needed_gb, 4),
        "vram_headroom_gb": round(vram_headroom_gb, 4),
        "per_gpu_tokens_per_sec": round(per_gpu_tokens_per_sec, 4),
        "per_user_tokens_per_sec": round(per_user_tokens_per_sec, 4),
        "required_tokens_per_sec": round(required_tokens_per_sec, 4),
        "gpus_needed": gpus_needed,
        "billed_hours_per_week": billed_hours_per_week,
        "selfhost_weekly_usd": round(selfhost_weekly_usd, 4),
        "api_weekly_usd": round(api_weekly_usd, 4),
        "weekly_savings_usd": round(weekly_savings_usd, 4),
        "savings_pct": round(savings_pct, 4),
        "verdict": verdict,
        "derivation": derivation,
    }


def compute_finetune(inp: dict) -> dict:
    active_params_b = _require(inp, "active_params_b")
    total_params_b = _require(inp, "total_params_b")
    method = _require(inp, "method")
    num_examples = _require(inp, "num_examples")
    tokens_per_example = _require(inp, "tokens_per_example")
    epochs = _require(inp, "epochs")
    experiments_multiplier_in = _require(inp, "experiments_multiplier")
    prep_cost_usd = _require(inp, "prep_cost_usd")
    gpu = _require(inp, "gpu")
    gpu_vram_gb = _require(gpu, "vram_gb")
    gpu_usd_per_hr = _require(gpu, "usd_per_hr")
    gpu_bf16_tflops = _require(gpu, "bf16_tflops")
    gpus_per_node = gpu.get("gpus_per_node", 8)

    if method not in FT_METHODS:
        raise ValueError(f"unknown method {method}")

    m = FT_METHODS[method]

    total_tokens = num_examples * tokens_per_example * epochs
    full_flops = FLOPS_PER_TOKEN_PER_PARAM * active_params_b * 1e9 * total_tokens
    method_flops = full_flops * m["compute_mult"]

    peak_flops_per_sec = gpu_bf16_tflops * 1e12
    effective_flops_per_sec = peak_flops_per_sec * BASELINE_MFU * m["mfu_penalty"]
    single_gpu_hours = method_flops / effective_flops_per_sec / 3600.0

    ft_vram_gb = total_params_b * m["bytes_per_param"]
    node_vram_gb = gpu_vram_gb * gpus_per_node

    if ft_vram_gb <= gpu_vram_gb:
        cluster_overhead = 1.10
        cluster_topology = "single-gpu"
    elif ft_vram_gb <= node_vram_gb * 1.15:
        cluster_overhead = 1.35
        cluster_topology = "multi-gpu"
    else:
        cluster_overhead = 1.70
        cluster_topology = "multi-node"

    hours_with_cluster = single_gpu_hours * cluster_overhead
    single_run_gpu_cost_usd = hours_with_cluster * gpu_usd_per_hr
    experiments_multiplier = max(1.0, experiments_multiplier_in)
    gpu_cost_total_usd = single_run_gpu_cost_usd * experiments_multiplier
    total_capex_usd = gpu_cost_total_usd + prep_cost_usd

    derivation = [
        {"step": "total_tokens", "formula": "num_examples * tokens_per_example * epochs", "value": total_tokens},
        {"step": "method_flops", "formula": "6 * active_params * total_tokens * method_multiplier", "value": method_flops},
        {"step": "effective_flops_per_sec", "formula": "bf16_tflops * 1e12 * BASELINE_MFU * mfu_penalty", "value": effective_flops_per_sec},
        {"step": "single_gpu_hours", "formula": "method_flops / effective_flops_per_sec / 3600", "value": single_gpu_hours},
        {"step": "ft_vram_gb", "formula": "total_params_b * bytes_per_param[method]", "value": ft_vram_gb},
        {"step": "cluster_overhead", "formula": "topology bucket from ft_vram_gb vs node_vram_gb", "value": cluster_overhead},
        {"step": "hours_with_cluster", "formula": "single_gpu_hours * cluster_overhead", "value": hours_with_cluster},
        {"step": "single_run_gpu_cost_usd", "formula": "hours_with_cluster * usd_per_hr", "value": single_run_gpu_cost_usd},
        {"step": "gpu_cost_total_usd", "formula": "single_run_gpu_cost_usd * experiments_multiplier", "value": gpu_cost_total_usd},
        {"step": "total_capex_usd", "formula": "gpu_cost_total_usd + prep_cost_usd", "value": total_capex_usd},
    ]

    return {
        "total_tokens": total_tokens,
        "method_flops": method_flops,
        "effective_flops_per_sec": effective_flops_per_sec,
        "single_gpu_hours": round(single_gpu_hours, 4),
        "ft_vram_gb": round(ft_vram_gb, 4),
        "cluster_overhead": cluster_overhead,
        "cluster_topology": cluster_topology,
        "hours_with_cluster": round(hours_with_cluster, 4),
        "single_run_gpu_cost_usd": round(single_run_gpu_cost_usd, 4),
        "experiments_multiplier": experiments_multiplier,
        "gpu_cost_total_usd": round(gpu_cost_total_usd, 4),
        "prep_cost_usd": prep_cost_usd,
        "total_capex_usd": round(total_capex_usd, 4),
        "derivation": derivation,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing subcommand"}))
        sys.exit(2)
    sub = sys.argv[1]
    if sub not in ("inference", "finetune"):
        print(json.dumps({"error": f"unknown subcommand {sub}"}))
        sys.exit(2)
    try:
        inp = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON: {e}"}))
        sys.exit(2)
    try:
        if sub == "inference":
            result = compute_inference(inp)
        else:
            result = compute_finetune(inp)
    except KeyError as e:
        print(json.dumps({"error": f"missing field {e.args[0]}"}))
        sys.exit(2)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
