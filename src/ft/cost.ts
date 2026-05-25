import type {
  ModelSpec,
  FtTraining,
  FtOptions,
  FtMethod,
  FtCostBreakdown,
  FtStage,
  FtWarning,
} from "./types";
import { FT_METHODS, FLOPS_PER_TOKEN_PER_PARAM, BASELINE_MFU } from "./methods";
import { ftVramGb, pickClusterOverhead, pickFtGpu } from "./hardware";
import { PRICING } from "../engine";

/**
 * Largest single-node VRAM available across the GPU catalog.
 * Used as the "does any known hardware fit this?" feasibility ceiling.
 * For an FT workload whose VRAM exceeds this, we still return a number
 * (the FLOPs math is independent of feasibility), but flag the caller.
 */
function largestKnownNodeVramGb(): number {
  let best = 0;
  for (const g of PRICING.gpus) {
    if (typeof g.bf16_tflops !== "number" || g.bf16_tflops <= 0) continue;
    const per = g.single_gpu_vram_gb ?? g.vram_gb ?? 0;
    const node = per * (g.gpus_per_node ?? 8);
    if (node > best) best = node;
  }
  return best || 640; // fallback to 8×H100 node if catalog has nothing tagged
}

/**
 * Pretraining-vs-fine-tuning heuristic. Chinchilla-optimal pretraining is
 * ~20 tokens per parameter; production FT recipes are typically 0.01–1
 * tokens per param. At >10 tokens/param the workload is closer to
 * continued pretraining than fine-tuning — same math, wrong tool framing.
 */
const PRETRAIN_TOKENS_PER_PARAM = 10;

/** Clamp a numeric input: reject NaN/Infinity, force non-negative. */
function clampNonNeg(n: number, fallback = 0): number {
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}

/**
 * Pipeline-style FT cost calculation. Records every intermediate value
 * as a named FtStage so the derivation is inspectable end-to-end.
 *
 * Math is intentionally bit-identical to the legacy `computeFtCapex` in
 * `src/engine.ts` — see Stream E's golden tests.
 */
export function computeFtCost(
  spec: ModelSpec,
  training: FtTraining,
  method: FtMethod,
  options?: FtOptions
): FtCostBreakdown {
  const stages: FtStage[] = [];
  const warnings: FtWarning[] = [];

  const num = clampNonNeg(training.num_examples);
  const tok = clampNonNeg(training.tokens_per_example);
  const epochs = clampNonNeg(training.epochs);
  const prep = clampNonNeg(training.prep_cost_usd);
  const active_params_b = clampNonNeg(spec.active_params_b);
  const active_params = active_params_b * 1e9;
  const total_params_b = clampNonNeg(spec.total_params_b);

  // Validate the ModelSpec for internal consistency. Don't reject — the math
  // is well-defined either way — but flag so the caller can warn the user.
  if (active_params_b > total_params_b && total_params_b > 0) {
    warnings.push({
      code: "active_exceeds_total",
      severity: "warning",
      message: `Model spec inconsistency: active params (${active_params_b}B) > total params (${total_params_b}B). One of the two is almost certainly wrong.`,
    });
  }

  const total_tokens = num * tok * epochs;
  stages.push({
    name: "total_tokens",
    value: total_tokens,
    unit: "tokens",
    formula: "num_examples × tokens_per_example × epochs",
    source: "FtTraining inputs",
  });

  const methodSpec = FT_METHODS[method];

  const full_flops = FLOPS_PER_TOKEN_PER_PARAM * active_params * total_tokens;
  stages.push({
    name: "full_ft_flops",
    value: full_flops,
    unit: "FLOPs",
    formula: "6 × active_params × total_tokens",
    source: "Kaplan 6N rule",
  });

  const method_flops = full_flops * methodSpec.computeMultiplier;
  stages.push({
    name: "method_flops",
    value: method_flops,
    unit: "FLOPs",
    formula: "full_ft_flops × FT_METHODS[method].computeMultiplier",
    source: `FT_METHODS.${method}.computeMultiplier=${methodSpec.computeMultiplier}`,
  });

  const ftGpu = pickFtGpu(PRICING);
  const peak_tflops = ftGpu?.bf16_tflops ?? 989;
  stages.push({
    name: "peak_tflops",
    value: peak_tflops,
    unit: "TFLOPS",
    formula: "ftGpu.bf16_tflops ?? 989",
    source: `pickFtGpu → ${ftGpu?.name ?? "fallback"}`,
  });

  const effective_flops_per_sec =
    peak_tflops * 1e12 * BASELINE_MFU * methodSpec.mfuPenalty;
  stages.push({
    name: "effective_flops_per_sec",
    value: effective_flops_per_sec,
    unit: "FLOPS",
    formula: "peak_tflops × 1e12 × BASELINE_MFU × method.mfuPenalty",
    source: "BASELINE_MFU × FT_METHODS[method].mfuPenalty",
  });

  const seconds = method_flops / effective_flops_per_sec;
  const single_gpu_hours = seconds / 3600;
  stages.push({
    name: "single_gpu_hours",
    value: single_gpu_hours,
    unit: "hours",
    formula: "method_flops / effective_flops_per_sec / 3600",
    source: "derived",
  });

  const ft_vram_gb = ftVramGb(total_params_b, method);
  stages.push({
    name: "ft_vram_gb",
    value: ft_vram_gb,
    unit: "GB",
    formula: "ftVramGb(total_params_b, method)",
    source: "ftVramGb (bytes-per-param table)",
  });

  const auto = pickClusterOverhead(
    ft_vram_gb,
    ftGpu?.single_gpu_vram_gb ?? ftGpu?.vram_gb ?? 80,
    ftGpu?.gpus_per_node ?? 8
  );
  // User override path: clamp non-negative first, then enforce the physical
  // floor of 1.0 (a "less than full speed" cluster doesn't exist — comms
  // can only add overhead, never subtract from compute time).
  let cluster_overhead = clampNonNeg(
    options?.cluster_overhead ?? auto.multiplier,
    auto.multiplier
  );
  if (
    options?.cluster_overhead !== undefined &&
    Number.isFinite(options.cluster_overhead) &&
    options.cluster_overhead < 1.0
  ) {
    cluster_overhead = 1.0;
    warnings.push({
      code: "cluster_overhead_clamped",
      severity: "info",
      message: `cluster_overhead override of ${options.cluster_overhead} is non-physical (<1.0); clamped to 1.0. Comms cost can only add overhead, never subtract from compute time.`,
    });
  }

  // Hardware feasibility: does the FT VRAM footprint fit any known node?
  const max_node_vram = largestKnownNodeVramGb();
  if (ft_vram_gb > max_node_vram) {
    warnings.push({
      code: "vram_exceeds_known_hardware",
      severity: "warning",
      message: `Estimated FT VRAM (${ft_vram_gb.toFixed(0)} GB) exceeds the largest known node in the catalog (${max_node_vram} GB). Cost is mathematically derived but may not be physically buildable with off-the-shelf clusters; expect multi-node sharding overhead beyond what cluster_overhead models.`,
    });
  }

  // Pretraining-scale workload — same FLOPs math, wrong tool framing.
  if (total_params_b > 0 && total_tokens > PRETRAIN_TOKENS_PER_PARAM * total_params_b * 1e9) {
    warnings.push({
      code: "pretrain_scale_workload",
      severity: "blocker",
      message: `Total tokens (${total_tokens.toExponential(1)}) exceeds ${PRETRAIN_TOKENS_PER_PARAM}× model params. This is closer to continued pretraining than fine-tuning — Chinchilla-optimal pretraining is ~20 tokens/param. FT estimates above this scale stop being meaningful.`,
    });
  }
  stages.push({
    name: "cluster_overhead",
    value: cluster_overhead,
    unit: "x",
    formula:
      "user override or pickClusterOverhead(ft_vram_gb, gpu_vram, gpus_per_node)",
    source:
      options?.cluster_overhead !== undefined
        ? "user override"
        : "pickClusterOverhead auto",
  });

  // Topology label tracks the EFFECTIVE overhead so user overrides surface
  // honestly (forcing 1.7× on a tiny model reads "multi-node", not "single").
  // Thresholds sit at the midpoints between the canonical overhead values
  // (1.10 / 1.35 / 1.70) so an exact match lands in the right bucket and
  // freeform user values round to the nearest topology.
  const cluster_topology: "single-gpu" | "multi-gpu" | "multi-node" =
    cluster_overhead < 1.225
      ? "single-gpu"
      : cluster_overhead < 1.525
        ? "multi-gpu"
        : "multi-node";

  const hours = single_gpu_hours * cluster_overhead;
  stages.push({
    name: "hours_with_cluster",
    value: hours,
    unit: "hours",
    formula: "single_gpu_hours × cluster_overhead",
    source: "derived",
  });

  const cheapestRate = ftGpu
    ? Math.min(ftGpu.modal_per_hr, ftGpu.lambda_per_hr, ftGpu.runpod_per_hr)
    : 4.0;
  stages.push({
    name: "cheapest_rate_per_hr",
    value: cheapestRate,
    unit: "USD",
    formula: "min(modal, lambda, runpod)",
    source: `pickFtGpu → ${ftGpu?.name ?? "fallback"}`,
  });

  const single_run_gpu_cost = hours * cheapestRate;
  stages.push({
    name: "single_run_gpu_cost",
    value: single_run_gpu_cost,
    unit: "USD",
    formula: "hours_with_cluster × cheapest_rate_per_hr",
    source: "derived",
  });

  const raw_xm = clampNonNeg(training.experiments_multiplier, 1);
  const xm = raw_xm < 1 ? 1 : raw_xm;
  stages.push({
    name: "experiments_multiplier",
    value: xm,
    unit: "x",
    formula: "clamped up to 1.0 from training.experiments_multiplier",
    source: "FtTraining.experiments_multiplier",
  });

  const gpu_cost_total = single_run_gpu_cost * xm;
  stages.push({
    name: "gpu_cost_total",
    value: gpu_cost_total,
    unit: "USD",
    formula: "single_run_gpu_cost × experiments_multiplier",
    source: "derived",
  });

  stages.push({
    name: "prep_cost",
    value: prep,
    unit: "USD",
    formula: "training.prep_cost_usd (not multiplied)",
    source: "FtTraining.prep_cost_usd",
  });

  const total_capex = gpu_cost_total + prep;
  stages.push({
    name: "total_capex",
    value: total_capex,
    unit: "USD",
    formula: "gpu_cost_total + prep_cost",
    source: "derived",
  });

  const gpu_hours_total = hours * xm;

  // Trivial workload — result is real but probably below decision threshold.
  // Threshold: < $1 GPU cost. Lets the UI gray out / hide a panel that's
  // effectively "$0.00" so users don't try to make decisions from noise.
  if (total_tokens > 0 && gpu_cost_total < 1.0) {
    warnings.push({
      code: "trivial_workload",
      severity: "info",
      message: `GPU cost (${gpu_cost_total.toFixed(4)}) rounds to under $1 — the workload is too small for the result to be meaningful. Either the dataset is a smoke-test, or the model is much smaller than the rest of the calculator assumes.`,
    });
  }

  return {
    gpu_hours: gpu_hours_total,
    gpu_cost_usd: gpu_cost_total,
    total_capex_usd: total_capex,
    method,
    single_run_gpu_cost_usd: single_run_gpu_cost,
    experiments_multiplier: xm,
    cluster_overhead,
    cluster_topology,
    ft_vram_gb,
    stages,
    warnings,
  };
}
