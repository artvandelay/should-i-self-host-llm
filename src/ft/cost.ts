import type {
  ModelSpec,
  FtTraining,
  FtOptions,
  FtMethod,
  FtCostBreakdown,
  FtStage,
} from "./types";
import { FT_METHODS, FLOPS_PER_TOKEN_PER_PARAM, BASELINE_MFU } from "./methods";
import { ftVramGb, pickClusterOverhead, pickFtGpu } from "./hardware";
import { PRICING } from "../engine";

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

  const num = clampNonNeg(training.num_examples);
  const tok = clampNonNeg(training.tokens_per_example);
  const epochs = clampNonNeg(training.epochs);
  const prep = clampNonNeg(training.prep_cost_usd);
  const active_params = clampNonNeg(spec.active_params_b) * 1e9;
  const total_params_b = clampNonNeg(spec.total_params_b);

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
  const cluster_overhead = clampNonNeg(
    options?.cluster_overhead ?? auto.multiplier,
    auto.multiplier
  );
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

  const cluster_topology: "single-gpu" | "multi-gpu" | "multi-node" =
    cluster_overhead <= 1.0
      ? "single-gpu"
      : cluster_overhead <= 1.3
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
  };
}
