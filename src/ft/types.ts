/** Fine-tuning method. */
export type FtMethod = "lora" | "qlora" | "full";

/** Model architecture. Dense = all params fire per token; MoE = only active. */
export type Arch = "dense" | "moe";

/**
 * Single source of truth for the model being fine-tuned.
 * `total_params_b` drives VRAM and cluster-overhead sizing (every weight loads
 * into memory regardless of routing). `active_params_b` drives compute FLOPs
 * (only the experts that fire per token do work). For dense models the two
 * are equal.
 */
export interface ModelSpec {
  name: string;
  arch: Arch;
  total_params_b: number;
  active_params_b: number;
}

/** Training-workload inputs — what you're training on, not what you're training. */
export interface FtTraining {
  num_examples: number;
  tokens_per_example: number;
  epochs: number;
  /** One-time prep cost (data labeling, eng time). Not multiplied by experiments. */
  prep_cost_usd: number;
  /** Experiments / failed-runs multiplier. Clamped UP to 1.0. Default 2.5. */
  experiments_multiplier: number;
}

/** Optional caller overrides; the engine auto-picks sensible values otherwise. */
export interface FtOptions {
  /** Override cluster-overhead multiplier. Auto-picked from VRAM if omitted. */
  cluster_overhead?: number;
}

/** One named stage in the cost pipeline — kept for transparency/debugging. */
export interface FtStage {
  name: string;
  value: number;
  unit: string;
  formula: string;
  source: string;
}

/**
 * Soft warning attached to a cost breakdown. The engine still returns a
 * number even when a warning fires — it's the caller's job (UI) to decide
 * whether to surface, gray out, or block on a warning.
 *
 * Severity:
 *   - "info"     — heads-up; non-physical input was silently clamped
 *   - "warning"  — result is plausible but the input regime is suspect
 *   - "blocker"  — result is mathematically derived but is almost certainly
 *                  the wrong tool for this question (e.g. pretraining)
 */
export interface FtWarning {
  code:
    | "active_exceeds_total"        // MoE spec inconsistency
    | "vram_exceeds_known_hardware" // FT VRAM > biggest catalog SKU × node
    | "pretrain_scale_workload"     // tokens >> Chinchilla-optimal for params
    | "trivial_workload"            // result rounds to ~$0; not meaningful
    | "cluster_overhead_clamped";   // user passed < 1.0; clamped to 1.0
  severity: "info" | "warning" | "blocker";
  message: string;
}

/** Pipeline output — totals plus every stage and any warnings that fired. */
export interface FtCostBreakdown {
  gpu_hours: number;
  gpu_cost_usd: number;
  total_capex_usd: number;
  method: FtMethod;
  single_run_gpu_cost_usd: number;
  experiments_multiplier: number;
  cluster_overhead: number;
  cluster_topology: "single-gpu" | "multi-gpu" | "multi-node";
  ft_vram_gb: number;
  stages: FtStage[];
  /** Soft caveats; never block computation. Empty array = clean run. */
  warnings: FtWarning[];
}
