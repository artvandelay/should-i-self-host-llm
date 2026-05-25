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

/** Pipeline output — totals plus every stage that produced them. */
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
}
