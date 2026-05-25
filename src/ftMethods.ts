export type FtMethod = "lora" | "qlora" | "full";

export interface FtMethodSpec {
  id: FtMethod;
  label: string;
  /**
   * Fraction of full fine-tuning FLOPs this method actually consumes.
   *
   * Common misconception: LoRA/QLoRA reduce trainable *parameters* by ~99%, so
   * people assume compute drops by ~99% too. It does not. The backward pass
   * must still compute activation gradients through the frozen base weights,
   * which dominates cost. Per the EU AI Act SageMaker formula
   * (F_ft = 4·N_total + 2·N_trainable) and CE-LoRA's analysis
   * (arxiv 2502.01378), LoRA reduces backward compute by AT MOST half — so
   * total training compute drops to ~2/3 of full FT, not to a few percent.
   */
  computeMultiplier: number;
  /**
   * Effective MFU multiplier vs. a well-tuned full-FT baseline.
   * QLoRA pays a dequantization tax (~45% slower per-step than LoRA on A100
   * per TildAlice benchmarks); LoRA's smaller matmuls hit lower arithmetic
   * intensity than full FT. Captures real-world wall-clock deviation from
   * the pure-FLOPs estimate.
   */
  mfuPenalty: number;
  description: string;
  citation: string;
}

export const FT_METHODS: Record<FtMethod, FtMethodSpec> = {
  lora: {
    id: "lora",
    label: "LoRA",
    computeMultiplier: 0.67,
    mfuPenalty: 0.85,
    description: "Low-rank adaptation — train small adapter matrices",
    citation: "Hu et al. 2021 (arxiv 2106.09685); CE-LoRA arxiv 2502.01378",
  },
  qlora: {
    id: "qlora",
    label: "QLoRA",
    computeMultiplier: 0.67,
    mfuPenalty: 0.70,
    description: "Quantized LoRA — 4-bit base weights, full-precision adapters",
    citation: "Dettmers et al. 2023 (arxiv 2305.14314)",
  },
  full: {
    id: "full",
    label: "Full fine-tune",
    computeMultiplier: 1.0,
    mfuPenalty: 1.0,
    description: "Update all model parameters",
    citation: "Kaplan et al. 2020, arxiv 2001.08361 (eq 2.1)",
  },
};

/**
 * Training FLOPs per token per parameter (Kaplan 6N rule):
 * 2N forward + 4N backward = 6N. This is the standard Chinchilla/Kaplan
 * training-compute approximation, not the forward-only inference cost.
 */
export const FLOPS_PER_TOKEN_PER_PARAM = 6;

/**
 * Baseline sustained model-FLOPs-utilization (MFU) assumed for fine-tuning.
 * 30% reflects realistic FT workloads (small effective batches, attention-
 * heavy, single-node) per industry reporting (technolynx, stevengong notes).
 * Pretraining at scale can hit 40–50% MFU but FT recipes typically don't.
 * Per-method adjustments live in FT_METHODS[m].mfuPenalty (e.g. QLoRA's
 * dequant tax).
 */
export const BASELINE_MFU = 0.3;

/** H100 FP16 peak throughput × baseline MFU. */
export const H100_FP16_FLOPS_PER_SEC = 989e12 * BASELINE_MFU;
