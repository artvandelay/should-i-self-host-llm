export type FtMethod = "lora" | "qlora" | "full";

export interface FtMethodSpec {
  id: FtMethod;
  label: string;
  computeMultiplier: number;
  description: string;
  citation: string;
}

export const FT_METHODS: Record<FtMethod, FtMethodSpec> = {
  lora: {
    id: "lora",
    label: "LoRA",
    computeMultiplier: 0.05,
    description: "Low-rank adaptation — train small adapter matrices",
    citation: "Hu et al. 2021, arxiv 2106.09685",
  },
  qlora: {
    id: "qlora",
    label: "QLoRA",
    computeMultiplier: 0.03,
    description: "Quantized LoRA — 4-bit base weights, full-precision adapters",
    citation: "Dettmers et al. 2023, arxiv 2305.14314",
  },
  full: {
    id: "full",
    label: "Full fine-tune",
    computeMultiplier: 1.0,
    description: "Update all model parameters",
    citation: "Kaplan et al. 2020, arxiv 2001.08361 (eq 2.1)",
  },
};

/** Forward-pass FLOPs per token per parameter (Kaplan scaling). */
export const FLOPS_PER_TOKEN_PER_PARAM = 6;

/** H100 FP16 peak throughput × 40% MFU (typical sustained utilization). */
export const H100_FP16_FLOPS_PER_SEC = 989e12 * 0.4;
