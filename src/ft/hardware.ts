import type { FtMethod } from "./types";
import type { GpuRow, Pricing } from "../engine";

/**
 * Rough VRAM needed to *fine-tune* a model of `params_b` billion params with
 * the given method. Much larger than inference VRAM because of activations,
 * gradients, and optimizer state.
 *
 * Bytes-per-param calibrated against published 70B numbers
 * (TildAlice / channel.tel):
 *   - full FT  (BF16 weights 2 + grads 2 + FP32 Adam m+v 8 + acts 2)  ≈ 14 B/p
 *   - LoRA     (BF16 frozen weights + small adapter state + acts)     ≈ 1.0 B/p
 *   - QLoRA    (4-bit frozen weights + tiny adapter state + acts)     ≈ 0.5 B/p
 *
 * Yields: 70B QLoRA ≈ 43 GB ✓; 70B LoRA ≈ 78 GB ✓; 70B full ≈ 988 GB ✓.
 */
export function ftVramGb(params_b: number, method: FtMethod): number {
  const bpp = method === "full" ? 14 : method === "lora" ? 1.0 : 0.5;
  return Math.max(0, params_b) * bpp + 8; // +8 GB constant overhead
}

/**
 * Pick a cluster-overhead multiplier from the FT VRAM footprint, sized to the
 * actual training GPU's per-unit VRAM and node size.
 *   ft_vram ≤ single_gpu_vram                      → 1.0× (single GPU)
 *   ft_vram ≤ single_gpu_vram × gpus_per_node      → 1.3× (intra-node NVLink)
 *   ft_vram > that                                 → 1.6× (multi-node IB)
 *
 * Defaults to 80 GB / 8 GPUs per node — sane for H100/H200/B200 SXM — but
 * the actual numbers come from the GpuRow so a future MI300X (192 GB) or
 * GB200 NVL36 (36 GPUs/node) just works.
 */
export function pickClusterOverhead(
  ft_vram_gb: number,
  single_gpu_vram_gb = 80,
  gpus_per_node = 8
): { multiplier: number; topology: "single-gpu" | "multi-gpu" | "multi-node" } {
  const node_vram = single_gpu_vram_gb * gpus_per_node;
  if (ft_vram_gb <= single_gpu_vram_gb)
    return { multiplier: 1.0, topology: "single-gpu" };
  if (ft_vram_gb <= node_vram)
    return { multiplier: 1.3, topology: "multi-gpu" };
  return { multiplier: 1.6, topology: "multi-node" };
}

/**
 * Pick the best GPU row for fine-tuning from the pricing config: any row with
 * `bf16_tflops > 0` is eligible. Among eligible rows, pick the one with the
 * **best $/TFLOP-hr** — i.e. the most compute per dollar — using the cheapest
 * of the three vendor rates as the price. That's the right metric for FT
 * because total cost = FLOPs / (TFLOPs × MFU) × $/hr, and FLOPs is fixed by
 * the workload; minimizing $/TFLOP minimizes total cost.
 *
 * (Multi-GPU rows like "8xH100 640GB" tend to lose this contest because their
 * $/hr scales linearly with GPU count while TFLOPs stays per-unit in this
 * field's interpretation — keep `bf16_tflops` per-single-GPU and the math
 * works out. Cluster comms overhead is modeled separately.)
 */
export function pickFtGpu(pricing: Pricing): GpuRow | null {
  const eligible = pricing.gpus.filter(
    (g) => typeof g.bf16_tflops === "number" && g.bf16_tflops > 0
  );
  if (eligible.length === 0) return null;
  const score = (g: GpuRow) => {
    const rate = Math.min(g.modal_per_hr, g.lambda_per_hr, g.runpod_per_hr);
    return rate / (g.bf16_tflops ?? 1); // $/TFLOP-hr — lower is better
  };
  return eligible.reduce((best, g) => (score(g) < score(best) ? g : best));
}
