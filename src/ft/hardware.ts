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
 * Tolerance band on the VRAM-to-node boundary. Activation checkpointing,
 * ZeRO/FSDP CPU offload, and aggressive sequence packing all let real
 * workloads fit ~15% more VRAM than the naive `bytes-per-param × params`
 * estimate. Without this, models like Mixtral 8x7B full-FT (666 GB vs an
 * 8xH100 node's 640 GB) flip to "multi-node" and pay the IB tax even
 * though almost everyone runs them intra-node. 1.15× is conservative.
 */
const NODE_VRAM_TOLERANCE = 1.15;

/**
 * Effective-utilization multipliers applied to GPU-hours to account for
 * the fact that nobody actually gets 100% of theoretical throughput, even
 * before NCCL crosses a wire.
 *
 *   single-gpu  1.10×  — kernel launch overhead, dataloader stalls, HtoD
 *                        transfer; ~90% effective utilization.
 *   multi-gpu   1.35×  — adds NCCL all-reduce on NVLink (~5-10% loss
 *                        beyond single-gpu) plus uneven-batch idle bubbles.
 *   multi-node  1.70×  — adds Infiniband cross-node communication
 *                        (~25-30% loss vs single, on top of the above).
 *
 * Values sit inside the published industry range (FSDP/DeepSpeed reports:
 * 1.05–1.5× intra-node, 1.4–1.8× inter-node). The single-gpu floor of
 * 1.0× was too generous — even a perfectly tuned single-GPU run loses
 * a few percent to non-compute time.
 */
const OVERHEAD_SINGLE_GPU = 1.10;
const OVERHEAD_MULTI_GPU = 1.35;
const OVERHEAD_MULTI_NODE = 1.70;

/**
 * Pick a cluster-overhead multiplier from the FT VRAM footprint, sized to the
 * actual training GPU's per-unit VRAM and node size.
 *   ft_vram ≤ single_gpu_vram                              → single-gpu (1.10×)
 *   ft_vram ≤ single_gpu_vram × gpus_per_node × tolerance  → multi-gpu  (1.35×)
 *   ft_vram > that                                         → multi-node (1.70×)
 *
 * Defaults to 80 GB / 8 GPUs per node — sane for H100/H200/B200 SXM — but
 * the actual numbers come from the GpuRow so a future MI300X (192 GB) or
 * GB200 NVL36 (36 GPUs/node) just works.
 *
 * The multi-node boundary has a 15% tolerance band — see NODE_VRAM_TOLERANCE.
 */
export function pickClusterOverhead(
  ft_vram_gb: number,
  single_gpu_vram_gb = 80,
  gpus_per_node = 8
): { multiplier: number; topology: "single-gpu" | "multi-gpu" | "multi-node" } {
  const node_vram = single_gpu_vram_gb * gpus_per_node;
  if (ft_vram_gb <= single_gpu_vram_gb)
    return { multiplier: OVERHEAD_SINGLE_GPU, topology: "single-gpu" };
  if (ft_vram_gb <= node_vram * NODE_VRAM_TOLERANCE)
    return { multiplier: OVERHEAD_MULTI_GPU, topology: "multi-gpu" };
  return { multiplier: OVERHEAD_MULTI_NODE, topology: "multi-node" };
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
