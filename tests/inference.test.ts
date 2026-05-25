import { describe, it, expect } from "vitest";
import {
  evaluateConfig,
  throughputTokensPerSec,
  type EvalArgs,
  type Pricing,
} from "../src/engine";

// Synthetic pricing fixture so tests don't depend on the bundled catalog.
const pricing: Pricing = {
  last_updated: "test",
  gpus: [
    { name: "L4 24GB", vram_gb: 24, modal_per_hr: 0.8, lambda_per_hr: 0.7, runpod_per_hr: 0.55 },
    { name: "A100 80GB", vram_gb: 80, modal_per_hr: 3.1, lambda_per_hr: 2.5, runpod_per_hr: 2.2, bf16_tflops: 312, single_gpu_vram_gb: 80, gpus_per_node: 8 },
    { name: "H100 80GB", vram_gb: 80, modal_per_hr: 4.2, lambda_per_hr: 3.3, runpod_per_hr: 2.9, bf16_tflops: 989, single_gpu_vram_gb: 80, gpus_per_node: 8 },
    { name: "8xH100 640GB", vram_gb: 640, modal_per_hr: 33.6, lambda_per_hr: 26.4, runpod_per_hr: 23.2, bf16_tflops: 989, single_gpu_vram_gb: 80, gpus_per_node: 8 },
  ],
  apis: {},
};

const baseArgs: Omit<EvalArgs, "params_b" | "active_params_b" | "arch"> = {
  pricing,
  quant: "fp16",
  queries_per_week: 10_000,
  output_tokens: 500,
  pattern: "uniform",
  vendor: "runpod",
};

describe("throughputTokensPerSec — multi-GPU scaling", () => {
  it("8x80GB row reports ~8x the per-unit throughput", () => {
    // Anchor: single H100 (80 GB, 989 TFLOPS, 8B params) ≈ 120 tok/s.
    const single = throughputTokensPerSec(8, 80, 989, 80);
    const eight = throughputTokensPerSec(8, 640, 989, 80);
    expect(eight / single).toBeCloseTo(8, 1);
  });

  it("future high-TFLOPS row (B200-ish, 2250) scales linearly", () => {
    const h100 = throughputTokensPerSec(8, 80, 989, 80);
    const b200 = throughputTokensPerSec(8, 80, 2250, 80);
    expect(b200 / h100).toBeCloseTo(2250 / 989, 2);
  });
});

describe("evaluateConfig — MoE inference uses active for FLOPs, total for VRAM", () => {
  it("Mixtral-class MoE (47B/12B) needs less throughput than dense 47B", () => {
    const moe = evaluateConfig({ ...baseArgs, params_b: 47, active_params_b: 12, arch: "moe" });
    const dense = evaluateConfig({ ...baseArgs, params_b: 47, active_params_b: 47, arch: "dense" });
    expect(moe).not.toBeNull();
    expect(dense).not.toBeNull();
    expect(moe!.tps).toBeGreaterThan(dense!.tps); // smaller active → faster
  });

  it("MoE picks GPU based on TOTAL params VRAM (not active)", () => {
    // 200B MoE at int8 → ~200 GB weights → forces multi-GPU row even though
    // active is 17B (which alone would fit on a single H100).
    const moe = evaluateConfig({ ...baseArgs, params_b: 200, active_params_b: 17, arch: "moe", quant: "int8" });
    expect(moe).not.toBeNull();
    expect(moe!.gpu).toBe("8xH100 640GB");
  });
});

describe("evaluateConfig — input_tokens included in compute time", () => {
  it("higher input_tokens => higher self-host cost (RAG case)", () => {
    const small_in = evaluateConfig({ ...baseArgs, params_b: 8, active_params_b: 8, arch: "dense", input_tokens: 100 });
    const big_in = evaluateConfig({ ...baseArgs, params_b: 8, active_params_b: 8, arch: "dense", input_tokens: 20_000 });
    expect(small_in).not.toBeNull();
    expect(big_in).not.toBeNull();
    expect(big_in!.weekly_cost).toBeGreaterThan(small_in!.weekly_cost);
  });

  it("default (no input_tokens) === input_tokens: 0", () => {
    const a = evaluateConfig({ ...baseArgs, params_b: 8, active_params_b: 8, arch: "dense" });
    const b = evaluateConfig({ ...baseArgs, params_b: 8, active_params_b: 8, arch: "dense", input_tokens: 0 });
    expect(a?.weekly_cost).toBeCloseTo(b!.weekly_cost, 6);
  });
});

describe("evaluateConfig — billing math", () => {
  it("per-second billing isn't multiplied by replicas (no double-count)", () => {
    // High QPS bursty pattern → saturated → replicas_needed > 1.
    const r = evaluateConfig({
      ...baseArgs,
      params_b: 70,
      active_params_b: 70,
      arch: "dense",
      queries_per_week: 5_000_000, // ridiculous to force saturation
      output_tokens: 1000,
      pattern: "bursty",
    });
    expect(r).not.toBeNull();
    const perSec = r!.all_billing_options.find((b) => b.mode === "per_second");
    expect(perSec).toBeDefined();
    // per_second weekly_cost should equal billed_hours * price_per_hr exactly
    // (no hidden replica multiplier in the cost line).
    expect(perSec!.weekly_cost).toBeCloseTo(perSec!.billed_hours * r!.gpu_price_per_hr, 4);
  });

  it("picks GPU by min weekly cost, not min $/hr (H100 beats A100 at high QPS)", () => {
    // Restrict to just H100 80GB vs A100 80GB so we test the picker's
    // throughput-vs-rate trade-off in isolation.
    const restricted: Pricing = {
      ...pricing,
      gpus: pricing.gpus.filter((g) => g.name === "A100 80GB" || g.name === "H100 80GB"),
    };
    const r = evaluateConfig({
      ...baseArgs,
      pricing: restricted,
      params_b: 30,
      active_params_b: 30,
      arch: "dense",
      queries_per_week: 2_000_000,
      pattern: "uniform",
    });
    expect(r).not.toBeNull();
    expect(r!.gpu).toBe("H100 80GB"); // 989 TFLOPS @ $2.9 beats 312 TFLOPS @ $2.2
  });

  it("hourly billing scales by per-hour replicas, not peak", () => {
    // Bursty pattern with realistic peak: hourly should be less than
    // always-on (less than peak * 168 * price).
    const r = evaluateConfig({
      ...baseArgs,
      params_b: 70,
      active_params_b: 70,
      arch: "dense",
      queries_per_week: 100_000,
      pattern: "bursty",
    });
    expect(r).not.toBeNull();
    const hourly = r!.all_billing_options.find((b) => b.mode === "hourly");
    const always = r!.all_billing_options.find((b) => b.mode === "always_on");
    expect(hourly).toBeDefined();
    expect(always).toBeDefined();
    expect(hourly!.weekly_cost).toBeLessThanOrEqual(always!.weekly_cost);
  });
});
