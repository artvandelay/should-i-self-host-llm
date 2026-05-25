import { describe, it, expect } from "vitest";

import {
  computeFtCapex,
  cumulativeProjection,
  pickFtGpu,
  WEEKS_PER_MONTH,
  type Pricing,
} from "../src/engine";

const baseInputs = {
  num_examples: 100_000,
  tokens_per_example: 1000,
  method: "lora" as const,
  epochs: 3,
  prep_cost_usd: 0,
};

describe("computeFtCapex", () => {
  it("lora < qlora < full wall-clock for same 70B workload 100k examples", () => {
    // LoRA and QLoRA do the same FLOPs (~2/3 of full FT — backward through
    // frozen base weights still dominates). But QLoRA pays a dequantization
    // tax (~45% slower per-step than LoRA on real hardware, per TildAlice
    // benchmarks). QLoRA's win is memory (fits on smaller GPU), not speed.
    const qlora = computeFtCapex(70, { ...baseInputs, method: "qlora" });
    const lora = computeFtCapex(70, { ...baseInputs, method: "lora" });
    const full = computeFtCapex(70, { ...baseInputs, method: "full" });
    expect(lora.gpu_cost_usd).toBeLessThan(qlora.gpu_cost_usd);
    expect(qlora.gpu_cost_usd).toBeLessThan(full.gpu_cost_usd);
  });

  it("Guanaco-65B QLoRA anchor: ~24 hours on single pro GPU equivalent", () => {
    // QLoRA paper (Dettmers et al. 2023) trained Guanaco-65B in 24h on a
    // single 48GB pro GPU (~150 TFLOPS BF16). Dataset OASST1: ~10k examples,
    // ~500 tokens/ex, ~3 epochs. Our engine assumes a single H100 (~3x faster
    // than that pro GPU), so we expect ~5-15 GPU-hours on H100 — within 3x
    // of the paper's anchor when scaled to H100 throughput.
    const r = computeFtCapex(65, {
      num_examples: 10_000,
      tokens_per_example: 500,
      method: "qlora",
      epochs: 3,
      prep_cost_usd: 0,
    });
    expect(r.gpu_hours).toBeGreaterThan(2);
    expect(r.gpu_hours).toBeLessThan(20);
  });

  it("prep_cost_usd added to total", () => {
    const noPrep = computeFtCapex(70, { ...baseInputs, prep_cost_usd: 0 });
    const withPrep = computeFtCapex(70, { ...baseInputs, prep_cost_usd: 5000 });
    expect(withPrep.total_capex_usd).toBe(noPrep.total_capex_usd + 5000);
    expect(withPrep.gpu_cost_usd).toBe(noPrep.gpu_cost_usd);
  });

  it("NaN/negative inputs clamp to 0", () => {
    const r = computeFtCapex(NaN, {
      num_examples: -100,
      tokens_per_example: NaN,
      method: "lora",
      epochs: -2,
      prep_cost_usd: -500,
    });
    expect(r.gpu_hours).toBe(0);
    expect(r.gpu_cost_usd).toBe(0);
    expect(r.total_capex_usd).toBe(0);
  });

  it("doubles examples doubles gpu cost", () => {
    const single = computeFtCapex(70, { ...baseInputs, num_examples: 50_000 });
    const doubled = computeFtCapex(70, { ...baseInputs, num_examples: 100_000 });
    expect(doubled.gpu_cost_usd).toBeCloseTo(single.gpu_cost_usd * 2, 6);
  });
});

describe("pickFtGpu", () => {
  it("picks best $/TFLOP-hr, not smallest VRAM", () => {
    // A100 ($1.6/hr / 312 TFLOPS) = $5.13/TFLOP-hr loses to
    // H100 ($2.9/hr / 989 TFLOPS) = $2.93/TFLOP-hr.
    const fakePricing: Pricing = {
      last_updated: "test",
      gpus: [
        {
          name: "A100 80GB",
          vram_gb: 80,
          modal_per_hr: 2.1,
          lambda_per_hr: 1.85,
          runpod_per_hr: 1.6,
          bf16_tflops: 312,
          single_gpu_vram_gb: 80,
          gpus_per_node: 8,
        },
        {
          name: "H100 80GB",
          vram_gb: 80,
          modal_per_hr: 4.2,
          lambda_per_hr: 3.3,
          runpod_per_hr: 2.9,
          bf16_tflops: 989,
          single_gpu_vram_gb: 80,
          gpus_per_node: 8,
        },
      ],
      apis: {},
    };
    expect(pickFtGpu(fakePricing)?.name).toBe("H100 80GB");
  });

  it("picks a hypothetical B200 over H100 when cheaper $/TFLOP-hr", () => {
    const fakePricing: Pricing = {
      last_updated: "test",
      gpus: [
        {
          name: "H100 80GB",
          vram_gb: 80,
          modal_per_hr: 4.2,
          lambda_per_hr: 3.3,
          runpod_per_hr: 2.9,
          bf16_tflops: 989,
        },
        {
          name: "B200 192GB",
          vram_gb: 192,
          modal_per_hr: 6.0,
          lambda_per_hr: 5.5,
          runpod_per_hr: 5.0,
          bf16_tflops: 2250,
        },
      ],
      apis: {},
    };
    expect(pickFtGpu(fakePricing)?.name).toBe("B200 192GB");
  });

  it("returns null when no GPU is tagged for training", () => {
    const fakePricing: Pricing = {
      last_updated: "test",
      gpus: [
        { name: "L4 24GB", vram_gb: 24, modal_per_hr: 0.8, lambda_per_hr: 0.7, runpod_per_hr: 0.55 },
      ],
      apis: {},
    };
    expect(pickFtGpu(fakePricing)).toBeNull();
  });
});

describe("cumulativeProjection", () => {
  it("points.length = horizon+1", () => {
    const proj = cumulativeProjection(100, 50, 1000, 24);
    expect(proj.points.length).toBe(25);
    expect(proj.horizon_months).toBe(24);
  });

  it("month 0: api=0, selfhost=capex", () => {
    const proj = cumulativeProjection(500, 200, 8000, 12);
    expect(proj.points[0]).toEqual({
      month: 0,
      api_cumulative: 0,
      selfhost_cumulative: 8000,
    });
  });

  it("crossover detected when api=1000, sh=100, capex=10000", () => {
    const proj = cumulativeProjection(1000, 100, 10_000, 24);
    expect(proj.crossover_month).not.toBeNull();
    const m = proj.crossover_month!;
    const weeks = m * WEEKS_PER_MONTH;
    expect(100 * weeks + 10_000).toBeLessThanOrEqual(1000 * weeks);
    if (m > 1) {
      const prevWeeks = (m - 1) * WEEKS_PER_MONTH;
      expect(100 * prevWeeks + 10_000).toBeGreaterThan(1000 * prevWeeks);
    }
  });

  it("null crossover when api=100, sh=100", () => {
    const proj = cumulativeProjection(100, 100, 5000, 24);
    expect(proj.crossover_month).toBeNull();
  });

  it("null crossover when capex huge", () => {
    const proj = cumulativeProjection(200, 100, 10_000_000, 24);
    expect(proj.crossover_month).toBeNull();
  });
});
