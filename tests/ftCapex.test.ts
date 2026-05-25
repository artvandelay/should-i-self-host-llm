import { describe, it, expect } from "vitest";

import {
  computeFtCapex,
  computeFtCost,
  ftVramGb,
  pickClusterOverhead,
  pickFtGpu,
  type ModelSpec,
  type FtTraining,
} from "../src/ft";
import {
  cumulativeProjection,
  WEEKS_PER_MONTH,
  type Pricing,
} from "../src/engine";

const baseInputs = {
  num_examples: 100_000,
  tokens_per_example: 1000,
  method: "lora" as const,
  epochs: 3,
  prep_cost_usd: 0,
  experiments_multiplier: 1.0,
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
      experiments_multiplier: 1.0,
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
      experiments_multiplier: NaN,
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

describe("experiments_multiplier", () => {
  it("2x doubles gpu cost vs 1x", () => {
    const single = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 1 });
    const doubled = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 2 });
    expect(doubled.gpu_cost_usd).toBeCloseTo(single.gpu_cost_usd * 2, 6);
  });

  it("below 1 clamps to 1", () => {
    const low = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 0.5 });
    const one = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 1 });
    expect(low.gpu_cost_usd).toBeCloseTo(one.gpu_cost_usd, 6);
    expect(low.experiments_multiplier).toBe(1);
  });

  it("does not multiply prep_cost", () => {
    const r = computeFtCapex(70, {
      ...baseInputs,
      prep_cost_usd: 5000,
      experiments_multiplier: 2,
    });
    expect(r.total_capex_usd).toBeCloseTo(r.gpu_cost_usd + 5000, 6);
  });

  it("single_run_gpu_cost_usd is reported independent of multiplier", () => {
    const single = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 1 });
    const tripled = computeFtCapex(70, { ...baseInputs, experiments_multiplier: 3 });
    expect(tripled.single_run_gpu_cost_usd).toBeCloseTo(single.single_run_gpu_cost_usd, 6);
  });
});

describe("MoE: compute uses active params, VRAM uses total", () => {
  it("Llama-4-Scout-class (109B total / 17B active) costs like 17B for raw compute", () => {
    const moe = computeFtCapex(17, baseInputs, 109);   // 17 active, 109 total
    const dense17 = computeFtCapex(17, baseInputs);     // dense 17B
    // Raw single-GPU FLOPs (gpu_hours stripped of cluster overhead) must
    // match dense-17B exactly — active_params drives compute. MoE may pay
    // a higher cluster tax because total is bigger; that's expected.
    const moe_base = moe.gpu_hours / moe.cluster_overhead / moe.experiments_multiplier;
    const d17_base = dense17.gpu_hours / dense17.cluster_overhead / dense17.experiments_multiplier;
    expect(moe_base).toBeCloseTo(d17_base, 4);
  });

  it("MoE total drives VRAM/cluster overhead, not active", () => {
    // 397B total → ~207 GB FT VRAM (QLoRA) → multi-gpu (1.35x), even though
    // active is small. Pass method=qlora for the realistic case.
    const moe = computeFtCapex(
      17,
      { ...baseInputs, method: "qlora" },
      397
    );
    const dense17 = computeFtCapex(17, { ...baseInputs, method: "qlora" });
    expect(moe.cluster_topology).toBe("multi-gpu");
    expect(dense17.cluster_topology).toBe("single-gpu");
    // Same FLOPs but MoE pays the multi-gpu vs single-gpu overhead delta
    // (1.35 / 1.10 ≈ 1.227× more wall-clock).
    expect(moe.gpu_cost_usd).toBeCloseTo(dense17.gpu_cost_usd * (1.35 / 1.10), 4);
  });

  it("omitting total_params_b falls back to dense behavior", () => {
    const explicit = computeFtCapex(17, baseInputs, 17);
    const implicit = computeFtCapex(17, baseInputs);
    expect(explicit.gpu_cost_usd).toBeCloseTo(implicit.gpu_cost_usd, 6);
    expect(explicit.ft_vram_gb).toBeCloseTo(implicit.ft_vram_gb, 6);
  });
});

describe("cluster overhead", () => {
  it("ftVramGb hits the right ballpark for 70B", () => {
    expect(ftVramGb(70, "qlora")).toBeGreaterThan(35); // ≈ 43 GB
    expect(ftVramGb(70, "qlora")).toBeLessThan(60);
    expect(ftVramGb(70, "lora")).toBeGreaterThan(60);  // ≈ 78 GB
    expect(ftVramGb(70, "lora")).toBeLessThan(120);
    expect(ftVramGb(70, "full")).toBeGreaterThan(800); // ≈ 988 GB
  });

  it("pickClusterOverhead boundaries with default 80GB/8-per-node", () => {
    // Updated 2026: floors moved to 1.10 / 1.35 / 1.70 to acknowledge that
    // even single-GPU runs don't hit 100% utilization (kernel launches,
    // dataloader stalls). Node boundary has a 15% tolerance so workloads
    // like Mixtral-47B full-FT (~666 GB) still classify as multi-gpu, not
    // multi-node.
    expect(pickClusterOverhead(40).multiplier).toBe(1.10);    // fits 1 H100
    expect(pickClusterOverhead(40).topology).toBe("single-gpu");
    expect(pickClusterOverhead(200).multiplier).toBe(1.35);   // multi-GPU
    expect(pickClusterOverhead(200).topology).toBe("multi-gpu");
    expect(pickClusterOverhead(666).topology).toBe("multi-gpu");  // tolerance: Mixtral-47B full
    expect(pickClusterOverhead(2000).multiplier).toBe(1.70);  // multi-node
    expect(pickClusterOverhead(2000).topology).toBe("multi-node");
  });

  it("pickClusterOverhead respects future GPU sizing (MI300X 192GB)", () => {
    // 150GB fits in a single MI300X — should be single-gpu, not multi-gpu.
    expect(pickClusterOverhead(150, 192, 8).topology).toBe("single-gpu");
  });

  it("pickClusterOverhead respects GB200 NVL36 (36 per node)", () => {
    // 1000GB on 8-per-node H100 is multi-node; on 36-per-node GB200 it's
    // still intra-node multi-GPU.
    expect(pickClusterOverhead(1000, 80, 8).topology).toBe("multi-node");
    expect(pickClusterOverhead(1000, 192, 36).topology).toBe("multi-gpu");
  });

  it("user override beats auto-pick", () => {
    const auto = computeFtCapex(7, { ...baseInputs, method: "lora" }); // tiny → auto 1.10×
    const forced = computeFtCapex(7, {
      ...baseInputs,
      method: "lora",
      cluster_overhead: 1.70,
    });
    expect(auto.cluster_overhead).toBe(1.10);
    expect(forced.cluster_overhead).toBe(1.70);
    expect(forced.gpu_cost_usd).toBeCloseTo(auto.gpu_cost_usd * (1.70 / 1.10), 6);
  });

  it("70B full-FT auto-picks multi-node (~988 GB > one node)", () => {
    const r = computeFtCapex(70, { ...baseInputs, method: "full" });
    expect(r.cluster_topology).toBe("multi-node");
    expect(r.cluster_overhead).toBe(1.70);
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

describe("FtWarning soft caveats (computeFtCost)", () => {
  const denseSpec = (n: number): ModelSpec => ({
    name: "test", arch: "dense", total_params_b: n, active_params_b: n,
  });
  const training: FtTraining = {
    num_examples: 10_000, tokens_per_example: 1000, epochs: 3,
    prep_cost_usd: 0, experiments_multiplier: 1,
  };

  it("emits no warnings on a normal workload", () => {
    const r = computeFtCost(denseSpec(8), training, "lora");
    expect(r.warnings).toEqual([]);
  });

  it("emits active_exceeds_total when MoE spec is inconsistent", () => {
    const r = computeFtCost(
      { name: "broken-moe", arch: "moe", total_params_b: 20, active_params_b: 100 },
      training,
      "lora"
    );
    expect(r.warnings.some(w => w.code === "active_exceeds_total")).toBe(true);
  });

  it("emits cluster_overhead_clamped when user passes < 1.0 and clamps to 1.0", () => {
    const r = computeFtCost(denseSpec(8), training, "lora", { cluster_overhead: 0.5 });
    expect(r.cluster_overhead).toBe(1.0);
    expect(r.warnings.some(w => w.code === "cluster_overhead_clamped")).toBe(true);
  });

  it("does NOT clamp or warn when user passes >= 1.0", () => {
    const r = computeFtCost(denseSpec(8), training, "lora", { cluster_overhead: 1.5 });
    expect(r.cluster_overhead).toBe(1.5);
    expect(r.warnings.some(w => w.code === "cluster_overhead_clamped")).toBe(false);
  });

  it("emits vram_exceeds_known_hardware for impossibly large models", () => {
    // 5T dense full FT ≈ 70TB VRAM — no real cluster fits this.
    const r = computeFtCost(denseSpec(5000), training, "full");
    expect(r.warnings.some(w => w.code === "vram_exceeds_known_hardware")).toBe(true);
  });

  it("emits pretrain_scale_workload at >10 tokens per param", () => {
    // 7B params × 100 tokens/param = 700B tokens → pretraining territory
    const r = computeFtCost(denseSpec(7), {
      ...training, num_examples: 100_000_000, tokens_per_example: 1000, epochs: 7,
    }, "full");
    expect(r.warnings.some(w => w.code === "pretrain_scale_workload")).toBe(true);
    const w = r.warnings.find(w => w.code === "pretrain_scale_workload")!;
    expect(w.severity).toBe("blocker");
  });

  it("emits trivial_workload when GPU cost rounds under $1", () => {
    const r = computeFtCost(denseSpec(0.1), {
      ...training, num_examples: 10, tokens_per_example: 100, epochs: 1,
    }, "qlora");
    expect(r.warnings.some(w => w.code === "trivial_workload")).toBe(true);
  });

  it("zero-data workload does NOT trigger trivial_workload (it's a clean zero)", () => {
    const r = computeFtCost(denseSpec(70), {
      ...training, num_examples: 0, tokens_per_example: 0, epochs: 0,
    }, "qlora");
    expect(r.warnings.some(w => w.code === "trivial_workload")).toBe(false);
  });

  it("back-compat computeFtCapex still strips stages AND warnings", () => {
    const r = computeFtCapex(8, { ...training, method: "lora", cluster_overhead: 0.5 });
    expect("stages" in r).toBe(false);
    expect("warnings" in r).toBe(false);
    // But the clamp still happens — visible via the returned cluster_overhead.
    expect(r.cluster_overhead).toBe(1.0);
  });
});
