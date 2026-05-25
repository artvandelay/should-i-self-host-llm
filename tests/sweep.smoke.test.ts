import { describe, it, expect } from "vitest";

import { recommendTiers } from "../src/engine";
import type { Pricing, KnownModel } from "../src/engine";

const pricing: Pricing = {
  last_updated: "2026-05-25",
  gpu_last_updated: "2026-05-25",
  gpus: [
    { name: "L4 24GB", vram_gb: 24, modal_per_hr: 0.8, lambda_per_hr: 0.7, runpod_per_hr: 0.55 },
    { name: "L40S 48GB", vram_gb: 48, modal_per_hr: 1.95, lambda_per_hr: 1.8, runpod_per_hr: 1.5 },
    { name: "A100 40GB", vram_gb: 40, modal_per_hr: 2.1, lambda_per_hr: 1.85, runpod_per_hr: 1.6 },
    { name: "A100 80GB", vram_gb: 80, modal_per_hr: 3.1, lambda_per_hr: 2.5, runpod_per_hr: 2.0 },
    { name: "H100 80GB", vram_gb: 80, modal_per_hr: 4.2, lambda_per_hr: 3.5, runpod_per_hr: 2.7 },
    { name: "H200 140GB", vram_gb: 140, modal_per_hr: 6.5, lambda_per_hr: 5.5, runpod_per_hr: 4.2 },
    { name: "GH200 288GB", vram_gb: 288, modal_per_hr: 12.0, lambda_per_hr: 10.0, runpod_per_hr: 8.5 },
  ],
  apis: {
    openai_gpt4o: {
      label: "OpenAI GPT-4o",
      input_per_1m: 2500,
      output_per_1m: 10000,
    },
  },
};

const knownModels: KnownModel[] = [
  { name: "Llama 3.2 1B Instruct", params_b: 1.0, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Llama 3.2 3B Instruct", params_b: 3.2, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Llama 3.1 8B Instruct", params_b: 8.0, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Llama 3.1 70B Instruct", params_b: 70.0, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Llama 3.1 405B Instruct", params_b: 405.0, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Mixtral 8x22B", params_b: 140.0, arch: "moe", active_b: 39.0, source: "seed", last_seen: "seed" },
  // Carries an ELO so we can assert it propagates through to ConfigResult
  { name: "DeepSeek V3", params_b: 671.0, arch: "moe", active_b: 37.0, source: "seed", last_seen: "seed", elo: 1380, eloRank: 7 },
  { name: "Gemma 2 2B Instruct", params_b: 2.0, arch: "dense", active_b: undefined, source: "seed", last_seen: "seed" },
  { name: "Qwen3 30B A3B", params_b: 30.0, arch: "moe", active_b: 3.0, source: "seed", last_seen: "seed", elo: 1310 },
  { name: "Qwen3 235B A22B", params_b: 235.0, arch: "moe", active_b: 22.0, source: "seed", last_seen: "seed", elo: 1360 },
  { name: "Qwen3 Coder 480B A35B", params_b: 480.0, arch: "moe", active_b: 35.0, source: "seed", last_seen: "seed" },
];

function makeArgs(overrides: Partial<Parameters<typeof recommendTiers>[0]>) {
  return {
    pricing,
    queries_per_week: 10_000,
    input_tokens: 800,
    output_tokens: 300,
    api_key: "openai_gpt4o",
    pattern: "uniform" as const,
    vendor: "lambda" as const,
    quant_pref: "int4" as const,
    min_params_b: 1,
    overhead_gb: 2,
    cold_start_sec: 5,
    ft_cost: 1200,
    ft_weeks: 8,
    knownModels,
    ...overrides,
  };
}

describe("sweep smoke", () => {
  it("-> high-traffic clone LinkIn => nothing above GPT-4o", () => {
    const r = recommendTiers(
      makeArgs({ queries_per_week: 1_200_000, input_tokens: 4000, output_tokens: 1200 })
    );
    expect(r.api_cost).toBeGreaterThan(0);
    expect(r.tiers.length).toBeGreaterThan(0);
    for (const t of r.tiers) {
      expect(t.weekly_cost_with_ft ?? t.weekly_cost).toBeLessThanOrEqual(r.api_cost);
    }
    expect(r.largest).not.toBeNull();
  });

  it("-> low-traffic startup SaaS => some tier saves >80%, no models below knownModels floor", () => {
    const r = recommendTiers(makeArgs({ queries_per_week: 5_000, ft_weeks: 6, ft_cost: 900 }));
    expect(r.api_cost).toBeGreaterThan(0);
    // At least one affordable tier should clear 80% savings.
    const bestSavingsPct = Math.max(
      ...r.tiers.map((t) => {
        const w = t.weekly_cost_with_ft ?? t.weekly_cost;
        return ((r.api_cost - w) / r.api_cost) * 100;
      })
    );
    expect(bestSavingsPct).toBeGreaterThan(80);

    // Density: no buckets below 1B because min_params_b=1 and knownModels starts at 1B
    for (const t of r.all_candidates) {
      expect(t.params_b).toBeGreaterThanOrEqual(1);
    }
  });

  it("-> agency post-FT => some tier under 300/wk", () => {
    const r = recommendTiers(
      makeArgs({ queries_per_week: 40_000, input_tokens: 1200, output_tokens: 800, ft_cost: 0, ft_weeks: 0 })
    );
    expect(r.api_cost).toBeGreaterThan(0);
    const minCost = Math.min(
      ...r.tiers.map((t) => t.weekly_cost_with_ft ?? t.weekly_cost)
    );
    expect(minCost).toBeLessThanOrEqual(300);
  });

  it("-> gradedTiers: distinct from largest, meet savings floor, ordered by descending floor", () => {
    const r = recommendTiers(makeArgs({ queries_per_week: 200_000, input_tokens: 1500, output_tokens: 500 }));
    expect(r.largest).not.toBeNull();
    const largest = r.largest!;
    const largestKey = `${largest.arch}-${largest.params_b}`;

    // Ordered by descending savings floor.
    for (let i = 1; i < r.gradedTiers.length; i++) {
      expect(r.gradedTiers[i - 1].savingsFloor).toBeGreaterThan(r.gradedTiers[i].savingsFloor);
    }

    const seen = new Set<string>();
    for (const g of r.gradedTiers) {
      const key = `${g.tier.arch}-${g.tier.params_b}`;
      // Distinct from largest.
      expect(key).not.toBe(largestKey);
      // Distinct from other graded picks.
      expect(seen.has(key)).toBe(false);
      seen.add(key);
      // Savings actually meets the band's floor.
      const w = g.tier.weekly_cost_with_ft ?? g.tier.weekly_cost;
      const savings = (r.api_cost - w) / r.api_cost;
      expect(savings).toBeGreaterThanOrEqual(g.savingsFloor);
    }
  });

  it("-> ELO propagates from KnownModel to ConfigResult.elo / eloRank", () => {
    const r = recommendTiers(makeArgs({ queries_per_week: 200_000, input_tokens: 1500, output_tokens: 500 }));
    // Qwen3 235B A22B carries elo 1360 in our seed and fits on a GH200 at int4.
    const qwenCandidates = r.all_candidates.filter((c) => c.arch === "moe" && c.params_b === 235);
    expect(qwenCandidates.length).toBeGreaterThan(0);
    expect(qwenCandidates[0].elo).toBe(1360);

    // Models without elo in the seed produce undefined; that's expected.
    const llama70 = r.all_candidates.find((c) => c.arch === "dense" && c.params_b === 70);
    expect(llama70?.elo).toBeUndefined();
  });

  it("-> min_elo filter drops candidates below the floor and keeps unscored", () => {
    // Floor of 1350 should drop Qwen3 30B A3B (elo 1310) but keep Qwen3 235B (1360) and DeepSeek V3 (1380).
    // Models WITHOUT an elo (Llama, Mixtral, Coder 480B) must still be kept (no data != bad).
    const r = recommendTiers(
      makeArgs({
        queries_per_week: 1_200_000,
        input_tokens: 4000,
        output_tokens: 1200,
        min_elo: 1350,
      })
    );
    for (const t of r.tiers) {
      // Either no elo, or >= 1350.
      if (t.elo != null) expect(t.elo).toBeGreaterThanOrEqual(1350);
    }
    // Without the filter, the Qwen3 30B A3B tier should be present somewhere; with the filter it must be absent.
    const hasFilteredOut = r.tiers.some(
      (t) => t.arch === "moe" && t.params_b === 30 && t.elo === 1310
    );
    expect(hasFilteredOut).toBe(false);
  });

  it("-> reject patterns: knownModels must NOT contain banned model families", () => {
    const banned = [
      /\b(embed|embedding|retriev|reranker|rerank|nemoretriever)\b/i,
      /\b(prompt[\s-]?guard|moderation|toxic)\b/i,
      /\b(paddle[\s-]?ocr|ocr|vl|cosmos[\s-]?transfer)\b/i,
      /\b(whisper|asr|tts|speech[\s-]?to[\s-]?text|text[\s-]?to[\s-]?speech)\b/i,
      /\b(esm[12]|protein|biomed)\b/i,
      /\b(codestral|devstral|prover[\s-]?v?\d)\b/i,
      /\b(flux|sdxl|stable[\s-]?diffusion|imagen|wan[\s-]?gen)\b/i,
    ];
    for (const m of knownModels) {
      for (const re of banned) {
        expect(re.test(m.name), `Failed regex ${re.source} on "${m.name}"`).toBe(false);
      }
    }
  });
});
