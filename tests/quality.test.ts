import { describe, it, expect } from "vitest";

import { recommendTiers } from "../src/engine";
import type { Pricing, KnownModel } from "../src/engine";
import { tierFor, tierFromSize } from "../src/qualityTiers";

const pricing: Pricing = {
  last_updated: "2026-05-26",
  gpu_last_updated: "2026-05-26",
  gpus: [
    { name: "L4 24GB", vram_gb: 24, modal_per_hr: 0.8, lambda_per_hr: 0.7, runpod_per_hr: 0.55 },
    { name: "L40S 48GB", vram_gb: 48, modal_per_hr: 1.95, lambda_per_hr: 1.8, runpod_per_hr: 1.5 },
    { name: "A100 80GB", vram_gb: 80, modal_per_hr: 3.1, lambda_per_hr: 2.5, runpod_per_hr: 2.0 },
    { name: "H100 80GB", vram_gb: 80, modal_per_hr: 4.2, lambda_per_hr: 3.5, runpod_per_hr: 2.7 },
    { name: "2xH100 160GB", vram_gb: 160, modal_per_hr: 8.4, lambda_per_hr: 7.0, runpod_per_hr: 5.4 },
    { name: "4xH100 320GB", vram_gb: 320, modal_per_hr: 16.8, lambda_per_hr: 14.0, runpod_per_hr: 10.8 },
    { name: "8xH100 640GB", vram_gb: 640, modal_per_hr: 33.6, lambda_per_hr: 28.0, runpod_per_hr: 21.6 },
  ],
  apis: {
    // Frontier tier
    openai_o3: { label: "OpenAI o3", input_per_1m: 60, output_per_1m: 240 },
    // Strong tier
    openai_gpt4o: { label: "OpenAI GPT-4o", input_per_1m: 2.5, output_per_1m: 10 },
    // Mid tier
    openai_gpt4o_mini: { label: "OpenAI GPT-4o-mini", input_per_1m: 0.15, output_per_1m: 0.6 },
    // No tier match in our curated list
    obscure_model: { label: "Obscure ModelLab UnknownX 99B", input_per_1m: 1.0, output_per_1m: 3.0 },
  },
};

const knownModels: KnownModel[] = [
  { name: "Llama 3.2 1B Instruct", params_b: 1.0, arch: "dense" },
  { name: "Llama 3.2 3B Instruct", params_b: 3.2, arch: "dense" },
  { name: "Qwen2.5 7B Instruct", params_b: 7.0, arch: "dense" },
  { name: "Llama 3.1 8B Instruct", params_b: 8.0, arch: "dense" },
  { name: "Gemma 2 9B Instruct", params_b: 9.0, arch: "dense" },
  { name: "Qwen2.5 14B Instruct", params_b: 14.0, arch: "dense" },
  { name: "Qwen2.5 32B Instruct", params_b: 32.0, arch: "dense" },
  { name: "Llama 3.1 70B Instruct", params_b: 70.0, arch: "dense" },
  { name: "Qwen2.5 72B Instruct", params_b: 72.0, arch: "dense" },
  { name: "Mixtral 8x22B", params_b: 140.0, arch: "moe", active_b: 39.0 },
  { name: "Qwen3 235B A22B", params_b: 235.0, arch: "moe", active_b: 22.0 },
  { name: "Llama 3.1 405B Instruct", params_b: 405.0, arch: "dense" },
  { name: "Qwen3 Coder 480B A35B", params_b: 480.0, arch: "moe", active_b: 35.0 },
  { name: "DeepSeek V3", params_b: 671.0, arch: "moe", active_b: 37.0 },
];

function makeArgs(overrides: Partial<Parameters<typeof recommendTiers>[0]>) {
  return {
    pricing,
    queries_per_week: 200_000,
    input_tokens: 1500,
    output_tokens: 400,
    api_key: "openai_gpt4o",
    pattern: "uniform" as const,
    vendor: "runpod" as const,
    quant_pref: "fp16" as const,
    min_params_b: 1,
    overhead_gb: 4,
    cold_start_sec: 30,
    ft_cost: 0,
    ft_weeks: 0,
    knownModels,
    ...overrides,
  };
}

describe("qualityTiers — rule lookup", () => {
  it("matches frontier closed-API names", () => {
    expect(tierFor("OpenAI o3")).toBe("frontier");
    expect(tierFor("Anthropic Claude Sonnet 4")).toBe("frontier");
  });

  it("matches strong tier", () => {
    expect(tierFor("OpenAI GPT-4o")).toBe("strong");
    expect(tierFor("Anthropic Claude 3.5 Sonnet")).toBe("strong");
    expect(tierFor("Llama 3.1 70B Instruct")).toBe("strong");
  });

  it("matches mid tier", () => {
    expect(tierFor("OpenAI GPT-4o-mini")).toBe("mid");
    expect(tierFor("Llama 3.1 8B Instruct")).toBe("mid");
  });

  it("matches small tier", () => {
    expect(tierFor("Llama 3.2 3B Instruct")).toBe("small");
    expect(tierFor("Gemma 2 2B Instruct")).toBe("small");
  });

  it("returns null for unknown labels", () => {
    expect(tierFor("Obscure ModelLab UnknownX 99B")).toBeNull();
    expect(tierFor("My Custom FineTune V7")).toBeNull();
  });

  it("size fallback gives sane buckets", () => {
    expect(tierFromSize(1)).toBe("small");
    expect(tierFromSize(8)).toBe("small");
    expect(tierFromSize(32)).toBe("mid");
    expect(tierFromSize(80)).toBe("strong");
  });
});

describe("recommendTiers — comparableQuality", () => {
  it("API model with no curated tier match → no comparable card", () => {
    const r = recommendTiers(makeArgs({ api_key: "obscure_model", queries_per_week: 500_000 }));
    expect(r.api_cost).toBeGreaterThan(0);
    expect(r.tiers.length).toBeGreaterThan(0);
    expect(r.comparableQuality).toBeNull();
  });

  it("custom api_override → no comparable card (no quality signal)", () => {
    const r = recommendTiers(
      makeArgs({
        api_key: "openai_gpt4o", // ignored when override is present
        api_override: { input_per_1m: 1.0, output_per_1m: 3.0 },
        queries_per_week: 500_000,
      })
    );
    expect(r.comparableQuality).toBeNull();
  });

  it("mid-tier API → returns the cheapest mid-or-better open-weight that fits", () => {
    // GPT-4o-mini is very cheap; budget is tight, but mid-tier 8B/14B should fit.
    const r = recommendTiers(
      makeArgs({ api_key: "openai_gpt4o_mini", queries_per_week: 5_000_000 })
    );
    expect(r.api_cost).toBeGreaterThan(0);
    if (r.comparableQuality) {
      const cq = r.comparableQuality;
      expect(cq.apiTier).toBe("mid");
      // floor is mid (exact) or small (one-step fallback)
      expect(["mid", "small"]).toContain(cq.floor);
      // If floor==mid, picked model must be at least mid; if floor==small,
      // small is allowed.
      if (cq.floor === "mid") {
        expect(["frontier", "strong", "mid"]).toContain(cq.modelTier);
      } else {
        expect(["frontier", "strong", "mid", "small"]).toContain(cq.modelTier);
      }
      // And must actually fit under budget.
      const w = r.comparableQuality.tier.weekly_cost_with_ft ?? r.comparableQuality.tier.weekly_cost;
      expect(w).toBeLessThanOrEqual(r.api_cost);
    } else {
      // Acceptable: budget too tight for any mid-or-better self-host fit.
      expect(r.tiers.every((t) => (t.quality_tier ?? "small") === "small")).toBe(true);
    }
  });

  it("strong-tier API with healthy budget → returns a strong-or-frontier pick distinct from largest", () => {
    const r = recommendTiers(makeArgs({ api_key: "openai_gpt4o", queries_per_week: 1_000_000 }));
    expect(r.largest).not.toBeNull();
    expect(r.comparableQuality).not.toBeNull();
    const cq = r.comparableQuality!;
    expect(cq.apiTier).toBe("strong");
    // floor was either strong or mid (one-step fallback)
    expect(["strong", "mid"]).toContain(cq.floor);
    // The picked open-weight tier respects the floor.
    expect(["frontier", "strong", "mid"]).toContain(cq.modelTier);
  });

  it("frontier API + tight budget → may return null", () => {
    // o3 is expensive per token but tiny query volume keeps API budget low.
    const r = recommendTiers(
      makeArgs({
        api_key: "openai_o3",
        queries_per_week: 500,
        input_tokens: 200,
        output_tokens: 100,
      })
    );
    // Either nothing fits at all (engine returns null largest), or a comparable
    // card may still appear — but if it does, its tier must be a real pick.
    if (!r.largest) {
      expect(r.comparableQuality).toBeNull();
    } else if (r.comparableQuality) {
      // Should be at frontier tier or one step below (strong).
      expect(["frontier", "strong"]).toContain(r.comparableQuality.modelTier);
    }
  });

  it("comparableQuality.tier always has a quality_tier", () => {
    const r = recommendTiers(makeArgs({ api_key: "openai_gpt4o", queries_per_week: 2_000_000 }));
    if (r.comparableQuality) {
      expect(r.comparableQuality.tier.quality_tier).toBeDefined();
      expect(r.comparableQuality.tier.quality_tier).not.toBeNull();
    }
  });

  it("ConfigResult.quality_tier is populated for every candidate", () => {
    const r = recommendTiers(makeArgs({ api_key: "openai_gpt4o", queries_per_week: 1_000_000 }));
    for (const c of r.all_candidates) {
      // size fallback guarantees a tier even without a name match
      expect(c.quality_tier).toBeTruthy();
    }
  });
});
