/**
 * One-off scenario printer for spot-checking recommendations.
 * Run: npx tsx scripts/print-scenarios.ts
 */

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
    openai_gpt4o: { label: "OpenAI GPT-4o", input_per_1m: 2500, output_per_1m: 10000 },
  },
};

import knownModelsJson from "../src/knownModels.json";
const knownModels = knownModelsJson as KnownModel[];

function fmt$(n: number) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1000) return `$${(n / 1000).toFixed(2)}k`;
  return `$${n.toFixed(2)}`;
}

function scenario(name: string, args: Parameters<typeof recommendTiers>[0]) {
  const r = recommendTiers(args);
  console.log(`\n=== ${name} ===`);
  console.log(`API cost: ${fmt$(r.api_cost)}/wk`);
  console.log(`Candidates: ${r.all_candidates.length}`);
  console.log(`Cheapest: ${r.cheapest?.params_b ?? "N/A"}B ${r.cheapest?.arch}  ${fmt$(r.cheapest?.weekly_cost_with_ft ?? r.cheapest?.weekly_cost ?? 0)}/wk`);
  console.log(`Largest : ${r.largest?.params_b ?? "N/A"}B ${r.largest?.arch}  ${fmt$(r.largest?.weekly_cost_with_ft ?? r.largest?.weekly_cost ?? 0)}/wk`);
  // Show first 10 candidates for 
  console.log("Top candidates by size:");
  const sorted = [...r.all_candidates].sort((a, b) => a.params_b - b.params_b);
  for (const c of sorted.slice(0, 10)) {
    const cost = c.weekly_cost_with_ft ?? c.weekly_cost;
    console.log(`  ${String(c.params_b).padStart(4)}B ${c.arch.padEnd(5)}  ${fmt$(cost).padStart(12)}/wk  ${c.gpu.padEnd(12)}  ${c.billing_label}`);
  }
}

const base = {
  pricing,
  api_key: "openai_gpt4o" as const,
  pattern: "uniform" as const,
  vendor: "lambda" as const,
  quant_pref: "int4" as const,
  min_params_b: 1,
  overhead_gb: 2,
  cold_start_sec: 5,
  ft_cost: 1200,
  ft_weeks: 8,
  knownModels,
  input_tokens: 800,
  output_tokens: 300,
};

scenario("high-traffic SaaS", {
  ...base,
  queries_per_week: 1_200_000,
  input_tokens: 4000,
  output_tokens: 1200,
});

scenario("low-traffic startup", {
  ...base,
  queries_per_week: 5_000,
  ft_weeks: 6,
  ft_cost: 900,
});

scenario("agency post-FT", {
  ...base,
  queries_per_week: 40_000,
  input_tokens: 1200,
  output_tokens: 800,
  ft_cost: 0,
  ft_weeks: 0,
});

scenario("tiny hobby", {
  ...base,
  queries_per_week: 500,
  input_tokens: 500,
  output_tokens: 200,
  ft_cost: 0,
  ft_weeks: 0,
});

console.log("\n=== Sanity checks ===");

// No 0B
const all0B = knownModels.filter((m) => m.params_b <= 0);
console.log(`Models with params_b <= 0: ${all0B.length}`);

// No banned families
const banned = /(embed|embedding|retriev|rerank|guard|moderation|toxic|paddleocr|whisper|tts|asr|esm[12]|protein|biomed|codestral|devstral|flux|sdxl|stable-diffusion|imagen|osmosis|structure|e5|gme|nq|cosmos-transfer|cosmos-predict)/i;
const bad = knownModels.filter((m) => banned.test(m.name));
console.log(`Banned-family matches: ${bad.length}` + (bad.length ? ` e.g. "${bad[0].name}"` : ""));

// Smallest
const smallest = [...knownModels].sort((a,b) => a.params_b - b.params_b)[0];
console.log(`Smallest model: ${smallest.params_b}B "${smallest.name}"`);
