/**
 * Stress test the engine with extreme inputs.
 * Run: npx tsx scripts/stress-test.ts
 */
import { recommendTiers } from "../src/engine";
import type { Pricing, KnownModel } from "../src/engine";
import knownModelsJson from "../src/knownModels.json";
import pricingJson from "../src/pricing.json";

const pricing = pricingJson as Pricing;
const knownModels = knownModelsJson as KnownModel[];

const apiKey = Object.keys(pricing.apis)[0];

function fmt$(n: number) {
  if (!Number.isFinite(n)) return "INVALID";
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(2)}k`;
  if (n < 0.01) return `$${n.toExponential(2)}`;
  return `$${n.toFixed(2)}`;
}

function check(name: string, args: Parameters<typeof recommendTiers>[0]) {
  console.log(`\n=== ${name} ===`);
  let r;
  try {
    r = recommendTiers(args);
  } catch (e: any) {
    console.log(`  THROWN: ${e.message}`);
    return;
  }

  const issues: string[] = [];

  // Sanity checks
  if (!Number.isFinite(r.api_cost)) issues.push(`api_cost not finite: ${r.api_cost}`);
  if (r.api_cost < 0) issues.push(`api_cost negative: ${r.api_cost}`);
  if (r.tiers.length > 0) {
    for (const t of r.tiers) {
      const w = t.weekly_cost_with_ft ?? t.weekly_cost;
      if (!Number.isFinite(w)) issues.push(`tier ${t.params_b}B cost not finite: ${w}`);
      if (w < 0) issues.push(`tier ${t.params_b}B cost negative: ${w}`);
      if (r.api_cost > 0 && w > r.api_cost * 1.001) issues.push(`tier ${t.params_b}B cost ${fmt$(w)} > api ${fmt$(r.api_cost)}`);
      if (t.params_b <= 0) issues.push(`tier params_b <= 0: ${t.params_b}`);
      if (t.replicas_needed < 1) issues.push(`tier ${t.params_b}B replicas < 1: ${t.replicas_needed}`);
    }
  }

  // Graded tier sanity
  const seenSizes = new Set<string>();
  let lastFloor = 1;
  for (const g of r.gradedTiers) {
    if (g.savingsFloor > lastFloor) issues.push(`graded tier order broken: ${g.label}`);
    lastFloor = g.savingsFloor;
    const w = g.tier.weekly_cost_with_ft ?? g.tier.weekly_cost;
    const actualSavings = (r.api_cost - w) / r.api_cost;
    if (actualSavings < g.savingsFloor - 0.001) {
      issues.push(`graded tier ${g.label}: actual savings ${(actualSavings * 100).toFixed(1)}% < floor ${(g.savingsFloor * 100).toFixed(0)}%`);
    }
    const key = `${g.tier.arch}-${g.tier.params_b}`;
    if (seenSizes.has(key)) issues.push(`graded tier dedupe failed: ${key}`);
    seenSizes.add(key);
    if (r.largest && g.tier.params_b === r.largest.params_b && g.tier.arch === r.largest.arch) {
      issues.push(`graded tier matches largest: ${key}`);
    }
  }

  console.log(`  api_cost: ${fmt$(r.api_cost)}/wk  candidates: ${r.all_candidates.length}  affordable: ${r.tiers.length}`);
  if (r.largest) {
    const w = r.largest.weekly_cost_with_ft ?? r.largest.weekly_cost;
    const sav = ((r.api_cost - w) / r.api_cost) * 100;
    console.log(`  Largest:  ${r.largest.params_b}B ${r.largest.arch}  ${fmt$(w)}/wk  saves ${sav.toFixed(0)}%`);
  } else {
    console.log("  Largest:  none (no model fits API budget)");
  }
  for (const g of r.gradedTiers) {
    const w = g.tier.weekly_cost_with_ft ?? g.tier.weekly_cost;
    const sav = ((r.api_cost - w) / r.api_cost) * 100;
    console.log(`  ${g.label}: ${g.tier.params_b}B ${g.tier.arch}  ${fmt$(w)}/wk  saves ${sav.toFixed(0)}%`);
  }

  if (issues.length) {
    console.log(`  ISSUES (${issues.length}):`);
    for (const i of issues) console.log(`    - ${i}`);
  } else {
    console.log(`  OK`);
  }
}

const base = {
  pricing,
  api_key: apiKey,
  pattern: "uniform" as const,
  vendor: "lambda" as const,
  quant_pref: "int4" as const,
  min_params_b: 1,
  overhead_gb: 2,
  cold_start_sec: 5,
  ft_cost: 0,
  ft_weeks: 0,
  knownModels,
  input_tokens: 800,
  output_tokens: 300,
};

// EXTREMES
check("zero queries", { ...base, queries_per_week: 0 });
check("1 query/wk", { ...base, queries_per_week: 1 });
check("10 queries/wk", { ...base, queries_per_week: 10 });
check("100 queries/wk", { ...base, queries_per_week: 100 });
check("1000 queries/wk", { ...base, queries_per_week: 1000 });
check("10k queries/wk", { ...base, queries_per_week: 10_000 });
check("1M queries/wk", { ...base, queries_per_week: 1_000_000 });
check("100M queries/wk", { ...base, queries_per_week: 100_000_000 });
check("1B queries/wk (Google scale)", { ...base, queries_per_week: 1_000_000_000 });

console.log("\n--- TOKEN EXTREMES ---");
check("1 input token", { ...base, queries_per_week: 10_000, input_tokens: 1 });
check("128k context", { ...base, queries_per_week: 10_000, input_tokens: 128_000, output_tokens: 4000 });
check("1M context", { ...base, queries_per_week: 10_000, input_tokens: 1_000_000, output_tokens: 100_000 });
check("zero output tokens", { ...base, queries_per_week: 10_000, output_tokens: 0 });

console.log("\n--- MIN_PARAMS / FT EXTREMES ---");
check("min_params=1000B (only mega)", { ...base, queries_per_week: 1_000_000, min_params_b: 1000 });
check("min_params=10000B (impossible)", { ...base, queries_per_week: 1_000_000, min_params_b: 10_000 });
check("FT cost=$10M, 1 week", { ...base, queries_per_week: 1_000_000, ft_cost: 10_000_000, ft_weeks: 1 });
check("FT cost=$10M, 1000 weeks", { ...base, queries_per_week: 1_000_000, ft_cost: 10_000_000, ft_weeks: 1000 });

console.log("\n--- QUANT / VENDOR EXTREMES ---");
check("FP16 (memory-hungry)", { ...base, queries_per_week: 1_000_000, quant_pref: "fp16" as const });
check("INT4 + Modal", { ...base, queries_per_week: 1_000_000, vendor: "modal" as const });
check("INT4 + Runpod", { ...base, queries_per_week: 1_000_000, vendor: "runpod" as const });

console.log("\n--- OVERHEAD EXTREMES ---");
check("overhead=0 GB", { ...base, queries_per_week: 1_000_000, overhead_gb: 0 });
check("overhead=1000 GB", { ...base, queries_per_week: 1_000_000, overhead_gb: 1000 });

console.log("\n--- PATTERN EXTREMES ---");
for (const p of ["uniform", "business", "bursty", "cold_per_query", "always_warm"] as const) {
  check(`pattern=${p}`, { ...base, queries_per_week: 50_000, pattern: p });
}

console.log("\n--- DEGENERATE INPUTS ---");
check("negative queries", { ...base, queries_per_week: -100 });
check("NaN queries", { ...base, queries_per_week: NaN });
check("Infinity queries", { ...base, queries_per_week: Infinity });
check("empty knownModels", { ...base, queries_per_week: 10_000, knownModels: [] });

console.log("\n=== DONE ===");
