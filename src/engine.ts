import pricingJson from "./pricing.json";
import knownModelsJson from "./knownModels.json";

// =============================================================================
// TYPES
// =============================================================================

export type Vendor = "runpod" | "lambda" | "modal";
export type Quant = "fp16" | "int8" | "int4";
export type Arch = "dense" | "moe";
export type Pattern =
  | "uniform"
  | "business"
  | "bursty"
  | "cold_per_query"
  | "always_warm";

export interface GpuRow {
  name: string;
  vram_gb: number;
  modal_per_hr: number;
  lambda_per_hr: number;
  runpod_per_hr: number;
}

export interface ApiRow {
  label: string;
  input_per_1m: number;
  output_per_1m: number;
}

export interface Pricing {
  last_updated: string;
  gpus: GpuRow[];
  apis: Record<string, ApiRow>;
  gpu_last_updated?: string;
}

export const PRICING: Pricing = pricingJson as Pricing;

export const QUANT_BYTES: Record<Quant, number> = { fp16: 2, int8: 1, int4: 0.5 };
export const QUANT_LABEL: Record<Quant, string> = {
  fp16: "FP16 (full precision)",
  int8: "INT8",
  int4: "INT4 (most compressed)",
};

export interface KnownModel {
  params_b: number;
  arch: Arch;
  active_b?: number;
  name: string;
  source?: string;
  last_seen?: string;
}

// Bundled known models loaded from JSON at build time.
// The live useLiveData hook can override with fresh models.dev data.
export const KNOWN_MODELS: KnownModel[] = (knownModelsJson as any[]).map((m: any) => ({
  params_b: m.params_b,
  active_b: m.active_b ?? undefined,
  name: m.name,
  arch: m.arch as Arch,
  source: m.source,
  last_seen: m.last_seen,
}));

/** Given a params count + arch, find the nearest named model in the supplied list. */
export function nearestModelInList(params_b: number, arch: Arch, list: KnownModel[]): KnownModel | null {
  const candidates = list.filter((m) => m.arch === arch);
  if (candidates.length === 0) return null;
  return candidates.reduce((best, m) =>
    Math.abs(m.params_b - params_b) < Math.abs(best.params_b - params_b) ? m : best
  );
}

// =============================================================================
// TRAFFIC PATTERNS — fraction of weekly traffic per hour-of-week (168 hours)
// =============================================================================

export function trafficShape(pattern: Pattern): number[] {
  const shape = new Array<number>(168).fill(0);
  if (pattern === "uniform" || pattern === "always_warm" || pattern === "cold_per_query") {
    return shape.fill(1 / 168);
  }
  if (pattern === "business") {
    for (let day = 0; day < 7; day++) {
      for (let hour = 0; hour < 24; hour++) {
        const idx = day * 24 + hour;
        if (day < 5 && hour >= 9 && hour < 18) shape[idx] = 1;
      }
    }
    const total = shape.reduce((a, b) => a + b, 0);
    return shape.map((v) => v / total);
  }
  // bursty: 80/20-ish — handful of weekday peak hours dominate
  for (let day = 0; day < 7; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const idx = day * 24 + hour;
      if (day < 5 && (hour === 10 || hour === 14 || hour === 16)) shape[idx] = 5;
      else if (day < 5 && hour >= 9 && hour < 18) shape[idx] = 0.5;
      else shape[idx] = 0.05;
    }
  }
  const total = shape.reduce((a, b) => a + b, 0);
  return shape.map((v) => v / total);
}

// =============================================================================
// CORE ENGINE
// =============================================================================

export function vramRequired(params_b: number, quant: Quant, overhead_gb = 4): number {
  return params_b * QUANT_BYTES[quant] + overhead_gb;
}

/**
 * Heuristic VRAM overhead (KV cache + activations buffer) that scales with
 * model size. Calibrated for ~4-8K context, 1-4 concurrent requests.
 * For MoE, callers should pass `active_params_b` -- KV cache scales with
 * active params, not total.
 */
export function scaledOverhead(active_params_b: number): number {
  if (active_params_b <= 13) return 4;
  if (active_params_b <= 34) return 8;
  if (active_params_b <= 80) return 12;
  if (active_params_b <= 200) return 18;
  return 24;
}

// Conservative throughput estimate: H100-class hardware does ~120 tok/s
// per 8B active params on an 80GB unit; scales linearly with GPU count.
export function throughputTokensPerSec(active_params_b: number, vram_gb: number): number {
  const gpu_units = Math.max(1, vram_gb / 80);
  const base_per_unit = 960 / Math.max(active_params_b, 1);
  return base_per_unit * gpu_units;
}

export function pickCheapestGpu(
  pricing: Pricing,
  vram_needed: number,
  vendor: Vendor
): GpuRow | null {
  const eligible = pricing.gpus.filter((g) => g.vram_gb >= vram_needed);
  if (eligible.length === 0) return null;
  const priceKey = `${vendor}_per_hr` as const;
  return eligible.reduce(
    (min, g) => (g[priceKey] < min[priceKey] ? g : min),
    eligible[0]
  );
}

export function weeklyApiCost(
  pricing: Pricing,
  queries_per_week: number,
  input_tokens: number,
  output_tokens: number,
  apiKey: string,
  override?: { input_per_1m: number; output_per_1m: number }
): number {
  const api = pricing.apis[apiKey];
  const rates = override ?? api;
  const cost_per_query =
    (input_tokens * rates.input_per_1m + output_tokens * rates.output_per_1m) /
    1_000_000;
  return queries_per_week * cost_per_query;
}

export interface EvalArgs {
  pricing: Pricing;
  params_b: number;
  active_params_b: number;
  arch: Arch;
  quant: Quant;
  queries_per_week: number;
  output_tokens: number;
  pattern: Pattern;
  vendor: Vendor;
  cold_start_sec?: number;
  overhead_gb?: number;
  knownModels?: KnownModel[];
}

export interface BillingOption {
  mode: "always_on" | "hourly" | "per_second";
  label: string;
  weekly_cost: number;
  billed_hours: number;
}

export interface ConfigResult {
  params_b: number;
  active_params_b: number;
  arch: Arch;
  quant: Quant;
  gpu: string;
  gpu_price_per_hr: number;
  vram_needed_gb: number;
  vram_available_gb: number;
  /** Overhead actually used in vram math (max of user input + size-scaled floor). */
  effective_overhead_gb: number;
  billing_mode: BillingOption["mode"];
  billing_label: string;
  billed_hours: number;
  weekly_cost: number;
  all_billing_options: BillingOption[];
  nearest_named: KnownModel | null;
  tps: number;
  cold_starts: number;
  /** number of GPU replicas needed to keep up with peak hour, given throughput */
  replicas_needed: number;
  /** true if a single GPU can't serve the peak hour without queuing */
  saturated: boolean;
  /** added by recommendTiers() */
  ft_weekly?: number;
  weekly_cost_with_ft?: number;
}

export function evaluateConfig(args: EvalArgs): ConfigResult | null {
  const {
    pricing,
    params_b,
    active_params_b,
    arch,
    quant,
    queries_per_week,
    output_tokens,
    pattern,
    vendor,
    cold_start_sec = 30,
    overhead_gb = 4,
  } = args;

  // Effective overhead: scales with active params, with the user-supplied value as a floor.
  // Lets users with measured KV-cache numbers raise the bar; never silently lowers it.
  const effective_overhead = Math.max(overhead_gb, scaledOverhead(active_params_b));
  const vram_needed = vramRequired(params_b, quant, effective_overhead);
  const gpu = pickCheapestGpu(pricing, vram_needed, vendor);
  if (!gpu) return null;

  const price_per_hr = gpu[`${vendor}_per_hr`];
  const shape = trafficShape(pattern);
  const tps = throughputTokensPerSec(active_params_b, gpu.vram_gb);

  const queries_per_hour = shape.map((f) => f * queries_per_week);
  const peak_qph = Math.max(...queries_per_hour);
  const peak_seconds_needed = (peak_qph * output_tokens) / tps;
  const replicas_needed = Math.max(1, Math.ceil(peak_seconds_needed / 3600));
  const saturated = replicas_needed > 1;

  // Billing modes — all costs scaled by replicas_needed so saturation isn't silent
  const always_on_cost = price_per_hr * 168 * replicas_needed;

  let warm_hours = 0;
  for (const qph of queries_per_hour) {
    if (qph > 0.1) warm_hours += 1;
  }
  const hourly_cost = price_per_hr * warm_hours * replicas_needed;

  let serve_seconds = 0;
  let cold_starts = 0;

  if (pattern === "cold_per_query") {
    cold_starts = queries_per_week;
    const seconds_serving = (queries_per_week * output_tokens) / tps;
    serve_seconds = seconds_serving + queries_per_week * cold_start_sec;
  } else {
    let was_idle = true;
    for (const qph of queries_per_hour) {
      if (qph > 0.1) {
        const seconds_serving = (qph * output_tokens) / tps;
        // cap per replica-hour at 3600s; replicas_needed already factored in
        serve_seconds += Math.min(seconds_serving, 3600);
        if (was_idle) {
          cold_starts += 1;
          serve_seconds += cold_start_sec;
        }
        was_idle = false;
      } else {
        was_idle = true;
      }
    }
  }
  const per_second_cost = price_per_hr * (serve_seconds / 3600) * replicas_needed;

  let billing_options: BillingOption[];
  if (pattern === "always_warm") {
    billing_options = [
      {
        mode: "always_on",
        label: "Always-on (forced — no scale-down)",
        weekly_cost: always_on_cost,
        billed_hours: 168 * replicas_needed,
      },
    ];
  } else {
    billing_options = [
      { mode: "always_on", label: "Always-on", weekly_cost: always_on_cost, billed_hours: 168 * replicas_needed },
      { mode: "hourly", label: "Hourly warm during active hours", weekly_cost: hourly_cost, billed_hours: warm_hours * replicas_needed },
      { mode: "per_second", label: "Scale-to-zero (pay only while serving)", weekly_cost: per_second_cost, billed_hours: (serve_seconds / 3600) * replicas_needed },
    ];
  }

  const cheapest = billing_options.reduce((min, b) =>
    b.weekly_cost < min.weekly_cost ? b : min
  );
  const named = nearestModelInList(params_b, arch, args.knownModels ?? KNOWN_MODELS);

  return {
    params_b,
    active_params_b,
    arch,
    quant,
    gpu: gpu.name,
    gpu_price_per_hr: price_per_hr,
    vram_needed_gb: vram_needed,
    vram_available_gb: gpu.vram_gb,
    effective_overhead_gb: effective_overhead,
    billing_mode: cheapest.mode,
    billing_label: cheapest.label,
    billed_hours: cheapest.billed_hours,
    weekly_cost: cheapest.weekly_cost,
    all_billing_options: billing_options,
    nearest_named: named,
    tps,
    cold_starts,
    replicas_needed,
    saturated,
  };
}

export interface RecommendArgs {
  pricing: Pricing;
  queries_per_week: number;
  input_tokens: number;
  output_tokens: number;
  api_key: string;
  api_override?: { input_per_1m: number; output_per_1m: number };
  pattern: Pattern;
  vendor: Vendor;
  quant_pref: Quant;
  min_params_b: number;
  overhead_gb: number;
  cold_start_sec: number;
  ft_cost: number;
  ft_weeks: number;
  knownModels?: KnownModel[];
}

export interface RecommendResult {
  api_cost: number;
  tiers: ConfigResult[];
  cheapest: ConfigResult | null;
  largest: ConfigResult | null;
  all_candidates: ConfigResult[];
}

/** Derive DENSE_SIZES and MOE_SIZES from the model list dynamically.
 *  Dedupe by exact params_b (3-decimal), no rounding, no zero bucket.
 */
function deriveSizeBuckets(models: KnownModel[]) {
  const denseSet = new Set<number>();
  const moeMap = new Map<number, { total: number; active: number }>();

  for (const m of models) {
    if (m.params_b <= 0) continue;
    if (m.arch === "moe") {
      if (!moeMap.has(m.params_b)) {
        moeMap.set(m.params_b, { total: m.params_b, active: m.active_b ?? m.params_b });
      }
    } else {
      denseSet.add(m.params_b);
    }
  }

  const dense = [...denseSet].sort((a, b) => a - b);
  const moe = [...moeMap.values()].sort((a, b) => a.total - b.total);
  return { dense, moe };
}

export function recommendTiers(args: RecommendArgs): RecommendResult {
  const {
    pricing,
    queries_per_week,
    input_tokens,
    output_tokens,
    api_key,
    api_override,
    pattern,
    vendor,
    quant_pref,
    min_params_b,
    overhead_gb,
    cold_start_sec,
    ft_cost,
    ft_weeks,
    knownModels,
  } = args;

  const models = knownModels ?? KNOWN_MODELS;
  const { dense: DENSE_SIZES, moe: MOE_SIZES } = deriveSizeBuckets(models);

  const api_cost = weeklyApiCost(
    pricing,
    queries_per_week,
    input_tokens,
    output_tokens,
    api_key,
    api_override
  );

  const candidates: ConfigResult[] = [];

  for (const p of DENSE_SIZES) {
    if (p < min_params_b) continue;
    const c = evaluateConfig({
      pricing,
      params_b: p,
      active_params_b: p,
      arch: "dense",
      quant: quant_pref,
      queries_per_week,
      output_tokens,
      pattern,
      vendor,
      cold_start_sec,
      overhead_gb,
      knownModels: models,
    });
    if (c) candidates.push(c);
  }
  for (const m of MOE_SIZES) {
    if (m.total < min_params_b) continue;
    const c = evaluateConfig({
      pricing,
      params_b: m.total,
      active_params_b: m.active,
      arch: "moe",
      quant: quant_pref,
      queries_per_week,
      output_tokens,
      pattern,
      vendor,
      cold_start_sec,
      overhead_gb,
      knownModels: models,
    });
    if (c) candidates.push(c);
  }

  const ft_weekly = ft_weeks > 0 ? ft_cost / ft_weeks : 0;
  for (const c of candidates) {
    c.ft_weekly = ft_weekly;
    c.weekly_cost_with_ft = c.weekly_cost + ft_weekly;
  }

  const affordable = candidates.filter(
    (c) => (c.weekly_cost_with_ft ?? c.weekly_cost) <= api_cost
  );
  affordable.sort((a, b) => b.params_b - a.params_b);

  const cheapest =
    affordable.length === 0
      ? null
      : affordable.reduce((min, c) =>
          (c.weekly_cost_with_ft ?? c.weekly_cost) <
          (min.weekly_cost_with_ft ?? min.weekly_cost)
            ? c
            : min
        );

  return {
    api_cost,
    tiers: affordable,
    largest: affordable[0] ?? null,
    cheapest,
    all_candidates: candidates,
  };
}
