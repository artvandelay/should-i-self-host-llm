import pricingJson from "./pricing.json";
import knownModelsJson from "./knownModels.json";
import {
  tierFor,
  tierFromSize,
  tierMinusOne,
  TIER_RANK,
  type QualityTier,
} from "./qualityTiers";
import {
  FT_METHODS,
  FLOPS_PER_TOKEN_PER_PARAM,
  BASELINE_MFU,
  type FtMethod,
} from "./ftMethods";

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
  /**
   * Optional: peak BF16/FP16 dense matmul throughput in TFLOPS, used for
   * fine-tuning cost estimates. Examples:
   *   H100 SXM     989
   *   H200 SXM     989  (same compute as H100; more HBM)
   *   B200         2250
   *   MI300X       1307
   *   A100 80GB    312
   * Omit (or leave 0) for GPUs that should NOT be considered for training.
   */
  bf16_tflops?: number;
  /**
   * Optional: how many of THIS single-GPU unit live in one NVLink/NVSwitch
   * node. Used to set the cluster-overhead boundary between "multi-GPU
   * (intra-node, fast)" and "multi-node (Infiniband, slow)". Default 8.
   */
  gpus_per_node?: number;
  /**
   * Optional: VRAM (GB) of ONE individual accelerator in this row. For a
   * multi-GPU row like "4xH100 320GB" this is 80, not 320. Used to size
   * cluster overhead. Defaults to vram_gb if omitted (correct for single-GPU
   * rows).
   */
  single_gpu_vram_gb?: number;
}

export interface ApiRow {
  label: string;
  input_per_1m: number;
  output_per_1m: number;
  /** LMArena ELO score, when a match exists (see src/eloMatch.ts). */
  elo?: number;
  /** Rank within the LMArena text leaderboard at last refresh. */
  eloRank?: number;
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
  family?: string;
  /** LMArena ELO score, when a match exists (see src/eloMatch.ts). */
  elo?: number;
  /** Rank within the LMArena text leaderboard at last refresh. */
  eloRank?: number;
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

/**
 * Conservative inference throughput estimate. Calibrated against
 * H100-class hardware (~120 tok/s per 8B active params on an 80 GB unit,
 * which corresponds to ~989 TFLOPS BF16 peak). When a `bf16_tflops` value
 * is provided we scale that anchor linearly with the GPU's actual peak
 * compute, so a future B200 (~2250 TFLOPS) or MI300X (~1307 TFLOPS) gets
 * a higher per-unit throughput automatically. Cross-GPU scaling stays
 * linear in raw VRAM ÷ single-unit VRAM — same heuristic as before.
 *
 * For MoE: pass `active_params_b`, not total.
 */
export function throughputTokensPerSec(
  active_params_b: number,
  vram_gb: number,
  bf16_tflops?: number,
  single_gpu_vram_gb = 80
): number {
  const gpu_units = Math.max(1, vram_gb / single_gpu_vram_gb);
  const tflops_scale = bf16_tflops ? bf16_tflops / 989 : 1;
  const base_per_unit = (960 * tflops_scale) / Math.max(active_params_b, 1);
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

// =============================================================================
// COST PROJECTIONS — derive monthly / annual from any weekly cost
// =============================================================================

/** Average weeks per month: 52 / 12 ≈ 4.345. Used so monthly × 12 ≈ annual. */
export const WEEKS_PER_MONTH = 52 / 12;
export const WEEKS_PER_YEAR = 52;

export type CostView = "weekly" | "monthly" | "annual";

export interface CostProjection {
  weekly: number;
  monthly: number;
  annual: number;
}

/** Project a weekly cost into weekly/monthly/annual figures. Additive helper. */
export function projectCost(weekly_cost: number): CostProjection {
  return {
    weekly: weekly_cost,
    monthly: weekly_cost * WEEKS_PER_MONTH,
    annual: weekly_cost * WEEKS_PER_YEAR,
  };
}

/** Scale a weekly cost to the chosen view. */
export function costForView(weekly_cost: number, view: CostView): number {
  if (view === "monthly") return weekly_cost * WEEKS_PER_MONTH;
  if (view === "annual") return weekly_cost * WEEKS_PER_YEAR;
  return weekly_cost;
}

export function costViewSuffix(view: CostView): string {
  if (view === "monthly") return "/mo";
  if (view === "annual") return "/yr";
  return "/wk";
}

export function costViewLabel(view: CostView): string {
  if (view === "monthly") return "Monthly cost";
  if (view === "annual") return "Annual cost";
  return "Weekly cost";
}

/**
 * Break-even analysis: weeks until cumulative API spend exceeds cumulative
 * self-host spend including a one-time setup/migration cost.
 *
 *   cumulative_api(w)      = api_weekly * w
 *   cumulative_selfhost(w) = setup_cost + selfhost_weekly * w
 *
 * Solve for the smallest integer w where cumulative_api >= cumulative_selfhost.
 *   w >= setup_cost / (api_weekly - selfhost_weekly)
 *
 * Returns null if self-hosting is never cheaper than the API at this weekly
 * rate (i.e. selfhost_weekly >= api_weekly) or if the break-even exceeds
 * `cap_weeks` (default 520 = 10 years).
 */
export function breakEvenWeeks(
  api_weekly: number,
  selfhost_weekly: number,
  setup_cost: number,
  cap_weeks = 520
): number | null {
  if (!Number.isFinite(api_weekly) || !Number.isFinite(selfhost_weekly)) return null;
  if (!Number.isFinite(setup_cost) || setup_cost < 0) return null;
  const weekly_savings = api_weekly - selfhost_weekly;
  if (weekly_savings <= 0) return null;
  if (setup_cost === 0) return 0;
  const weeks = Math.ceil(setup_cost / weekly_savings);
  if (weeks > cap_weeks) return null;
  return weeks;
}

export interface FtInputs {
  num_examples: number;
  tokens_per_example: number;
  method: FtMethod;
  epochs: number;
  prep_cost_usd: number;
  /**
   * Experiments/failures multiplier — production FT campaigns require
   * multiple runs (HP sweeps, early-stopped failures, ablations).
   * 1.0 = single successful run (theoretical floor). 2-3 = realistic
   * range for production work. Default: 2.5.
   */
  experiments_multiplier: number;
}

export interface FtCapexResult {
  gpu_hours: number;
  gpu_cost_usd: number;
  total_capex_usd: number;
  method: FtMethod;
  /** GPU cost for a single successful run, before the experiments multiplier. */
  single_run_gpu_cost_usd: number;
  /** Effective multiplier applied (after clamping to >= 1.0). */
  experiments_multiplier: number;
}

export interface CumulativePoint {
  month: number;
  api_cumulative: number;
  selfhost_cumulative: number;
}

export interface CumulativeProjection {
  points: CumulativePoint[];
  crossover_month: number | null;
  horizon_months: number;
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

export function computeFtCapex(params_b: number, inputs: FtInputs): FtCapexResult {
  const num = clampNonNeg(inputs.num_examples);
  const tok = clampNonNeg(inputs.tokens_per_example);
  const epochs = clampNonNeg(inputs.epochs);
  const prep = clampNonNeg(inputs.prep_cost_usd);
  const params = clampNonNeg(params_b) * 1e9;
  const total_tokens = num * tok * epochs;
  const spec = FT_METHODS[inputs.method];
  const full_flops = FLOPS_PER_TOKEN_PER_PARAM * params * total_tokens;
  const method_flops = full_flops * spec.computeMultiplier;
  // Pick the training GPU from the pricing config (tagged with bf16_tflops).
  // Fall back to a generic 989-TFLOPS / $4-hr H100 placeholder if the config
  // has no tagged GPU — keeps the engine working with stale pricing files.
  const ftGpu = pickFtGpu(PRICING);
  const peak_tflops = ftGpu?.bf16_tflops ?? 989;
  const cheapestRate = ftGpu
    ? Math.min(ftGpu.modal_per_hr, ftGpu.lambda_per_hr, ftGpu.runpod_per_hr)
    : 4.0;
  // Effective throughput = peak BF16 × baseline 30% MFU × per-method MFU
  // penalty. QLoRA's mfuPenalty captures the dequantization tax.
  const effective_flops_per_sec =
    peak_tflops * 1e12 * BASELINE_MFU * spec.mfuPenalty;
  const seconds = method_flops / effective_flops_per_sec;
  const hours = seconds / 3600;
  const single_run_gpu_cost = hours * cheapestRate;
  // Experiments multiplier: clamp UP to 1.0 (values below 1 are non-physical;
  // there's no such thing as "less than one run"). Applies to gpu_hours AND
  // gpu_cost (both scale linearly with number of runs), but NOT to prep_cost
  // (data prep / engineering is one-time across the whole campaign).
  const raw_xm = clampNonNeg(inputs.experiments_multiplier, 1);
  const xm = raw_xm < 1 ? 1 : raw_xm;
  const gpu_hours_total = hours * xm;
  const gpu_cost_total = single_run_gpu_cost * xm;
  return {
    gpu_hours: gpu_hours_total,
    gpu_cost_usd: gpu_cost_total,
    total_capex_usd: gpu_cost_total + prep,
    method: inputs.method,
    single_run_gpu_cost_usd: single_run_gpu_cost,
    experiments_multiplier: xm,
  };
}

export function cumulativeProjection(
  api_weekly: number,
  selfhost_weekly: number,
  capex_usd: number,
  horizon_months = 24
): CumulativeProjection {
  const apiW = clampNonNeg(api_weekly);
  const shW = clampNonNeg(selfhost_weekly);
  const cap = clampNonNeg(capex_usd);
  const horizon = Math.max(1, Math.floor(horizon_months));
  const points: CumulativePoint[] = [];
  let crossover: number | null = null;
  for (let m = 0; m <= horizon; m++) {
    const weeks = m * WEEKS_PER_MONTH;
    const api_cum = apiW * weeks;
    const sh_cum = shW * weeks + cap;
    points.push({ month: m, api_cumulative: api_cum, selfhost_cumulative: sh_cum });
    if (crossover === null && m > 0 && sh_cum <= api_cum) crossover = m;
  }
  return { points, crossover_month: crossover, horizon_months: horizon };
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
  /** Coarse quality tier for the named open-weight model (if known). */
  quality_tier?: QualityTier | null;
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
  /** LMArena ELO of `nearest_named`, when known. */
  elo?: number;
  /** Rank within LMArena text leaderboard at last refresh. */
  eloRank?: number;
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
  const tps = throughputTokensPerSec(
    active_params_b,
    gpu.vram_gb,
    gpu.bf16_tflops,
    gpu.single_gpu_vram_gb ?? gpu.vram_gb
  );

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
  const quality_tier = named ? tierFor(named.name) ?? tierFromSize(params_b, named.active_b ?? null) : tierFromSize(params_b, active_params_b);

  return {
    elo: named?.elo,
    eloRank: named?.eloRank,
    params_b,
    active_params_b,
    arch,
    quant,
    quality_tier,
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
  /**
   * Optional quality floor: drop self-host candidates whose nearest known
   * model has ELO below this. Models with no ELO are always kept (we don't
   * silently hide models just because LMArena hasn't ranked them).
   */
  min_elo?: number;
}

export interface GradedTier {
  label: string;
  savingsFloor: number;
  tier: ConfigResult;
}

export interface ComparableQualityPick {
  tier: ConfigResult;
  /** Tier of the API model we're comparing against (best-effort lookup). */
  apiTier: QualityTier | null;
  /** Tier of the picked open-weight model. */
  modelTier: QualityTier;
  /** Floor used to filter — apiTier or one tier below if nothing fits. */
  floor: QualityTier;
}

export interface RecommendResult {
  api_cost: number;
  tiers: ConfigResult[];
  largest: ConfigResult | null;
  gradedTiers: GradedTier[];
  /**
   * Cheapest open-weight model whose curated quality tier matches the
   * selected API model (or one tier below if no exact match fits the budget).
   * Null when we have no quality info for the API or nothing comparable fits.
   */
  comparableQuality: ComparableQualityPick | null;
  all_candidates: ConfigResult[];
}

const SAVINGS_BANDS = [
  { label: "80%+ savings", floor: 0.80 },
  { label: "50%+ savings", floor: 0.50 },
  { label: "20%+ savings", floor: 0.20 },
];

function pickGradedTiers(
  affordable: ConfigResult[],
  api_cost: number,
  largest: ConfigResult | null
): GradedTier[] {
  if (api_cost <= 0 || !largest) return [];
  const picks: GradedTier[] = [];
  const seen = new Set<string>();
  for (const band of SAVINGS_BANDS) {
    const eligible = affordable.filter((c) => {
      const w = c.weekly_cost_with_ft ?? c.weekly_cost;
      return (api_cost - w) / api_cost >= band.floor;
    });
    if (!eligible.length) continue;
    const pick = eligible.reduce((a, b) => (b.params_b > a.params_b ? b : a));
    const key = `${pick.arch}-${pick.params_b}`;
    if (key === `${largest.arch}-${largest.params_b}`) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    picks.push({ label: band.label, savingsFloor: band.floor, tier: pick });
  }
  return picks;
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

/** Clamp a numeric input: reject NaN/Infinity, force non-negative. */
function clampNonNeg(n: number, fallback = 0): number {
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}

export function recommendTiers(args: RecommendArgs): RecommendResult {
  const {
    pricing,
    api_key,
    api_override,
    pattern,
    vendor,
    quant_pref,
    knownModels,
  } = args;

  // Sanitize all numeric inputs against NaN/Infinity/negative.
  const queries_per_week = clampNonNeg(args.queries_per_week);
  const input_tokens = clampNonNeg(args.input_tokens);
  const output_tokens = clampNonNeg(args.output_tokens);
  const min_params_b = clampNonNeg(args.min_params_b);
  const overhead_gb = clampNonNeg(args.overhead_gb);
  const cold_start_sec = clampNonNeg(args.cold_start_sec);
  const ft_cost = clampNonNeg(args.ft_cost);
  const ft_weeks = clampNonNeg(args.ft_weeks);

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

  const min_elo = args.min_elo ?? 0;
  const affordable = candidates.filter((c) => {
    const fits = (c.weekly_cost_with_ft ?? c.weekly_cost) <= api_cost;
    if (!fits) return false;
    // Quality floor: models WITH an ELO must clear the floor; models
    // without an ELO are kept (no data != bad).
    if (min_elo > 0 && c.elo != null && c.elo < min_elo) return false;
    return true;
  });
  affordable.sort((a, b) => b.params_b - a.params_b);

  const largest = affordable[0] ?? null;

  const comparableQuality = pickComparableQuality(affordable, pricing, api_key, api_override, largest);

  return {
    api_cost,
    tiers: affordable,
    largest,
    gradedTiers: pickGradedTiers(affordable, api_cost, largest),
    comparableQuality,
    all_candidates: candidates,
  };
}

/**
 * Pick the cheapest affordable open-weight model whose quality tier is at
 * least as high as the API model's tier. If nothing matches the exact tier,
 * fall back one tier (e.g. frontier API → look for strong open-weight).
 *
 * Returns null when:
 *   - we don't recognize the API model's tier (curated list miss), OR
 *   - no affordable candidate clears the floor.
 */
function pickComparableQuality(
  affordable: ConfigResult[],
  pricing: Pricing,
  api_key: string,
  api_override: { input_per_1m: number; output_per_1m: number } | undefined,
  largest: ConfigResult | null
): ComparableQualityPick | null {
  if (api_override) return null; // custom rates -> no quality signal
  const apiRow = pricing.apis[api_key];
  if (!apiRow) return null;
  const apiTier = tierFor(apiRow.label);
  if (!apiTier) return null;
  const apiRank = TIER_RANK[apiTier];

  // Try exact tier first; fall back to one step below.
  const floors: QualityTier[] = [apiTier, tierMinusOne(apiTier)];
  for (const floor of floors) {
    const floorRank = TIER_RANK[floor];
    const eligible = affordable.filter(
      (c) => c.quality_tier && TIER_RANK[c.quality_tier] >= floorRank && TIER_RANK[c.quality_tier] <= apiRank
    );
    if (!eligible.length) continue;
    // Cheapest weekly cost wins; tie-break by larger params (more headroom).
    const cheapest = eligible.reduce((best, c) => {
      const bw = best.weekly_cost_with_ft ?? best.weekly_cost;
      const cw = c.weekly_cost_with_ft ?? c.weekly_cost;
      if (cw < bw) return c;
      if (cw === bw && c.params_b > best.params_b) return c;
      return best;
    });
    // If this is just the same pick as `largest`, skip on the first floor pass
    // so we don't surface a duplicate card. UI will hide on equality anyway.
    return {
      tier: cheapest,
      apiTier,
      modelTier: cheapest.quality_tier as QualityTier,
      floor,
    };
  }
  // Suppress unused-largest warning in some builds.
  void largest;
  return null;
}
