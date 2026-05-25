import { useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Check,
  Info,
  Link as LinkIcon,
  Zap,
  RefreshCw,
} from "lucide-react";
import {
  PRICING,
  QUANT_BYTES,
  QUANT_LABEL,
  KNOWN_MODELS,
  recommendTiers,
  costForView,
  costViewSuffix,
  breakEvenWeeks,
  type ConfigResult,
  type CostView,
  type Pattern,
  type Quant,
  type Vendor,
  type RecommendResult,
  type KnownModel as EngineKnownModel,
} from "./engine";
import { useUrlState } from "./useUrlState";
import { useLiveData } from "./useLiveData";
import type { ClosedApi, KnownModel } from "./modelsDev";
import { ApiCombobox } from "./ApiCombobox";
import { CostSizeChart } from "./CostSizeChart";
import { Presets, type PresetValues } from "./Presets";
import { ExportMenu } from "./ExportMenu";
// @ts-expect-error - Vite handles SVG imports as asset URLs at build time
import logoUrl from "./logo.svg";

// =============================================================================
// FORMATTING
// =============================================================================

const fmt = (n: number, d = 2) =>
  n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtCurrency = (n: number) => `$${fmt(n, 2)}`;
const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

// =============================================================================
// TINY UI BITS
// =============================================================================

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  suffix?: string;
  hint?: string;
  step?: number;
  min?: number;
}

/** Number input that allows transient empty/partial typing without snapping to 0. */
function NumberInput({ label, value, onChange, suffix, hint, step = 1, min = 0 }: NumberInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const display = draft ?? String(value);
  return (
    <div className="flex flex-col">
      <label className="text-sm font-medium text-slate-700 mb-1">{label}</label>
      <div className="relative">
        <input
          type="number"
          inputMode="decimal"
          value={display}
          step={step}
          min={min}
          onChange={(e) => {
            const raw = e.target.value;
            setDraft(raw);
            const n = parseFloat(raw);
            if (Number.isFinite(n)) onChange(n);
          }}
          onBlur={() => {
            if (draft === null) return;
            const n = parseFloat(draft);
            if (!Number.isFinite(n)) onChange(min);
            setDraft(null);
          }}
          className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400"
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-slate-500 pointer-events-none">
            {suffix}
          </span>
        )}
      </div>
      {hint && <span className="text-xs text-slate-500 mt-1">{hint}</span>}
    </div>
  );
}

function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex align-middle">
      <button
        type="button"
        aria-label="More info"
        onClick={() => setOpen((v) => !v)}
        onBlur={() => setOpen(false)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="text-slate-400 hover:text-slate-600 focus:text-slate-700 focus:outline-none"
      >
        <Info className="w-3.5 h-3.5" />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 w-64 max-w-[80vw] px-3 py-2 bg-slate-800 text-white text-xs rounded-md shadow-lg z-20 whitespace-normal leading-relaxed"
        >
          {text}
        </span>
      )}
    </span>
  );
}

interface SelectProps<T extends string> {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
  hint?: string;
  tooltip?: string;
}

function Select<T extends string>({ label, value, onChange, options, hint, tooltip }: SelectProps<T>) {
  return (
    <div className="flex flex-col">
      <label className="text-sm font-medium text-slate-700 mb-1 flex items-center gap-1.5">
        {label}
        {tooltip && <InfoTooltip text={tooltip} />}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 bg-white"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && <span className="text-xs text-slate-500 mt-1">{hint}</span>}
    </div>
  );
}

function Expander({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-slate-200 rounded-md overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3 flex items-center gap-2 text-left bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        <span className="font-medium text-slate-700">{title}</span>
      </button>
      {open && <div className="p-4 bg-white">{children}</div>}
    </div>
  );
}

// =============================================================================
// TIER CARD
// =============================================================================

function TierCard({
  tier,
  apiCost,
  badge,
  view,
}: {
  tier: ConfigResult;
  apiCost: number;
  badge?: { label: string; color: "indigo" | "green" } | null;
  view: CostView;
}) {
  const weekly = tier.weekly_cost_with_ft ?? tier.weekly_cost;
  const savings = apiCost - weekly;
  const savings_pct = apiCost > 0 ? (savings / apiCost) * 100 : 0;
  const viewedCost = costForView(weekly, view);
  const viewedSavings = costForView(savings, view);
  const suffix = costViewSuffix(view);
  const ringClass =
    badge?.color === "green"
      ? "border-emerald-400 bg-emerald-50/40 ring-2 ring-emerald-200"
      : badge?.color === "indigo"
      ? "border-indigo-400 bg-indigo-50/40 ring-2 ring-indigo-200"
      : "border-slate-200 bg-white";
  const badgeText =
    badge?.color === "green" ? "text-emerald-700" : "text-indigo-700";

  return (
    <div className={`border rounded-lg p-4 ${ringClass}`}>
      {badge && (
        <div className={`flex items-center gap-1 text-xs font-semibold mb-2 uppercase tracking-wide ${badgeText}`}>
          <Check className="w-3 h-3" /> {badge.label}
        </div>
      )}
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <div className="text-xl font-semibold text-slate-900">
            ~{tier.params_b}B {tier.arch === "moe" ? "MoE" : "dense"}
            {tier.arch === "moe" && (
              <span className="text-sm font-normal text-slate-500 ml-1">
                ({tier.active_params_b}B active)
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-slate-900">
            {fmtCurrency(viewedCost)}
            <span className="text-sm font-normal text-slate-500">{suffix}</span>
          </div>
          <div className="text-sm text-emerald-700 font-medium">
            saves {fmtCurrency(viewedSavings)} ({fmt(savings_pct, 0)}%)
          </div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <div>
          <span className="text-slate-500">Hardware:</span>{" "}
          <span className="font-medium text-slate-800">{tier.gpu}</span>
          {tier.replicas_needed > 1 && (
            <span className="ml-1 text-amber-700 font-medium">× {tier.replicas_needed}</span>
          )}
        </div>
        <div>
          <span className="text-slate-500">Quant:</span>{" "}
          <span className="font-medium text-slate-800">{QUANT_LABEL[tier.quant]}</span>
        </div>
        <div className="col-span-2">
          <span className="text-slate-500">Billing:</span>{" "}
          <span className="font-medium text-slate-800">{tier.billing_label}</span>
        </div>
        <div className="col-span-2">
          <span className="text-slate-500">GPU billed:</span>{" "}
          <span className="font-medium text-slate-800">
            {fmt(tier.billed_hours, 1)} hr/wk @ {fmtCurrency(tier.gpu_price_per_hr)}/hr
          </span>
        </div>
        {tier.ft_weekly !== undefined && tier.ft_weekly > 0 && (
          <div className="col-span-2 text-slate-500">
            includes {fmtCurrency(costForView(tier.ft_weekly, view))}{suffix} fine-tuning amortization
          </div>
        )}
      </div>

      {tier.saturated && (
        <div className="mt-3 flex items-start gap-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2.5 py-1.5">
          <Zap className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <span>
            Peak hour exceeds one GPU's throughput at this active-param size — costs
            assume <strong>{tier.replicas_needed} parallel replicas</strong>. Consider
            larger GPUs, lower-precision quant, or a smaller model.
          </span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// DERIVATION
// =============================================================================

function Derivation({
  inputs,
  result,
  effectiveApi,
}: {
  inputs: { queries_per_week: number; input_tokens: number; output_tokens: number; overhead_gb: number };
  result: RecommendResult;
  effectiveApi: { input_per_1m: number; output_per_1m: number };
}) {
  const top = result.largest;
  if (!top) return null;
  const weekly = top.weekly_cost_with_ft ?? top.weekly_cost;
  return (
    <div className="text-sm text-slate-700 space-y-1.5 font-mono leading-relaxed">
      <div>
        1. API cost per query = ({fmtInt(inputs.input_tokens)} in × {fmtCurrency(effectiveApi.input_per_1m)}/M +{" "}
        {fmtInt(inputs.output_tokens)} out × {fmtCurrency(effectiveApi.output_per_1m)}/M) / 1M ={" "}
        <span className="font-semibold">
          {fmtCurrency(
            (inputs.input_tokens * effectiveApi.input_per_1m +
              inputs.output_tokens * effectiveApi.output_per_1m) /
              1_000_000
          )}
        </span>
      </div>
      <div>
        2. Weekly API budget = {fmtInt(inputs.queries_per_week)} queries × cost/query ={" "}
        <span className="font-semibold">{fmtCurrency(result.api_cost)}</span>
      </div>
      <div>3. Target: largest model whose weekly self-host cost ≤ this budget.</div>
      <div>
        4. Top tier: <span className="font-semibold">~{top.params_b}B {top.arch}</span> at{" "}
        <span className="font-semibold">{QUANT_LABEL[top.quant]}</span>
      </div>
      <div>
        5. VRAM required = {top.params_b}B × {QUANT_BYTES[top.quant]} B/param + {top.effective_overhead_gb} GB overhead
        {top.effective_overhead_gb !== inputs.overhead_gb && (
          <span className="text-slate-500"> (auto-scaled from {inputs.overhead_gb} GB based on model size)</span>
        )} ={" "}
        <span className="font-semibold">{fmt(top.vram_needed_gb, 1)} GB</span>
      </div>
      <div>
        6. Cheapest GPU with ≥ {fmt(top.vram_needed_gb, 1)} GB VRAM ={" "}
        <span className="font-semibold">{top.gpu}</span> ({top.vram_available_gb} GB) @{" "}
        {fmtCurrency(top.gpu_price_per_hr)}/hr
        {top.replicas_needed > 1 && <> × {top.replicas_needed} replicas (saturation)</>}
      </div>
      <div>
        7. Cheapest billing mode for this traffic shape ={" "}
        <span className="font-semibold">{top.billing_label}</span>, {fmt(top.billed_hours, 1)} hr/wk billed
      </div>
      <div>
        8. Weekly self-host cost = {fmtCurrency(top.gpu_price_per_hr)}/hr × {fmt(top.billed_hours, 1)} hr ={" "}
        <span className="font-semibold">{fmtCurrency(top.weekly_cost)}</span>
        {top.ft_weekly !== undefined && top.ft_weekly > 0 && (
          <>
            {" "}+ {fmtCurrency(top.ft_weekly)} fine-tuning ={" "}
            <span className="font-semibold">{fmtCurrency(weekly)}</span>
          </>
        )}
      </div>
      <div className="pt-1">
        9. Savings vs API = {fmtCurrency(result.api_cost)} − {fmtCurrency(weekly)} ={" "}
        <span className="font-semibold text-emerald-700">{fmtCurrency(result.api_cost - weekly)}</span>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN APP
// =============================================================================

export default function App() {
  const [queries_per_week, setQpw] = useUrlState<number>("qpw", 500_000);
  const [input_tokens, setInTok] = useUrlState<number>("in", 1500);
  const [output_tokens, setOutTok] = useUrlState<number>("out", 400);
  const [api_key, setApiKey] = useUrlState<string>("api", "openai_gpt5");
  const [api_input_override, setApiInputOverride] = useUrlState<number>("ai", 1.0);
  const [api_output_override, setApiOutputOverride] = useUrlState<number>("ao", 3.0);
  const [pattern, setPattern] = useUrlState<Pattern>("pat", "business");
  const [quant_pref, setQuantPref] = useUrlState<Quant>("q", "fp16");
  const [vendor, setVendor] = useUrlState<Vendor>("v", "runpod");
  const [show_all_tiers, setShowAllTiers] = useState(false);
  const [cost_view, setCostView] = useState<CostView>(() => {
    if (typeof window === "undefined") return "weekly";
    const stored = localStorage.getItem("cost_view");
    if (stored === "weekly" || stored === "monthly" || stored === "annual") return stored;
    return "weekly";
  });
  const updateCostView = (v: CostView) => {
    setCostView(v);
    if (typeof window !== "undefined") localStorage.setItem("cost_view", v);
  };
  const [setup_cost, setSetupCost] = useState<number>(5000);
  const [dismissedBannerFp, setDismissedBannerFp] = useState<string | null>(
    () => (typeof window !== "undefined" ? localStorage.getItem("dismissed_banner_fp") : null)
  );

  // Advanced
  const [overhead_gb, setOverheadGb] = useUrlState<number>("oh", 4);
  const [cold_start_sec, setColdStartSec] = useUrlState<number>("cs", 30);
  const [min_params_b, setMinParamsB] = useUrlState<number>("mp", 0);
  const [ft_cost, setFtCost] = useUrlState<number>("ft", 0);
  const [ft_weeks, setFtWeeks] = useUrlState<number>("fw", 52);

  // Live data from models.dev
  const bundledApis: ClosedApi[] = Object.entries(PRICING.apis)
    .filter(([k]) => k !== "custom")
    .map(([k, v]) => ({ label: v.label, input_per_1m: v.input_per_1m, output_per_1m: v.output_per_1m, last_seen: PRICING.last_updated }));

  // Seed known models from the bundled engine JSON (for first render before models.dev responds)
  const bundledKnownModels: KnownModel[] = KNOWN_MODELS.map((m: EngineKnownModel) => ({
    params_b: m.params_b,
    active_b: m.active_b ?? null,
    name: m.name,
    arch: m.arch,
    source: m.source ?? "bundled",
    last_seen: m.last_seen ?? PRICING.last_updated,
  }));

  const { apis: liveApis, knownModels, status, refresh } = useLiveData(
    bundledApis,
    bundledKnownModels,
    PRICING.gpu_last_updated ?? PRICING.last_updated
  );

  // Merge live APIs into pricing shape for the engine
  const livePricingApis = useMemo(() => {
    const map: Record<string, { label: string; input_per_1m: number; output_per_1m: number }> = {};
    for (const a of liveApis) {
      const key = a.label.toLowerCase().replace(/[^a-z0-9]/g, "_");
      map[key] = { label: a.label, input_per_1m: a.input_per_1m, output_per_1m: a.output_per_1m };
    }
    // Always keep custom for manual overrides
    map["custom"] = PRICING.apis["custom"];
    return map;
  }, [liveApis]);

  const activePricing = useMemo(() => ({
    ...PRICING,
    apis: livePricingApis,
  }), [livePricingApis]);

  // Resolve the active api key: if URL/default key isn't in the current map, fall back to the first one.
  const resolvedApiKey = useMemo(() => {
    if (livePricingApis[api_key]) return api_key;
    const firstNonCustom = Object.keys(livePricingApis).find((k) => k !== "custom");
    return firstNonCustom ?? "custom";
  }, [livePricingApis, api_key]);

  const apiOverride = resolvedApiKey === "custom"
    ? { input_per_1m: api_input_override, output_per_1m: api_output_override }
    : undefined;

  const result = useMemo(
    () =>
      recommendTiers({
        pricing: activePricing,
        queries_per_week,
        input_tokens,
        output_tokens,
        api_key: resolvedApiKey,
        api_override: apiOverride,
        pattern,
        vendor,
        quant_pref,
        min_params_b,
        overhead_gb,
        cold_start_sec,
        ft_cost,
        ft_weeks,
        knownModels: knownModels as EngineKnownModel[],
      }),
    [
      activePricing,
      knownModels,
      queries_per_week,
      input_tokens,
      output_tokens,
      resolvedApiKey,
      api_input_override,
      api_output_override,
      pattern,
      vendor,
      quant_pref,
      min_params_b,
      overhead_gb,
      cold_start_sec,
      ft_cost,
      ft_weeks,
    ]
  );

  const apiOptions = useMemo(
    () =>
      Object.entries(livePricingApis).map(([k, v]) => {
        const isActiveCustom = k === "custom" && resolvedApiKey === "custom";
        const inP = isActiveCustom ? api_input_override : v.input_per_1m;
        const outP = isActiveCustom ? api_output_override : v.output_per_1m;
        // First whitespace-split token of the label is the provider name (e.g. "OpenAI GPT-4o" -> "OpenAI").
        const group =
          k === "custom"
            ? "Custom"
            : (v.label.split(" ")[0] || "Other");
        return {
          value: k,
          group,
          label:
            k === "custom" && resolvedApiKey !== "custom"
              ? v.label
              : `${v.label} ($${inP}/M in, $${outP}/M out)`,
        };
      }),
    [livePricingApis, resolvedApiKey, api_input_override, api_output_override]
  );

  const effectiveApiRates =
    resolvedApiKey === "custom"
      ? { input_per_1m: api_input_override, output_per_1m: api_output_override }
      : livePricingApis[resolvedApiKey] ?? { input_per_1m: 1, output_per_1m: 3 };

  const largest = result.largest;
  const graded = result.gradedTiers;

  const featuredKeys = new Set<string>();
  if (largest) featuredKeys.add(`${largest.arch}-${largest.params_b}`);
  for (const g of graded) featuredKeys.add(`${g.tier.arch}-${g.tier.params_b}`);
  const otherTiers = result.tiers.filter(
    (t) => !featuredKeys.has(`${t.arch}-${t.params_b}`)
  );

  const [copied, setCopied] = useState(false);
  const copyShareLink = async () => {
    if (typeof window === "undefined") return;
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore -- older browsers
    }
  };

  // Build failure banner message (dismissible, fingerprint-driven)
  const bannerErrors: string[] = [];
  if (status.apis === "cached" && status.apisError) bannerErrors.push(`APIs: ${status.apisError}`);
  if (status.models === "cached" && status.modelsError) bannerErrors.push(`Models: ${status.modelsError}`);
  const bannerFp = bannerErrors.join("|") || null;
  const showBanner = bannerFp && bannerFp !== dismissedBannerFp;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 sm:p-8">
      <div className="max-w-5xl mx-auto">

        {/* Failure banner */}
        {showBanner && bannerFp && (
          <div className="mb-4 flex items-start gap-3 bg-amber-50 border-2 border-amber-300 rounded-lg p-4">
            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-amber-900 font-medium">
                Live price refresh failed. Showing cached prices.
              </p>
              <p className="text-xs text-amber-800 mt-0.5">{bannerErrors.join(" | ")}</p>
            </div>
            <button
              onClick={() => {
                localStorage.setItem("dismissed_banner_fp", bannerFp);
                setDismissedBannerFp(bannerFp);
              }}
              className="text-amber-700 hover:text-amber-900 text-sm font-medium"
            >
              Dismiss
            </button>
          </div>
        )}

        <header className="mb-6 flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-start gap-3">
            <img
              src={logoUrl}
              alt="Should I self-host my LLM logo"
              className="w-12 h-12 flex-shrink-0 mt-0.5"
              width="48"
              height="48"
            />
            <div>
              <h1 className="text-3xl font-bold text-slate-900">
                API vs Self-Host LLM Cost Calculator
              </h1>
              <p className="text-slate-600 mt-1 max-w-3xl">
                Given your traffic and the API price you'd pay, find the largest open-weight model
                you can self-host for the same cost or less.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {status.apis === "loading" ? (
              <span className="text-xs text-slate-500 flex items-center gap-1 px-2 py-1 bg-slate-100 rounded">
                <RefreshCw className="w-3 h-3 animate-spin" /> Updating prices...
              </span>
            ) : (
              <button
                onClick={refresh}
                className="text-sm flex items-center gap-1.5 px-3 py-2 border border-slate-300 rounded-md hover:bg-white text-slate-700"
                title="Pull latest pricing from models.dev"
              >
                <RefreshCw className="w-4 h-4" /> Refresh prices
              </button>
            )}
            <button
              onClick={copyShareLink}
              className={`text-sm flex items-center gap-1.5 px-3 py-2 border rounded-md transition-colors ${
                copied
                  ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                  : "border-slate-300 hover:bg-white text-slate-700"
              }`}
              title="Copy a link with all your inputs encoded"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4" /> Copied!
                </>
              ) : (
                <>
                  <LinkIcon className="w-4 h-4" /> Copy share link
                </>
              )}
            </button>
          </div>
        </header>

        {/* Presets */}
        <Presets
          current={{ queries_per_week, input_tokens, output_tokens, pattern }}
          onApply={(p: PresetValues) => {
            setQpw(p.queries_per_week);
            setInTok(p.input_tokens);
            setOutTok(p.output_tokens);
            setPattern(p.pattern);
          }}
        />

        {/* Inputs */}
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5 mb-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 mb-4">
            Your workload
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <NumberInput label="Queries per week" value={queries_per_week} onChange={setQpw} step={1000} hint="Total LLM calls/week" />
            <NumberInput label="Avg input tokens" value={input_tokens} onChange={setInTok} step={100} hint="Per query (prompt + context)" />
            <NumberInput label="Avg output tokens" value={output_tokens} onChange={setOutTok} step={50} hint="Per query (response)" />
            <ApiCombobox
              label="Comparing against (API)"
              value={resolvedApiKey}
              onChange={setApiKey}
              options={apiOptions}
              hint={`${apiOptions.length} models available · type to search`}
            />
            <Select<Pattern>
              label="Traffic pattern"
              value={pattern}
              onChange={setPattern}
              tooltip="How traffic is distributed over the week. Uniform = even 24/7. Business hours = Mon-Fri 9-6. Bursty = spiky weekday peaks. Cold-start-per-query = traffic so sparse every request pays a cold boot. Always-warm = treat as continuous so the GPU never spins down."
              options={[
                { value: "uniform", label: "Uniform (24/7 even)" },
                { value: "business", label: "Business hours (Mon-Fri 9-6)" },
                { value: "bursty", label: "Bursty (spiky peaks)" },
                { value: "cold_per_query", label: "Cold start on every query (very sparse)" },
                { value: "always_warm", label: "Always warm (no scale-down)" },
              ]}
            />
            <Select<Quant>
              label="Quantization"
              value={quant_pref}
              onChange={setQuantPref}
              options={[
                { value: "fp16", label: "FP16 (best quality, biggest)" },
                { value: "int8", label: "INT8 (balanced)" },
                { value: "int4", label: "INT4 (smallest, some quality loss)" },
              ]}
              hint="Lower precision = fits bigger model on same GPU"
            />
          </div>

          {resolvedApiKey === "custom" && (
            <div className="mt-4 grid grid-cols-2 gap-4 p-3 bg-slate-50 rounded-md border border-slate-200">
              <NumberInput label="Input price ($/M tokens)" value={api_input_override} onChange={setApiInputOverride} step={0.1} />
              <NumberInput label="Output price ($/M tokens)" value={api_output_override} onChange={setApiOutputOverride} step={0.1} />
            </div>
          )}
        </div>

        {/* Advanced */}
        <div className="mb-5 space-y-2">
          <Expander title="Advanced settings">
            <p className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded px-3 py-2 mb-4">
              These defaults work for typical workloads. The engine auto-scales VRAM overhead with
              model size (4 GB for ≤13B, up to 24 GB for 200B+) — your value here is the floor.
              Override with measured numbers if you have them.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Select<Vendor>
                label="Cloud GPU vendor"
                value={vendor}
                onChange={setVendor}
                options={[
                  { value: "runpod", label: "Runpod (cheapest)" },
                  { value: "lambda", label: "Lambda Labs" },
                  { value: "modal", label: "Modal" },
                ]}
              />
              <NumberInput label="VRAM overhead floor (GB)" value={overhead_gb} onChange={setOverheadGb} step={1} hint="KV cache + activations. Engine auto-scales above this." />
              <NumberInput label="Cold-start penalty (sec)" value={cold_start_sec} onChange={setColdStartSec} step={5} hint="For scale-to-zero mode" />
              <NumberInput label="Min model size (B params)" value={min_params_b} onChange={setMinParamsB} step={1} hint="Filter out tiny models even if cheaper. 0 = no floor." />
              <NumberInput label="One-time fine-tuning cost ($)" value={ft_cost} onChange={setFtCost} step={100} hint="Added amortized to weekly cost" />
              <NumberInput label="Amortize over (weeks)" value={ft_weeks} onChange={setFtWeeks} step={1} min={1} hint="How long the fine-tune is in service" />
            </div>
          </Expander>
        </div>

        {/* Results */}
        {result.tiers.length === 0 ? (
          <div className="bg-amber-50 border-2 border-amber-300 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-amber-900">Stick with the API</h3>
                <p className="text-amber-800 mt-1">
                  At this volume ({fmtInt(queries_per_week)} queries/week, {fmtCurrency(result.api_cost)}/wk on
                  the API), no self-host config beats the API price — even the smallest models on the cheapest
                  GPUs cost more than you'd pay the API directly.
                </p>
                <p className="text-amber-800 mt-2 text-sm">
                  Self-hosting starts winning at higher volumes. Try increasing queries/week or output tokens
                  to see the crossover.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5 mb-5">
              <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
                <h2 className="text-lg font-semibold text-slate-900">
                  {result.tiers.length} self-host {result.tiers.length === 1 ? "option" : "options"} fit under
                  your API budget
                </h2>
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 text-xs">
                    {(["weekly", "monthly", "annual"] as CostView[]).map((v) => (
                      <button
                        key={v}
                        type="button"
                        onClick={() => updateCostView(v)}
                        aria-pressed={cost_view === v}
                        className={`px-2.5 py-1 rounded transition-colors ${
                          cost_view === v
                            ? "bg-white text-indigo-700 font-semibold shadow-sm border border-indigo-200"
                            : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        {v.charAt(0).toUpperCase() + v.slice(1)}
                      </button>
                    ))}
                  </div>
                  <div className="text-sm text-slate-600">
                    API cost:{" "}
                    <span className="font-semibold text-slate-900">
                      {fmtCurrency(costForView(result.api_cost, cost_view))}
                      {costViewSuffix(cost_view)}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-sm text-slate-600 mb-2 flex items-start gap-1.5">
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-slate-400" />
                <span>
                  The chart plots weekly cost vs model size (log-log) for every candidate; the red dashed line
                  is the API price. We highlight the <strong>largest</strong> model that fits (green ring) and
                  up to three <strong>savings-banded</strong> picks — the largest model at each of 80%+, 50%+,
                  and 20%+ savings (indigo rings) — so you can see the cost/capability trade-off at a glance.
                </span>
              </p>
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2 mb-4">
                <strong>Note:</strong> Inference costs only. Engineering time (serving infra, evals, monitoring,
                on-call) and any fine-tuning compute beyond what you enter in Advanced are not included. GPU
                hourly rates are approximate — verify with your vendor before quoting.
              </p>

              <CostSizeChart result={result} view={cost_view} />

              <div className="space-y-3">
                {largest && (
                  <TierCard
                    tier={largest}
                    apiCost={result.api_cost}
                    badge={{ label: "Largest that fits", color: "green" }}
                    view={cost_view}
                  />
                )}
                {graded.map((g) => (
                  <TierCard
                    key={`graded-${g.tier.arch}-${g.tier.params_b}`}
                    tier={g.tier}
                    apiCost={result.api_cost}
                    badge={{ label: g.label, color: "indigo" }}
                    view={cost_view}
                  />
                ))}
                {(show_all_tiers ? otherTiers : otherTiers.slice(0, 2)).map((tier) => (
                  <TierCard
                    key={`${tier.arch}-${tier.params_b}`}
                    tier={tier}
                    apiCost={result.api_cost}
                    badge={null}
                    view={cost_view}
                  />
                ))}
              </div>
              <div className="mt-4 flex justify-end">
                <ExportMenu
                  inputs={{
                    queries_per_week,
                    input_tokens,
                    output_tokens,
                    api_key: resolvedApiKey,
                    api_label: livePricingApis[resolvedApiKey]?.label ?? resolvedApiKey,
                    api_input_per_1m: effectiveApiRates.input_per_1m,
                    api_output_per_1m: effectiveApiRates.output_per_1m,
                    pattern,
                    quant_pref,
                    vendor,
                  }}
                  result={result}
                />
              </div>
              {otherTiers.length > 2 && (
                <button
                  onClick={() => setShowAllTiers(!show_all_tiers)}
                  className="mt-3 w-full px-4 py-2.5 text-sm font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-md transition-colors flex items-center justify-center gap-1.5"
                >
                  {show_all_tiers ? (
                    <>Show fewer <ChevronDown className="w-4 h-4 rotate-180" /></>
                  ) : (
                    <>Show all {result.tiers.length} options that fit <ChevronDown className="w-4 h-4" /></>
                  )}
                </button>
              )}
            </div>

            {largest && (
              <div className="mb-3">
                <Expander title="Break-even analysis (setup cost payback)">
                  {(() => {
                    const selfhostWeekly = largest.weekly_cost_with_ft ?? largest.weekly_cost;
                    const weeks = breakEvenWeeks(result.api_cost, selfhostWeekly, setup_cost);
                    const weeklySavings = result.api_cost - selfhostWeekly;
                    return (
                      <div className="space-y-3">
                        <p className="text-sm text-slate-600">
                          One-time costs (engineer ramp-up, infra setup, migration, evals) often
                          dominate the first months of self-hosting. Enter an estimate to see
                          when cumulative API spend overtakes cumulative self-host spend for the{" "}
                          <strong>largest-that-fits</strong> pick (~{largest.params_b}B{" "}
                          {largest.arch === "moe" ? "MoE" : "dense"}).
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          <NumberInput
                            label="One-time setup / migration cost ($)"
                            value={setup_cost}
                            onChange={setSetupCost}
                            step={500}
                            hint="Engineering, infra, evals, on-call ramp-up"
                          />
                          <div className="flex flex-col justify-end">
                            <div className="text-xs text-slate-500 mb-1">Weekly savings vs API</div>
                            <div className="font-mono text-sm text-slate-800">
                              {weeklySavings > 0
                                ? `${fmtCurrency(weeklySavings)}/wk (${fmtCurrency(
                                    costForView(weeklySavings, cost_view)
                                  )}${costViewSuffix(cost_view)})`
                                : "$0 (self-host is not cheaper at this volume)"}
                            </div>
                          </div>
                        </div>
                        <div className="rounded-md border border-indigo-200 bg-indigo-50/60 px-4 py-3">
                          {weeks === null ? (
                            <div className="text-sm text-slate-800">
                              {weeklySavings <= 0
                                ? "Self-host is not cheaper than the API at this workload — break-even never happens."
                                : "Never within 10 years at this setup cost."}
                            </div>
                          ) : (
                            <div className="text-sm text-slate-800">
                              Self-hosting pays back in{" "}
                              <span className="font-semibold text-indigo-700">
                                ~{weeks} {weeks === 1 ? "week" : "weeks"}
                              </span>
                              {weeks >= 4 && (
                                <span className="text-slate-600">
                                  {" "}(~{(weeks / (52 / 12)).toFixed(1)} months
                                  {weeks >= 52 ? `, ~${(weeks / 52).toFixed(1)} years` : ""})
                                </span>
                              )}
                              .
                            </div>
                          )}
                          <div className="text-xs text-slate-500 mt-1">
                            Capped at 520 weeks (10 years). Assumes weekly costs stay flat — real
                            workloads grow, prices change, and ongoing engineering effort is not
                            included.
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </Expander>
              </div>
            )}

            <Expander title="Show the math (derivation)">
              <Derivation
                inputs={{ queries_per_week, input_tokens, output_tokens, overhead_gb }}
                result={result}
                effectiveApi={effectiveApiRates}
              />
            </Expander>
          </>
        )}

        <footer className="mt-6 text-center text-xs text-slate-500 space-y-1">
          <div>
            <span className="font-medium">APIs:</span> models.dev{" "}
            {status.apis === "live" ? "live" : status.apis === "loading" ? "loading..." : "cached"}
            {status.apisFetchedAt ? ` (${status.apisFetchedAt})` : ""}
            {" | "}
            <span className="font-medium">Models:</span> models.dev{" "}
            {status.models === "live" ? "live" : status.models === "loading" ? "loading..." : "cached"}
            {" | "}
            <span className="font-medium">GPUs:</span> {status.gpus}
          </div>
          <div>
            Bundled prices last updated: <span className="font-mono">{PRICING.last_updated}</span>{" · "}
            <code className="px-1 py-0.5 bg-slate-200 rounded">src/pricing.json</code>
            {" (auto-refreshed nightly by GitHub Action)"}
          </div>
          <div className="pt-2 flex items-center justify-center gap-3 text-slate-500">
            <span>Maintained by Jigar Doshi</span>
            <span className="text-slate-300">·</span>
            <a
              href="https://github.com/artvandelay/should-i-self-host-llm"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-indigo-600 inline-flex items-center gap-1"
              aria-label="GitHub repository"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.92.58.11.79-.25.79-.55 0-.27-.01-1.17-.02-2.12-3.2.69-3.87-1.36-3.87-1.36-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.69 1.24 3.34.95.1-.74.4-1.24.72-1.53-2.55-.29-5.24-1.27-5.24-5.66 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.45.11-3.03 0 0 .96-.31 3.15 1.17a10.9 10.9 0 0 1 5.74 0c2.18-1.48 3.14-1.17 3.14-1.17.62 1.58.23 2.74.11 3.03.74.8 1.18 1.82 1.18 3.07 0 4.4-2.69 5.36-5.25 5.65.41.36.78 1.07.78 2.16 0 1.56-.02 2.81-.02 3.19 0 .31.21.67.8.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5Z"/>
              </svg>
              GitHub
            </a>
            <span className="text-slate-300">·</span>
            <a
              href="https://twitter.com/jigarkdoshi"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-indigo-600 inline-flex items-center gap-1"
              aria-label="Twitter / X profile"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231 5.45-6.231Zm-1.161 17.52h1.833L7.084 4.126H5.117l11.966 15.644Z"/>
              </svg>
              @jigarkdoshi
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}
