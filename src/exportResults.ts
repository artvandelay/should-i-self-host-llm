import { QUANT_LABEL, type ConfigResult, type Pattern, type Quant, type Vendor, type RecommendResult } from "./engine";

export interface ExportInputs {
  queries_per_week: number;
  input_tokens: number;
  output_tokens: number;
  api_key: string;
  api_label: string;
  api_input_per_1m: number;
  api_output_per_1m: number;
  pattern: Pattern;
  quant_pref: Quant;
  vendor: Vendor;
}

interface TierRow {
  role: string;
  params_b: number;
  active_params_b: number;
  arch: string;
  quant: string;
  gpu: string;
  replicas_needed: number;
  billing_mode: string;
  billed_hours_per_week: number;
  gpu_price_per_hr: number;
  weekly_cost: number;
  weekly_cost_with_ft: number;
  savings_vs_api: number;
  savings_pct: number;
}

function tierToRow(role: string, tier: ConfigResult, apiCost: number): TierRow {
  const weekly = tier.weekly_cost_with_ft ?? tier.weekly_cost;
  const savings = apiCost - weekly;
  return {
    role,
    params_b: tier.params_b,
    active_params_b: tier.active_params_b,
    arch: tier.arch,
    quant: QUANT_LABEL[tier.quant],
    gpu: tier.gpu,
    replicas_needed: tier.replicas_needed,
    billing_mode: tier.billing_label,
    billed_hours_per_week: Number(tier.billed_hours.toFixed(2)),
    gpu_price_per_hr: tier.gpu_price_per_hr,
    weekly_cost: Number(tier.weekly_cost.toFixed(2)),
    weekly_cost_with_ft: Number(weekly.toFixed(2)),
    savings_vs_api: Number(savings.toFixed(2)),
    savings_pct: apiCost > 0 ? Number(((savings / apiCost) * 100).toFixed(1)) : 0,
  };
}

export function buildExportRows(inputs: ExportInputs, result: RecommendResult): TierRow[] {
  const rows: TierRow[] = [];
  if (result.largest) rows.push(tierToRow("largest", result.largest, result.api_cost));
  const seen = new Set<string>();
  if (result.largest) seen.add(`${result.largest.arch}-${result.largest.params_b}`);
  for (const g of result.gradedTiers) {
    const key = `${g.tier.arch}-${g.tier.params_b}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push(tierToRow(g.label, g.tier, result.api_cost));
  }
  return rows;
}

function csvEscape(v: string | number): string {
  const s = String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function toCSV(inputs: ExportInputs, result: RecommendResult): string {
  const rows = buildExportRows(inputs, result);
  const lines: string[] = [];
  lines.push("# Workload inputs");
  lines.push("key,value");
  const inputPairs: [string, string | number][] = [
    ["queries_per_week", inputs.queries_per_week],
    ["input_tokens", inputs.input_tokens],
    ["output_tokens", inputs.output_tokens],
    ["pattern", inputs.pattern],
    ["quant_pref", inputs.quant_pref],
    ["vendor", inputs.vendor],
    ["api_key", inputs.api_key],
    ["api_label", inputs.api_label],
    ["api_input_per_1m_usd", inputs.api_input_per_1m],
    ["api_output_per_1m_usd", inputs.api_output_per_1m],
    ["api_weekly_cost_usd", Number(result.api_cost.toFixed(2))],
  ];
  for (const [k, v] of inputPairs) lines.push(`${csvEscape(k)},${csvEscape(v)}`);
  lines.push("");
  lines.push("# Self-host tiers");
  const headers: (keyof TierRow)[] = [
    "role",
    "params_b",
    "active_params_b",
    "arch",
    "quant",
    "gpu",
    "replicas_needed",
    "billing_mode",
    "billed_hours_per_week",
    "gpu_price_per_hr",
    "weekly_cost",
    "weekly_cost_with_ft",
    "savings_vs_api",
    "savings_pct",
  ];
  lines.push(headers.join(","));
  for (const r of rows) {
    lines.push(headers.map((h) => csvEscape(r[h])).join(","));
  }
  return lines.join("\n");
}

export function toJSON(inputs: ExportInputs, result: RecommendResult): string {
  const payload = {
    generated_at: new Date().toISOString(),
    workload: {
      queries_per_week: inputs.queries_per_week,
      input_tokens: inputs.input_tokens,
      output_tokens: inputs.output_tokens,
      pattern: inputs.pattern,
      quant_pref: inputs.quant_pref,
      vendor: inputs.vendor,
    },
    api: {
      key: inputs.api_key,
      label: inputs.api_label,
      input_per_1m_usd: inputs.api_input_per_1m,
      output_per_1m_usd: inputs.api_output_per_1m,
      weekly_cost_usd: Number(result.api_cost.toFixed(2)),
    },
    tiers: buildExportRows(inputs, result),
  };
  return JSON.stringify(payload, null, 2);
}
