import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  computeFtCapex,
  cumulativeProjection,
  costForView,
  costViewSuffix,
  type CostView,
} from "./engine";
import { buildPaybackSentence, queriesToAmortize } from "./ftPayback";
import { FT_METHODS, type FtMethod } from "./ftMethods";

const fmt = (n: number, d = 2) =>
  n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtCurrency = (n: number) => `$${fmt(n, 2)}`;
const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

interface FtPanelProps {
  params_b: number;
  apiWeeklyCost: number;
  selfhostWeeklyCost: number;
  queriesPerWeek: number;
  view: CostView;
}

export function FtPanel({
  params_b,
  apiWeeklyCost,
  selfhostWeeklyCost,
  queriesPerWeek,
  view,
}: FtPanelProps) {
  const [numExamples, setNumExamples] = useState(100_000);
  const [tokensPerExample, setTokensPerExample] = useState(1000);
  const [method, setMethod] = useState<FtMethod>("lora");
  const [epochs, setEpochs] = useState(3);
  const [prepCost, setPrepCost] = useState(0);

  const ftInputs = useMemo(
    () => ({
      num_examples: numExamples,
      tokens_per_example: tokensPerExample,
      method,
      epochs,
      prep_cost_usd: prepCost,
    }),
    [numExamples, tokensPerExample, method, epochs, prepCost]
  );

  const capex = useMemo(
    () => computeFtCapex(params_b, ftInputs),
    [params_b, ftInputs]
  );

  const projection = useMemo(
    () =>
      cumulativeProjection(
        apiWeeklyCost,
        selfhostWeeklyCost,
        capex.total_capex_usd,
        24
      ),
    [apiWeeklyCost, selfhostWeeklyCost, capex.total_capex_usd]
  );

  const payback = useMemo(
    () =>
      buildPaybackSentence(
        capex.total_capex_usd,
        apiWeeklyCost,
        selfhostWeeklyCost,
        projection
      ),
    [capex.total_capex_usd, apiWeeklyCost, selfhostWeeklyCost, projection]
  );

  const breakeven = useMemo(
    () =>
      queriesToAmortize(
        capex.total_capex_usd,
        apiWeeklyCost,
        selfhostWeeklyCost,
        queriesPerWeek
      ),
    [capex.total_capex_usd, apiWeeklyCost, selfhostWeeklyCost, queriesPerWeek]
  );

  const paybackToneClass =
    payback.tone === "good"
      ? "border-emerald-200 bg-emerald-50/70 text-emerald-900"
      : payback.tone === "bad"
      ? "border-rose-200 bg-rose-50/70 text-rose-900"
      : "border-amber-200 bg-amber-50/70 text-amber-900";

  const weeklySavings = apiWeeklyCost - selfhostWeeklyCost;
  const monthlySavings = costForView(weeklySavings, "monthly");
  const suffix = costViewSuffix(view);

  const inputClass =
    "w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400";

  return (
    <div className="mt-3 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <div className="flex flex-col">
          <label className="text-xs font-medium text-slate-700 mb-1">
            Training examples
          </label>
          <input
            type="number"
            min={0}
            step={1000}
            value={numExamples}
            onChange={(e) => setNumExamples(Number(e.target.value) || 0)}
            className={inputClass}
          />
        </div>
        <div className="flex flex-col">
          <label className="text-xs font-medium text-slate-700 mb-1">
            Tokens per example
          </label>
          <input
            type="number"
            min={0}
            step={100}
            value={tokensPerExample}
            onChange={(e) => setTokensPerExample(Number(e.target.value) || 0)}
            className={inputClass}
          />
        </div>
        <div className="flex flex-col">
          <label className="text-xs font-medium text-slate-700 mb-1">Epochs</label>
          <input
            type="number"
            min={0}
            step={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value) || 0)}
            className={inputClass}
          />
        </div>
        <div className="flex flex-col sm:col-span-2 lg:col-span-3">
          <span className="text-xs font-medium text-slate-700 mb-1">
            Fine-tuning method
          </span>
          <div className="flex flex-wrap gap-3">
            {(Object.keys(FT_METHODS) as FtMethod[]).map((id) => (
              <label
                key={id}
                className="inline-flex items-center gap-1.5 text-xs text-slate-700 cursor-pointer"
              >
                <input
                  type="radio"
                  name={`ft-method-${params_b}`}
                  checked={method === id}
                  onChange={() => setMethod(id)}
                  className="text-indigo-600 focus:ring-indigo-400"
                />
                {FT_METHODS[id].label}
              </label>
            ))}
          </div>
        </div>
        <div className="flex flex-col">
          <label className="text-xs font-medium text-slate-700 mb-1">
            Engineering / dataset prep ($)
          </label>
          <input
            type="number"
            min={0}
            step={500}
            value={prepCost}
            onChange={(e) => setPrepCost(Number(e.target.value) || 0)}
            className={inputClass}
          />
        </div>
      </div>

      <div className={`rounded-md border px-3 py-2.5 ${paybackToneClass}`}>
        <div className="text-sm font-semibold">{payback.headline}</div>
        {payback.supporting && (
          <div className="text-xs mt-1 opacity-90">{payback.supporting}</div>
        )}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
        <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            One-time capex
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {fmtCurrency(capex.total_capex_usd)}
          </div>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            GPU hours
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {fmt(capex.gpu_hours, 1)}
          </div>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            Monthly savings
          </div>
          <div className="text-sm font-semibold text-emerald-700">
            {fmtCurrency(monthlySavings)}
            {suffix}
          </div>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            Crossover
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {projection.crossover_month != null
              ? `Month ${projection.crossover_month}`
              : "Never within 24 months"}
          </div>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            Queries to break-even
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {breakeven === null
              ? "—"
              : breakeven.queries === 0
              ? "Already paid back"
              : fmtInt(breakeven.queries)}
          </div>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={projection.points} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 11 }}
              label={{ value: "Month", position: "insideBottom", offset: -2, fontSize: 11 }}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip
              formatter={(value) => fmtCurrency(Number(value ?? 0))}
              labelFormatter={(m) => `Month ${m}`}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="api_cumulative"
              name="API cumulative"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="selfhost_cumulative"
              name="Self-host cumulative"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
            />
            {projection.crossover_month != null && (
              <ReferenceLine
                x={projection.crossover_month}
                stroke="#22c55e"
                strokeDasharray="4 4"
                label={{ value: "Break-even", position: "top", fontSize: 10, fill: "#16a34a" }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="text-[10px] text-slate-400 leading-relaxed">
        Compute estimates use {FT_METHODS.lora.citation}; {FT_METHODS.qlora.citation};{" "}
        {FT_METHODS.full.citation}. H100 throughput assumes 40% MFU. Quality impact of
        fine-tuning is not estimated — compare Arena ELO of base vs fine-tuned models
        separately.
      </p>
    </div>
  );
}
