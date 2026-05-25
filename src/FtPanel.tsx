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
  cumulativeProjection,
  costForView,
  costViewSuffix,
  type CostView,
} from "./engine";
import { buildPaybackSentence, queriesToAmortize } from "./ftPayback";
import { computeFtCapex, FT_METHODS, type FtMethod } from "./ft";

const fmt = (n: number, d = 2) =>
  n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtCurrency = (n: number) => `$${fmt(n, 2)}`;
const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

interface FtPanelProps {
  /** Total parameters in billions — drives VRAM footprint (all weights load even for MoE). */
  params_b: number;
  /**
   * Active parameters per token in billions. For dense models pass the same
   * value as `params_b`. For MoE pass only the experts that fire (e.g.
   * Mixtral 8x7B: 47 total / 12 active; Llama-4-Scout: 109 / 17).
   * Compute (FLOPs) scales with active params, not total — this is the
   * whole point of MoE.
   */
  active_params_b: number;
  apiWeeklyCost: number;
  selfhostWeeklyCost: number;
  queriesPerWeek: number;
  view: CostView;
}

export function FtPanel({
  params_b,
  active_params_b,
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
  const [experimentsMultiplier, setExperimentsMultiplier] = useState(2.5);
  // "auto" lets the engine pick from VRAM footprint; otherwise the chosen
  // multiplier is passed through as an override.
  const [clusterOverhead, setClusterOverhead] = useState<"auto" | "1.0" | "1.3" | "1.6">(
    "auto"
  );

  const ftInputs = useMemo(
    () => ({
      num_examples: numExamples,
      tokens_per_example: tokensPerExample,
      method,
      epochs,
      prep_cost_usd: prepCost,
      experiments_multiplier: experimentsMultiplier,
      cluster_overhead: clusterOverhead === "auto" ? undefined : Number(clusterOverhead),
    }),
    [
      numExamples,
      tokensPerExample,
      method,
      epochs,
      prepCost,
      experimentsMultiplier,
      clusterOverhead,
    ]
  );

  // FLOPs scale with ACTIVE params (work per token). VRAM/cluster math uses
  // TOTAL via the third arg. For dense models these are equal so no-op.
  const capex = useMemo(
    () => computeFtCapex(active_params_b, ftInputs, params_b),
    [active_params_b, params_b, ftInputs]
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
  // Follow the global cost view so label, value and suffix agree. Previously
  // the value was always monthly but the suffix tracked `view`, so Weekly
  // showed "$X/wk" labeled "Monthly savings" — wrong number and wrong unit.
  const viewSavings = costForView(weeklySavings, view);
  const suffix = costViewSuffix(view);
  const savingsLabel =
    view === "monthly" ? "Monthly savings" : view === "annual" ? "Annual savings" : "Weekly savings";

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
        <div className="flex flex-col sm:col-span-2">
          <label className="text-xs font-medium text-slate-700 mb-1">
            Experiments / failed-runs multiplier
          </label>
          <input
            type="number"
            min={1}
            step={0.5}
            value={experimentsMultiplier}
            onChange={(e) =>
              setExperimentsMultiplier(Math.max(1, Number(e.target.value) || 1))
            }
            className={inputClass}
          />
          <span className="text-[10px] text-slate-500 mt-1">
            Most teams run 2–3 attempts before one sticks. 1× is the
            theoretical floor.
          </span>
        </div>
        <div className="flex flex-col sm:col-span-2">
          <label className="text-xs font-medium text-slate-700 mb-1">
            Cluster overhead
          </label>
          <select
            value={clusterOverhead}
            onChange={(e) =>
              setClusterOverhead(e.target.value as "auto" | "1.0" | "1.3" | "1.6")
            }
            className={inputClass}
          >
            <option value="auto">
              Auto ({fmt(capex.cluster_overhead, 2)}× — {capex.cluster_topology},{" "}
              {fmtInt(capex.ft_vram_gb)} GB FT VRAM)
            </option>
            <option value="1.0">1.0× — single GPU</option>
            <option value="1.3">1.3× — multi-GPU NVLink node</option>
            <option value="1.6">1.6× — multi-node</option>
          </select>
          <span className="text-[10px] text-slate-500 mt-1">
            Extra wall-clock time for gradient sync across GPUs. Auto picks
            from VRAM footprint.
          </span>
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
            One-time capex ({fmt(capex.experiments_multiplier, 1)}× campaign)
          </div>
          <div className="text-sm font-semibold text-slate-900">
            {fmtCurrency(capex.total_capex_usd)}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            1 run = {fmtCurrency(capex.single_run_gpu_cost_usd + prepCost)}
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
            {savingsLabel}
          </div>
          <div className="text-sm font-semibold text-emerald-700">
            {fmtCurrency(viewSavings)}
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
        {active_params_b < params_b && (
          <>
            <strong>MoE:</strong> FLOPs computed against {active_params_b}B active
            params per token (not the {params_b}B total). VRAM and cluster
            sizing still use the {params_b}B total since all experts load into
            memory.{" "}
          </>
        )}
        Anchored to the QLoRA paper's Guanaco-65B 24-hour benchmark. Cost is
        for the full campaign (multiplier above); 1-run cost shown in the
        breakdown. Quality impact of fine-tuning is not modeled — check Arena
        ELO of base vs tuned models separately.{" "}
        <a
          href="https://github.com/artvandelay/should-i-self-host-llm/blob/main/src/ft/ASSUMPTIONS.md"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-500 underline hover:text-slate-700"
        >
          Full assumptions & citations →
        </a>
      </p>
    </div>
  );
}
