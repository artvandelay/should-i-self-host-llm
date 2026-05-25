import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import type { ConfigResult, RecommendResult } from "./engine";
import { TIER_RANK, type QualityTier } from "./qualityTiers";

interface ChartPoint {
  params_b: number;
  weekly_cost: number;
  arch: "dense" | "moe";
  gpu: string;
  savings_pct: number;
  highlight: "largest" | "comparable" | "graded" | null;
  active_params_b: number;
  quality_tier: QualityTier | null;
}

function toPoint(
  c: ConfigResult,
  api_cost: number,
  highlight: ChartPoint["highlight"]
): ChartPoint {
  const weekly = c.weekly_cost_with_ft ?? c.weekly_cost;
  const savings_pct = api_cost > 0 ? ((api_cost - weekly) / api_cost) * 100 : 0;
  return {
    params_b: c.params_b,
    weekly_cost: Math.max(weekly, 0.01),
    arch: c.arch,
    gpu: c.gpu,
    savings_pct,
    highlight,
    active_params_b: c.active_params_b,
    quality_tier: c.quality_tier ?? null,
  };
}

function CustomDot(props: any) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const isDense = payload.arch === "dense";
  const baseFill = isDense ? "#6366f1" : "#0ea5e9";
  const baseStroke = isDense ? "#4338ca" : "#0369a1";
  const r = 5;

  if (payload.highlight === "largest") {
    return (
      <g>
        <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#10b981" strokeWidth={2.5} />
        {isDense ? (
          <circle cx={cx} cy={cy} r={r} fill={baseFill} stroke={baseStroke} strokeWidth={1} />
        ) : (
          <polygon
            points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`}
            fill={baseFill}
            stroke={baseStroke}
            strokeWidth={1}
          />
        )}
      </g>
    );
  }
  if (payload.highlight === "comparable") {
    return (
      <g>
        <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#0d9488" strokeWidth={2.5} />
        {isDense ? (
          <circle cx={cx} cy={cy} r={r} fill={baseFill} stroke={baseStroke} strokeWidth={1} />
        ) : (
          <polygon
            points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`}
            fill={baseFill}
            stroke={baseStroke}
            strokeWidth={1}
          />
        )}
      </g>
    );
  }
  if (payload.highlight === "graded") {
    return (
      <g>
        <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#6366f1" strokeWidth={2.5} />
        {isDense ? (
          <circle cx={cx} cy={cy} r={r} fill={baseFill} stroke={baseStroke} strokeWidth={1} />
        ) : (
          <polygon
            points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`}
            fill={baseFill}
            stroke={baseStroke}
            strokeWidth={1}
          />
        )}
      </g>
    );
  }
  // Subtle quality-tier hint: higher tiers render at higher opacity so the
  // eye can track the "comparable-quality" region of the chart without losing
  // the dense/MoE shape distinction.
  const tierOpacity =
    payload.quality_tier
      ? 0.4 + 0.15 * (TIER_RANK[payload.quality_tier as QualityTier] - 1)
      : 0.6;
  if (isDense) {
    return <circle cx={cx} cy={cy} r={r} fill={baseFill} stroke={baseStroke} strokeWidth={1} fillOpacity={tierOpacity} />;
  }
  return (
    <polygon
      points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`}
      fill={baseFill}
      stroke={baseStroke}
      strokeWidth={1}
      fillOpacity={tierOpacity}
    />
  );
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload as ChartPoint;
  return (
    <div className="bg-white border border-slate-300 rounded shadow-md px-2.5 py-1.5 text-xs">
      <div className="font-semibold text-slate-900">
        ~{p.params_b}B {p.arch === "moe" ? `MoE (${p.active_params_b}B active)` : "dense"}
      </div>
      <div className="text-slate-700">${p.weekly_cost.toFixed(2)}/wk</div>
      <div className="text-emerald-700">saves {p.savings_pct.toFixed(0)}%</div>
      <div className="text-slate-500">{p.gpu}</div>
    </div>
  );
}

export function CostSizeChart({ result }: { result: RecommendResult }) {
  const { api_cost, all_candidates, largest, gradedTiers, comparableQuality } = result;

  const highlightKeys = new Map<string, ChartPoint["highlight"]>();
  if (largest) highlightKeys.set(`${largest.arch}-${largest.params_b}`, "largest");
  if (comparableQuality) {
    const k = `${comparableQuality.tier.arch}-${comparableQuality.tier.params_b}`;
    if (!highlightKeys.has(k)) highlightKeys.set(k, "comparable");
  }
  for (const g of gradedTiers) {
    const k = `${g.tier.arch}-${g.tier.params_b}`;
    if (!highlightKeys.has(k)) highlightKeys.set(k, "graded");
  }

  const points: ChartPoint[] = all_candidates.map((c) =>
    toPoint(c, api_cost, highlightKeys.get(`${c.arch}-${c.params_b}`) ?? null)
  );

  const densePoints = points.filter((p) => p.arch === "dense");
  const moePoints = points.filter((p) => p.arch === "moe");

  const allCosts = points.map((p) => p.weekly_cost).concat([Math.max(api_cost, 0.01)]);
  const yMin = Math.max(0.01, Math.min(...allCosts) * 0.5);
  const yMax = Math.max(...allCosts) * 2;

  return (
    <div className="w-full h-[280px] sm:h-[320px] mb-4">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            type="number"
            dataKey="params_b"
            name="Model size"
            scale="log"
            domain={[1, 1500]}
            ticks={[1, 3, 10, 30, 100, 300, 1000]}
            tickFormatter={(v) => `${v}B`}
            label={{ value: "Model size (params, log)", position: "insideBottom", offset: -15, fontSize: 12, fill: "#64748b" }}
            stroke="#94a3b8"
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="weekly_cost"
            name="Weekly cost"
            scale="log"
            domain={[yMin, yMax]}
            tickFormatter={(v) => `$${v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(0) : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)}`}
            label={{ value: "Weekly cost (log)", angle: -90, position: "insideLeft", fontSize: 12, fill: "#64748b" }}
            stroke="#94a3b8"
            tick={{ fontSize: 11 }}
          />
          <ZAxis range={[60, 60]} />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
          <Legend verticalAlign="top" height={28} iconSize={10} wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine
            y={api_cost}
            stroke="#dc2626"
            strokeDasharray="5 5"
            label={{ value: `API cost $${api_cost.toFixed(2)}/wk`, position: "insideTopRight", fontSize: 11, fill: "#dc2626" }}
          />
          <Scatter name="Dense" data={densePoints} shape={<CustomDot />} fill="#6366f1" />
          <Scatter name="MoE" data={moePoints} shape={<CustomDot />} fill="#0ea5e9" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
