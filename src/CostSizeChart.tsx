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

interface ChartPoint {
  params_b: number;
  weekly_cost: number;
  arch: "dense" | "moe";
  gpu: string;
  savings_pct: number;
  highlight: "largest" | "graded" | null;
  active_params_b: number;
  elo?: number;
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
    elo: c.elo,
  };
}

/**
 * Map ELO to fill colour intensity. Higher ELO -> more saturated.
 * We pick the per-arch hue then darken with ELO. Models with no ELO
 * render at the dim end so the chart still shows them but with a
 * visual hint that the quality signal is missing.
 */
function eloFill(arch: "dense" | "moe", elo?: number): { fill: string; stroke: string } {
  // Anchor: 1300 -> dim, 1500 -> bright. Clamp.
  const t = elo == null ? 0 : Math.max(0, Math.min(1, (elo - 1300) / 200));
  if (arch === "dense") {
    // Indigo scale: #c7d2fe (light) -> #312e81 (dark)
    const light = [199, 210, 254];
    const dark = [67, 56, 202]; // indigo-700
    const fill = light.map((c, i) => Math.round(c + (dark[i] - c) * t));
    return { fill: `rgb(${fill.join(",")})`, stroke: `rgb(49,46,129)` };
  }
  // Sky scale: #bae6fd -> #0c4a6e
  const light = [186, 230, 253];
  const dark = [14, 165, 233]; // sky-500
  const fill = light.map((c, i) => Math.round(c + (dark[i] - c) * t));
  return { fill: `rgb(${fill.join(",")})`, stroke: `rgb(3,105,161)` };
}

function CustomDot(props: any) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const isDense = payload.arch === "dense";
  const { fill: baseFill, stroke: baseStroke } = eloFill(payload.arch, payload.elo);
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
  if (isDense) {
    return <circle cx={cx} cy={cy} r={r} fill={baseFill} stroke={baseStroke} strokeWidth={1} fillOpacity={0.7} />;
  }
  return (
    <polygon
      points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`}
      fill={baseFill}
      stroke={baseStroke}
      strokeWidth={1}
      fillOpacity={0.7}
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
      {p.elo != null ? (
        <div className="text-slate-600 mt-0.5">Arena ELO {p.elo}</div>
      ) : (
        <div className="text-slate-400 mt-0.5">No Arena ELO</div>
      )}
    </div>
  );
}

export function CostSizeChart({ result }: { result: RecommendResult }) {
  const { api_cost, all_candidates, largest, gradedTiers } = result;

  const highlightKeys = new Map<string, ChartPoint["highlight"]>();
  if (largest) highlightKeys.set(`${largest.arch}-${largest.params_b}`, "largest");
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
