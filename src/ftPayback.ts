import { type CumulativeProjection } from "./engine";

export type PaybackTone = "good" | "neutral" | "bad";

export interface PaybackSentence {
  tone: PaybackTone;
  headline: string;
  supporting?: string;
}

export interface QueriesToAmortize {
  queries: number;
  savings_per_query: number;
}

function fmtUSD(n: number): string {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function monthOrMonths(n: number): string {
  return n === 1 ? "month" : "months";
}

function cumulativeSavingsAtMonth(
  proj: CumulativeProjection,
  month: number
): number {
  const point = proj.points.find((p) => p.month === month);
  if (!point) return 0;
  return point.api_cumulative - point.selfhost_cumulative;
}

export function buildPaybackSentence(
  capex_usd: number,
  api_weekly: number,
  selfhost_weekly: number,
  proj: CumulativeProjection
): PaybackSentence {
  const weeklySavings = api_weekly - selfhost_weekly;

  if (selfhost_weekly >= api_weekly) {
    return {
      tone: "bad",
      headline: "Fine-tuning won't pay back at this workload",
      supporting:
        selfhost_weekly > api_weekly
          ? `Self-hosting already costs ${fmtUSD(selfhost_weekly)}/wk vs ${fmtUSD(api_weekly)}/wk on the API — add ${fmtUSD(capex_usd)} capex and you're further behind.`
          : `Self-hosting matches the API at ${fmtUSD(api_weekly)}/wk — a ${fmtUSD(capex_usd)} fine-tuning bill never amortizes.`,
    };
  }

  if (proj.crossover_month === null) {
    return {
      tone: "neutral",
      headline: "Fine-tuning capex won't pay back within 24 months",
      supporting: `You save ${fmtUSD(weeklySavings)}/wk vs the API, but ${fmtUSD(capex_usd)} upfront is too large to recover before month ${proj.horizon_months}.`,
    };
  }

  const n = proj.crossover_month;
  const savings12 = cumulativeSavingsAtMonth(proj, 12);
  const savings24 = cumulativeSavingsAtMonth(proj, 24);

  return {
    tone: "good",
    headline: `Fine-tuning pays for itself in ${n} ${monthOrMonths(n)}`,
    supporting: `After payback, you'd be ahead by ${fmtUSD(savings12)} at 12 months and ${fmtUSD(savings24)} at 24 months vs staying on the API.`,
  };
}

export function queriesToAmortize(
  capex_usd: number,
  api_weekly: number,
  selfhost_weekly: number,
  queries_per_week: number
): QueriesToAmortize | null {
  if (
    !Number.isFinite(capex_usd) ||
    !Number.isFinite(api_weekly) ||
    !Number.isFinite(selfhost_weekly) ||
    !Number.isFinite(queries_per_week)
  ) {
    return null;
  }

  const weeklySavings = api_weekly - selfhost_weekly;
  if (weeklySavings <= 0) return null;
  if (queries_per_week <= 0) return null;

  const savings_per_query = weeklySavings / queries_per_week;
  if (capex_usd === 0) {
    return { queries: 0, savings_per_query };
  }

  return {
    queries: Math.ceil(capex_usd / savings_per_query),
    savings_per_query,
  };
}

