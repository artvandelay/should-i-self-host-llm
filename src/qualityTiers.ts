/**
 * Curated quality tiers for major model families.
 *
 * models.dev does not expose benchmark scores, so we use a coarse 4-bucket
 * tier based on widely-reported public eval results (MMLU, MMLU-Pro, GPQA,
 * HumanEval/MBPP/SWE-bench, Arena Elo, IFEval) as of late 2024 / early 2026.
 *
 * Tiers (rough proxies — your mileage on YOUR task will vary):
 *   - frontier : top of the leaderboards; e.g. GPT-5, Claude 4 Sonnet/Opus,
 *                Gemini 2.5 Pro, DeepSeek-V3/R1, Llama 3.1 405B, Qwen3 235B
 *   - strong   : near-frontier; e.g. GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro,
 *                Llama 3.3 70B, Qwen2.5 72B, Mixtral 8x22B, Qwen3-Coder 480B
 *   - mid      : usable general-purpose; e.g. Llama 3.1 8B, Qwen2.5 14B/32B,
 *                Mistral Small/Nemo, Gemma 2 27B, Phi-4
 *   - small    : capable but limited; e.g. Llama 3.2 1B/3B, Gemma 2 2B,
 *                Qwen2.5 7B, Phi-3 Mini
 *
 * This list is intentionally short and conservative. Edits are welcome —
 * see https://github.com/artvandelay/should-i-self-host-llm
 */

export type QualityTier = "frontier" | "strong" | "mid" | "small";

export const TIER_RANK: Record<QualityTier, number> = {
  frontier: 4,
  strong: 3,
  mid: 2,
  small: 1,
};

export const TIER_LABEL: Record<QualityTier, string> = {
  frontier: "Frontier-tier",
  strong: "Strong-tier",
  mid: "Mid-tier",
  small: "Small-tier",
};

export const TIER_TIERS: QualityTier[] = ["frontier", "strong", "mid", "small"];

interface TierRule {
  /** All these substrings (lowercased) must appear in the model name/id. */
  contains: string[];
  /** Optional regex against name+id (lowercased). */
  re?: RegExp;
  tier: QualityTier;
}

/**
 * Ordered, longest-most-specific first. The first matching rule wins.
 * Matching is case-insensitive against `${name} ${id}` joined.
 */
const RULES: TierRule[] = [
  // -------------------------------------------------------------------------
  // SPECIFICITY ORDER MATTERS: "gpt-4o-mini" must be tested before "gpt-4o".
  // We list the small/mid variants of bigger family names FIRST so they win
  // before the broader frontier/strong rules below.
  // -------------------------------------------------------------------------

  // ---- Mid-tier early hits (must precede their strong/frontier siblings) --
  { contains: ["gpt-4o-mini"], tier: "mid" },
  { contains: ["gpt-4.1-mini"], tier: "mid" },
  { contains: ["gpt-5-mini"], tier: "mid" },
  { contains: ["claude-3-haiku"], tier: "mid" },
  { contains: ["claude-3-5-haiku"], tier: "mid" },
  { contains: ["claude-3.5-haiku"], tier: "mid" },
  { contains: ["gemini-1.5-flash"], tier: "mid" },
  { contains: ["gemini-2.0-flash"], tier: "mid" },

  // ---- Small-tier early hits (must precede strong/mid family rules) -------
  { contains: ["gpt-4o-nano"], tier: "small" },
  { contains: ["gpt-5-nano"], tier: "small" },
  { contains: ["gemini-1.5-flash-8b"], tier: "small" },

  // ---- Frontier -----------------------------------------------------------
  // Closed-API frontier
  { contains: ["gpt-5"], tier: "frontier" },
  { contains: ["o1"], re: /\bo1\b/i, tier: "frontier" },
  { contains: ["o3"], re: /\bo3\b/i, tier: "frontier" },
  { contains: ["o4"], re: /\bo4\b/i, tier: "frontier" },
  { contains: ["claude-opus-4"], tier: "frontier" },
  { contains: ["claude-sonnet-4"], tier: "frontier" },
  { contains: ["claude-4"], tier: "frontier" },
  { contains: ["gemini-2.5-pro"], tier: "frontier" },
  { contains: ["gemini-3"], tier: "frontier" },
  { contains: ["grok-4"], tier: "frontier" },
  { contains: ["grok-3"], tier: "frontier" },
  // Open-weight frontier
  { contains: ["deepseek-v3"], tier: "frontier" },
  { contains: ["deepseek-r1"], tier: "frontier" },
  { contains: ["deepseek v3"], tier: "frontier" },
  { contains: ["deepseek r1"], tier: "frontier" },
  { contains: ["llama-3.1-405b"], tier: "frontier" },
  { contains: ["llama 3.1 405b"], tier: "frontier" },
  { contains: ["llama-4"], tier: "frontier" },
  { contains: ["llama 4"], tier: "frontier" },
  { contains: ["qwen3-235b"], tier: "frontier" },
  { contains: ["qwen3 235b"], tier: "frontier" },
  { contains: ["kimi-k2"], tier: "frontier" },
  { contains: ["minimax-m1"], tier: "frontier" },

  // ---- Strong -------------------------------------------------------------
  { contains: ["gpt-4o"], tier: "strong" },
  { contains: ["gpt-4.1"], tier: "strong" },
  { contains: ["gpt-4-turbo"], tier: "strong" },
  { contains: ["claude-3.5-sonnet"], tier: "strong" },
  { contains: ["claude-3-5-sonnet"], tier: "strong" },
  { contains: ["claude-3.7"], tier: "strong" },
  { contains: ["claude-3-opus"], tier: "strong" },
  { contains: ["gemini-1.5-pro"], tier: "strong" },
  { contains: ["gemini-2.0-pro"], tier: "strong" },
  { contains: ["gemini-2.5-flash"], tier: "strong" },
  { contains: ["mistral-large"], tier: "strong" },
  { contains: ["command-a"], tier: "strong" },
  { contains: ["command-r-plus"], tier: "strong" },
  // Open-weight strong (70B-class dense or large MoE)
  { contains: ["llama-3.3-70b"], tier: "strong" },
  { contains: ["llama 3.3 70b"], tier: "strong" },
  { contains: ["llama-3.1-70b"], tier: "strong" },
  { contains: ["llama 3.1 70b"], tier: "strong" },
  { contains: ["qwen2.5-72b"], tier: "strong" },
  { contains: ["qwen2.5 72b"], tier: "strong" },
  { contains: ["qwen3-coder-480b"], tier: "strong" },
  { contains: ["qwen3 coder 480b"], tier: "strong" },
  { contains: ["qwen3-32b"], tier: "strong" },
  { contains: ["qwq-32b"], tier: "strong" },
  { contains: ["mixtral-8x22b"], tier: "strong" },
  { contains: ["mixtral 8x22b"], tier: "strong" },
  { contains: ["nemotron-340b"], tier: "strong" },
  { contains: ["dbrx"], tier: "strong" },

  // ---- Mid ---------------------------------------------------------------
  { contains: ["mistral-small"], tier: "mid" },
  { contains: ["mistral-nemo"], tier: "mid" },
  { contains: ["command-r"], tier: "mid" },
  { contains: ["llama-3.1-8b"], tier: "mid" },
  { contains: ["llama 3.1 8b"], tier: "mid" },
  { contains: ["llama-3.2-11b"], tier: "mid" },
  { contains: ["qwen2.5-32b"], tier: "mid" },
  { contains: ["qwen2.5 32b"], tier: "mid" },
  { contains: ["qwen2.5-14b"], tier: "mid" },
  { contains: ["qwen2.5 14b"], tier: "mid" },
  { contains: ["qwen3-14b"], tier: "mid" },
  { contains: ["qwen3-30b"], tier: "mid" },
  { contains: ["gemma-2-27b"], tier: "mid" },
  { contains: ["gemma 2 27b"], tier: "mid" },
  { contains: ["gemma-3-27b"], tier: "mid" },
  { contains: ["phi-4"], tier: "mid" },
  { contains: ["phi 4"], tier: "mid" },
  { contains: ["yi-34b"], tier: "mid" },
  { contains: ["mixtral-8x7b"], tier: "mid" },

  // ---- Small --------------------------------------------------------------
  { contains: ["llama-3.2-1b"], tier: "small" },
  { contains: ["llama 3.2 1b"], tier: "small" },
  { contains: ["llama-3.2-3b"], tier: "small" },
  { contains: ["llama 3.2 3b"], tier: "small" },
  { contains: ["qwen2.5-7b"], tier: "small" },
  { contains: ["qwen2.5 7b"], tier: "small" },
  { contains: ["qwen2.5-3b"], tier: "small" },
  { contains: ["qwen3-7b"], tier: "small" },
  { contains: ["qwen3-8b"], tier: "small" },
  { contains: ["gemma-2-2b"], tier: "small" },
  { contains: ["gemma 2 2b"], tier: "small" },
  { contains: ["gemma-2-9b"], tier: "small" },
  { contains: ["gemma 2 9b"], tier: "small" },
  { contains: ["gemma-3-4b"], tier: "small" },
  { contains: ["gemma-3-12b"], tier: "small" },
  { contains: ["mistral-7b"], tier: "small" },
  { contains: ["mistral 7b"], tier: "small" },
  { contains: ["ministral-8b"], tier: "small" },
  { contains: ["ministral-3b"], tier: "small" },
  { contains: ["phi-3"], tier: "small" },
  { contains: ["phi 3"], tier: "small" },
  { contains: ["phi-3.5"], tier: "small" },
];

/**
 * Collapse `-`, `_`, `.`, and whitespace runs to a single space, lowercase.
 * "Claude-3.5-Sonnet", "Claude 3.5 Sonnet", and "claude_3_5_sonnet" all
 * normalize to the same string so rules can be written once.
 */
function normalize(s: string): string {
  return s.toLowerCase().replace(/[-_.\s]+/g, " ").trim();
}

/**
 * Look up a quality tier from a free-form name and optional id.
 * Returns null if no rule matches — caller can decide how to fall back.
 *
 * Rules are evaluated in order; the first match wins. More specific rules
 * (e.g. "gpt 4o mini") MUST appear before broader ones ("gpt 4o").
 */
export function tierFor(name: string, id?: string): QualityTier | null {
  const hay = normalize(`${name} ${id ?? ""}`);
  for (const rule of RULES) {
    let ok = true;
    for (const needle of rule.contains) {
      if (!hay.includes(normalize(needle))) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    if (rule.re && !rule.re.test(hay)) continue;
    return rule.tier;
  }
  return null;
}

/**
 * Coarse size-based fallback tier — used only when nothing in the rules
 * matches. This is a *very* rough heuristic: "bigger usually beats smaller
 * in the same generation". Don't trust it for cross-family comparisons.
 */
export function tierFromSize(params_b: number, active_b?: number | null): QualityTier {
  // For MoE, active params drive most of the quality signal.
  const eff = active_b && active_b > 0 ? Math.max(active_b, params_b * 0.15) : params_b;
  if (eff >= 60 || params_b >= 200) return "strong";
  if (eff >= 13) return "mid";
  return "small";
}

export function tierAtLeast(t: QualityTier, floor: QualityTier): boolean {
  return TIER_RANK[t] >= TIER_RANK[floor];
}

/** Tier one step below `t`, or the same tier if already at the bottom. */
export function tierMinusOne(t: QualityTier): QualityTier {
  switch (t) {
    case "frontier":
      return "strong";
    case "strong":
      return "mid";
    case "mid":
      return "small";
    case "small":
      return "small";
  }
}

/** Public URL where curators can read or PR the source. */
export const QUALITY_TIERS_SOURCE_URL =
  "https://github.com/artvandelay/should-i-self-host-llm/blob/main/src/qualityTiers.ts";
