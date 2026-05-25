/**
 * LMArena ELO matcher.
 *
 * Source of ELO data:
 *   https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=text
 *
 * Why this source:
 *   - LMArena (formerly LMSYS Chatbot Arena) does NOT provide an official
 *     public API for the leaderboard. The closest official artifact is
 *     `lmarena/arena-catalog` on GitHub, but it only carries model metadata
 *     (pricing, license) — not the actual ELO scores; scores live in an
 *     internal `results.pkl` the LMArena team owns.
 *   - `api.wulong.dev/arena-ai-leaderboards` is a free, no-auth REST mirror
 *     backed by a daily-updated GitHub repo
 *     (https://github.com/oolong-tea-2026/arena-ai-leaderboards) that scrapes
 *     the public arena.ai leaderboard page and exposes it as structured JSON.
 *   - That repo covers the top ~30 text-arena entries — enough for every
 *     proprietary API we ship and for the headline open-weight models
 *     (GLM, Kimi, DeepSeek, etc). Smaller open-weight models that don't
 *     appear in the top 30 simply get no ELO; the app degrades gracefully.
 *
 * License/attribution: arena scores are derived from LMArena.ai user votes.
 * We credit LMArena (and the wulong.dev mirror) in the README. Re-publishing
 * the scores as a derived dataset is consistent with the source repo's MIT
 * license.
 */

export interface EloEntry {
  /** Canonical arena model id, e.g. "claude-opus-4-6-thinking". */
  model: string;
  vendor: string;
  license: "open" | "proprietary" | string;
  score: number;
  rank: number;
  votes?: number;
}

export interface EloSnapshot {
  last_updated: string;
  source: string;
  entries: EloEntry[];
}

/** Normalise a name for fuzzy comparison: lowercase, strip non-alnum, collapse. */
export function normaliseName(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/**
 * Tokenise into alnum runs (so "claude-opus-4-6" -> ["claude","opus","4","6"]).
 * Used for token-overlap scoring.
 */
function tokenise(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

/** Jaccard similarity over token sets. */
function jaccard(a: string[], b: string[]): number {
  const sa = new Set(a);
  const sb = new Set(b);
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  const uni = sa.size + sb.size - inter;
  return uni === 0 ? 0 : inter / uni;
}

/**
 * Pull the multiset of numeric tokens out of a raw string, normalising
 * "4.6" / "4-6" / "4_6" into the multi-token form ["4","6"], and bare
 * integers like "5" into ["5"]. We compare these as a SET — every numeric
 * token on one side must appear on the other.
 *
 * Why this strict rule: arena model IDs use hyphens for version segments
 * (`claude-opus-4-6`) while our label uses dots (`Claude Opus 4.6`). After
 * splitting on non-alphanumeric, both yield the same digit set `{4,6}`.
 * A different version like `Claude Opus 4.1` yields `{4,1}` and is
 * correctly rejected.
 *
 * Side-effect we accept: a label with no version digits (e.g. "GLM-5.1"
 * yielding `{5,1}` and arena `glm-5.1` yielding `{5,1}`) matches; but
 * "GPT-5 Nano" (`{5}`) does NOT match `gpt-5.5` (`{5,5}` -> set `{5}`)
 * because set equality is required. (Multiset would be better but
 * version tokens are short and the false-positive rate of set-equality
 * is acceptable.)
 */
function numericTokens(raw: string): string[] {
  // Split on non-alnum, keep tokens that are entirely digits.
  return raw
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => /^\d+$/.test(t));
}

function multisetEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  for (let i = 0; i < sa.length; i++) if (sa[i] !== sb[i]) return false;
  return true;
}

/**
 * Versions agree iff their digit-token MULTISETS match exactly (so
 * `gpt-5.5` -> [5,5] and `gpt-5` -> [5] are correctly distinguished),
 * OR one side has no digit tokens at all (nothing to contradict).
 *
 * We also accept the case where one side's digits are a strict subset
 * of the other AND every digit on the shorter side appears on the
 * longer side — this lets "Claude Opus 4.6" match `claude-opus-4-6`
 * even if upstream adds a date suffix like `claude-opus-4-6-20260520`
 * (digits would become [4,6,20260520], still containing [4,6]). To
 * avoid over-matching, require the longer side's *extra* digits to be
 * "date-like" (>=4 digits, e.g. years or YYYYMMDD).
 */
function versionsAgree(rawA: string, rawB: string): boolean {
  const va = numericTokens(rawA);
  const vb = numericTokens(rawB);
  if (va.length === 0 || vb.length === 0) return true;
  if (multisetEqual(va, vb)) return true;
  const [short, long] = va.length <= vb.length ? [va, vb] : [vb, va];
  const longCopy = [...long];
  for (const t of short) {
    const idx = longCopy.indexOf(t);
    if (idx === -1) return false;
    longCopy.splice(idx, 1);
  }
  // All leftover tokens on the long side must look like dates/build IDs.
  return longCopy.every((t) => t.length >= 4);
}

/**
 * Match a model label/name against a list of ELO entries.
 *
 * Strategy (in order):
 *   1. Exact normalised-name equality.
 *   2. Normalised-substring containment either way (handles
 *      "Anthropic Claude Opus 4.6" vs "claude-opus-4-6").
 *   3. Token-overlap (Jaccard) >= 0.5 — chooses the highest overlap.
 *
 * Returns the best match or null. The threshold is deliberately
 * conservative so we don't claim "GPT-3.5" matches "GPT-5".
 */
export function matchElo(
  label: string,
  entries: EloEntry[]
): EloEntry | null {
  if (!label || entries.length === 0) return null;

  const labelNorm = normaliseName(label);
  const labelTokens = tokenise(label);

  // Pass 1: exact normalised equality
  for (const e of entries) {
    if (normaliseName(e.model) === labelNorm) return e;
  }

  // Pass 2: substring containment, preferring longer entry-name match.
  // Still version-gated: "gpt5" should not gobble up "gpt54".
  let bestSub: EloEntry | null = null;
  let bestSubLen = 0;
  for (const e of entries) {
    const en = normaliseName(e.model);
    if (en.length < 4) continue;
    if (labelNorm.includes(en) || en.includes(labelNorm)) {
      if (!versionsAgree(label, e.model)) continue;
      if (en.length > bestSubLen) {
        bestSub = e;
        bestSubLen = en.length;
      }
    }
  }
  if (bestSub) return bestSub;

  // Pass 3: token Jaccard, version-gated.
  let bestTok: EloEntry | null = null;
  let bestScore = 0;
  for (const e of entries) {
    if (!versionsAgree(label, e.model)) continue;
    const eTokens = tokenise(e.model);
    const s = jaccard(labelTokens, eTokens);
    if (s > bestScore) {
      bestScore = s;
      bestTok = e;
    }
  }
  if (bestTok && bestScore >= 0.5) return bestTok;
  return null;
}

/**
 * Attach `elo` and `eloRank` fields to a list of items by matching their
 * `name`/`label` against the snapshot. Items that don't match are left
 * untouched. Returns the list of unmatched labels for auditing.
 */
export function attachElo<T extends { name?: string; label?: string; elo?: number; eloRank?: number }>(
  items: T[],
  snapshot: EloSnapshot
): { unmatched: string[]; matched: number } {
  const unmatched: string[] = [];
  let matched = 0;
  for (const it of items) {
    const key = it.name ?? it.label ?? "";
    if (!key) continue;
    const hit = matchElo(key, snapshot.entries);
    if (hit) {
      it.elo = hit.score;
      it.eloRank = hit.rank;
      matched++;
    } else {
      unmatched.push(key);
    }
  }
  return { unmatched, matched };
}
