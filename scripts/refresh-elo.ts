/**
 * CLI script: pull LMArena (Chatbot Arena) ELO scores and write to src/elo.json.
 *
 * Source: api.wulong.dev (free, no-auth mirror of arena.ai/leaderboard/text).
 * See src/eloMatch.ts for the rationale behind this source choice.
 *
 * Run nightly via .github/workflows/refresh-prices.yml.
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { matchElo, type EloEntry, type EloSnapshot } from "../src/eloMatch";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, "..");
const SRC = resolve(ROOT, "src");
const ELO_PATH = resolve(SRC, "elo.json");
const PRICING_PATH = resolve(SRC, "pricing.json");
const KNOWN_MODELS_PATH = resolve(SRC, "knownModels.json");

const ARENA_URL =
  process.env.ARENA_LEADERBOARD_URL ??
  "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=text";

async function fetchArena(): Promise<EloSnapshot | null> {
  try {
    const resp = await fetch(ARENA_URL, {
      headers: { "User-Agent": "Mozilla/5.0 (AI-PH-FT-calculation/1.0 refresh-elo)" },
      signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) {
      console.error("Arena fetch HTTP", resp.status);
      return null;
    }
    const data: any = await resp.json();
    const entries: EloEntry[] = (data.models ?? []).map((m: any) => ({
      model: m.model,
      vendor: m.vendor,
      license: m.license,
      score: m.score,
      rank: m.rank,
      votes: m.votes,
    }));
    return {
      last_updated: data.meta?.last_updated ?? new Date().toISOString().slice(0, 10),
      source: data.meta?.source_url ?? ARENA_URL,
      entries,
    };
  } catch (e) {
    console.error("Arena fetch failed:", e);
    return null;
  }
}

function auditMatches(snapshot: EloSnapshot) {
  // Audit-log: which pricing.json APIs and knownModels.json entries get an ELO?
  let pricing: any = {};
  let known: any[] = [];
  try { pricing = JSON.parse(readFileSync(PRICING_PATH, "utf-8")); } catch {}
  try { known = JSON.parse(readFileSync(KNOWN_MODELS_PATH, "utf-8")); } catch {}

  const apiLabels: string[] = Object.values(pricing.apis ?? {}).map((a: any) => a.label);
  let apiMatched = 0;
  const apiUnmatched: string[] = [];
  for (const label of apiLabels) {
    const hit = matchElo(label, snapshot.entries);
    if (hit) apiMatched++; else apiUnmatched.push(label);
  }

  let modelMatched = 0;
  const modelUnmatched: string[] = [];
  for (const m of known) {
    const hit = matchElo(m.name, snapshot.entries);
    if (hit) modelMatched++; else modelUnmatched.push(m.name);
  }

  const matchedArenaIds = new Set<string>();
  for (const label of apiLabels) {
    const h = matchElo(label, snapshot.entries);
    if (h) matchedArenaIds.add(h.model);
  }
  for (const m of known) {
    const h = matchElo(m.name, snapshot.entries);
    if (h) matchedArenaIds.add(h.model);
  }
  const orphanArena = snapshot.entries.filter((e) => !matchedArenaIds.has(e.model));

  console.log(`\nArena ELO match audit:`);
  console.log(`  APIs:   ${apiMatched}/${apiLabels.length} matched`);
  console.log(`  Models: ${modelMatched}/${known.length} matched`);
  console.log(`  Unmatched arena entries (no API/model carries them): ${orphanArena.length}`);
  if (orphanArena.length) {
    console.log(`    e.g. ${orphanArena.slice(0, 5).map((e) => e.model).join(", ")}`);
  }
  if (apiUnmatched.length && apiUnmatched.length <= 20) {
    console.log(`  Sample unmatched APIs: ${apiUnmatched.slice(0, 10).join(" | ")}`);
  }
  if (modelUnmatched.length && modelUnmatched.length <= 20) {
    console.log(`  Sample unmatched models: ${modelUnmatched.slice(0, 10).join(" | ")}`);
  }
}

async function main() {
  console.log(`Fetching LMArena leaderboard from ${ARENA_URL} …`);
  const snap = await fetchArena();
  if (!snap) {
    console.error("Arena fetch failed; leaving existing elo.json untouched.");
    process.exit(1);
  }
  console.log(`Fetched ${snap.entries.length} arena entries (last_updated=${snap.last_updated}).`);

  writeFileSync(ELO_PATH, JSON.stringify(snap, null, 2) + "\n");
  console.log(`Wrote ${ELO_PATH}`);
  auditMatches(snap);
}

main().catch((e) => {
  console.error("refresh-elo failed:", e);
  process.exit(1);
});
