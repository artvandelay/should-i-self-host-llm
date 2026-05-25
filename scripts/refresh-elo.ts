/**
 * CLI script: pull LMArena leaderboard ratings and write to src/elo.json.
 *
 * Source: the official `lmarena-ai/leaderboard-dataset` on Hugging Face,
 * published by the LMArena team. We fetch the `text` config, `latest`
 * split, and keep only rows where `category == "overall"` — i.e. the main
 * text-arena leaderboard at its most recent publish date.
 *
 * See src/eloMatch.ts for the rationale and schema.
 *
 * Run nightly via .github/workflows/refresh-prices.yml.
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { parquetReadObjects } from "hyparquet";
import { matchElo, type EloEntry, type EloSnapshot } from "../src/eloMatch";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, "..");
const SRC = resolve(ROOT, "src");
const ELO_PATH = resolve(SRC, "elo.json");
const PRICING_PATH = resolve(SRC, "pricing.json");
const KNOWN_MODELS_PATH = resolve(SRC, "knownModels.json");

const HF_DATASET = "lmarena-ai/leaderboard-dataset";
const HF_CONFIG = "text";
const HF_SPLIT = "latest";
const SOURCE_URL = `https://huggingface.co/datasets/${HF_DATASET}`;
// Direct parquet download from the HF resolve endpoint — one request,
// no rate-limit issues with the /rows API.
const PARQUET_URL =
  `https://huggingface.co/datasets/${HF_DATASET}/resolve/main/` +
  `${HF_CONFIG}/${HF_SPLIT}-00000-of-00001.parquet`;

interface HfRow {
  model_name: string;
  organization: string;
  license: string;
  rating: number;
  rating_lower?: number;
  rating_upper?: number;
  variance?: number;
  vote_count: number;
  rank: number;
  category: string;
  leaderboard_publish_date: string;
}

async function fetchParquet(): Promise<HfRow[]> {
  const resp = await fetch(PARQUET_URL, {
    headers: { "User-Agent": "Mozilla/5.0 (AI-PH-FT-calculation/1.0 refresh-elo)" },
    signal: AbortSignal.timeout(60_000),
  });
  if (!resp.ok) {
    throw new Error(`Parquet HTTP ${resp.status}`);
  }
  const buf = await resp.arrayBuffer();
  console.log(`  Downloaded ${(buf.byteLength / 1024).toFixed(0)} KB of parquet`);
  const rows = (await parquetReadObjects({ file: buf })) as unknown as HfRow[];
  return rows;
}

async function fetchLeaderboard(): Promise<EloSnapshot | null> {
  try {
    const all = await fetchParquet();
    console.log(`  Parsed ${all.length} rows from parquet`);

    // Keep only the main text-arena leaderboard (overall category).
    const overall = all.filter((r) => r.category === "overall");
    // Lock to the single most recent publish date (the "latest" split
    // sometimes contains a small trailing snapshot of the prior date).
    const latestDate = overall
      .map((r) => r.leaderboard_publish_date)
      .sort()
      .at(-1)!;
    const rows = overall.filter((r) => r.leaderboard_publish_date === latestDate);

    const entries: EloEntry[] = rows
      .sort((a, b) => a.rank - b.rank)
      .map((r) => {
        const isOpen = String(r.license).toLowerCase() !== "proprietary";
        return {
          // Arena uses lowercase, hyphenated model ids — matchElo already
          // normalises, but keep the source form for traceability.
          model: r.model_name,
          vendor: r.organization,
          license: isOpen ? "open" : "proprietary",
          score: Math.round(r.rating),
          rank: Math.round(r.rank),
          votes: Math.round(r.vote_count),
        };
      });

    return {
      last_updated: latestDate,
      source: SOURCE_URL,
      entries,
    };
  } catch (e) {
    console.error("LMArena fetch failed:", e);
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

  const openCount = snapshot.entries.filter((e) => e.license === "open").length;
  console.log(`\nArena ELO match audit:`);
  console.log(`  Snapshot: ${snapshot.entries.length} entries (${openCount} open-weight) on ${snapshot.last_updated}`);
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
  console.log(`Fetching LMArena leaderboard from ${SOURCE_URL} (${HF_CONFIG}/${HF_SPLIT}, category=overall)…`);
  const snap = await fetchLeaderboard();
  if (!snap || snap.entries.length === 0) {
    console.error("LMArena fetch failed or empty; leaving existing elo.json untouched.");
    process.exit(1);
  }
  console.log(`Fetched ${snap.entries.length} arena entries (publish_date=${snap.last_updated}).`);

  writeFileSync(ELO_PATH, JSON.stringify(snap, null, 2) + "\n");
  console.log(`Wrote ${ELO_PATH}`);
  auditMatches(snap);
}

main().catch((e) => {
  console.error("refresh-elo failed:", e);
  process.exit(1);
});
