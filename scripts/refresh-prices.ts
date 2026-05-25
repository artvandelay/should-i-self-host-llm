/**
 * CLI script run by the GitHub Action nightly.
 * Pulls models.dev for closed APIs + open-weight models,
 * scrapes GPU vendor pages via Firecrawl, and writes both JSONs.
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

import type { ModelsDevPayload, ClosedApi, OpenWeightModel, KnownModel } from "../src/modelsDev";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, "..");
const SRC = resolve(ROOT, "src");

const PRICING_PATH = resolve(SRC, "pricing.json");
const KNOWN_MODELS_PATH = resolve(SRC, "knownModels.json");

const FIRECRAWL_KEY = process.env.FIRECRAWL_API_KEY;
const MODELS_DEV_URL = process.env.MODELS_DEV_URL ?? "https://models.dev/api.json";

// Matches our Python module for portability
const AVG_TOKENS_PER_QUERY = 800;
const GPU_URLS: Record<string, string> = {
  Modal: "https://modal.com/pricing",
  Lambda: "https://lambdalabs.com/service/gpu-cloud",
  Runpod: "https://www.runpod.io/pricing",
};

// ---------------------------------------------------------------------------
// models.dev fetch
// ---------------------------------------------------------------------------

async function fetchModelsDev(): Promise<ModelsDevPayload | null> {
  try {
    const resp = await fetch(MODELS_DEV_URL, {
      headers: { "User-Agent": "Mozilla/5.0 (AI-PH-FT-calculation/1.0 refresh-script)" },
      signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) return null;
    return (await resp.json()) as ModelsDevPayload;
  } catch (e) {
    console.error("models.dev fetch failed:", e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Closed APIs extraction (duplicated from modelsDev.ts for Node CJS portability)
// ---------------------------------------------------------------------------

const FIRST_PARTY = new Set([
  "openai", "anthropic", "google", "deepseek", "mistral", "xai",
  "cohere", "cerebras", "groq", "perplexity", "nvidia", "alibaba",
  "zhipuai", "moonshotai", "minimax", "stepfun-ai", "upstage",
  "tencent-tokenhub", "sarvam", "fireworks-ai",
]);

function extractClosedApis(payload: ModelsDevPayload): ClosedApi[] {
  const results: ClosedApi[] = [];
  const seen = new Set<string>();
  for (const [pid, prov] of Object.entries(payload)) {
    if (!FIRST_PARTY.has(pid)) continue;
    const pn = prov.name ?? pid;
    for (const [mid, model] of Object.entries(prov.models ?? {})) {
      if (model.open_weights) continue;
      const c = model.cost;
      if (!c || c.input == null || c.output == null) continue;
      const label = `${pn} ${model.name ?? mid}`;
      if (seen.has(label)) continue;
      seen.add(label);
      results.push({
        label,
        input_per_1m: c.input,
        output_per_1m: c.output,
        last_seen: model.last_updated ?? "unknown",
      });
    }
  }
  return results;
}

function extractOpenWeightModels(payload: ModelsDevPayload): OpenWeightModel[] {
  const results: OpenWeightModel[] = [];
  const seen = new Set<string>();
  for (const [pid, prov] of Object.entries(payload)) {
    for (const [mid, model] of Object.entries(prov.models ?? {})) {
      if (!model.open_weights) continue;
      if (seen.has(mid)) continue;
      seen.add(mid);
      results.push({
        provider_id: pid,
        model_id: mid,
        name: model.name ?? mid,
        last_updated: model.last_updated ?? "unknown",
      });
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Params resolution (regex + HF fallback — Node can do it safely)
// ---------------------------------------------------------------------------

const MOE_RE = /(\d+(?:\.\d+)?)\s*[Bb]-?A\s*(\d+(?:\.\d+)?)\s*[Bb]?/i;
const MIL_RE = /(\d+(?:\.\d+)?)\s*[Mm]\b/;
const BIL_RE = /(\d+(?:\.\d+)?)\s*[Bb]\b/;

function resolveParamsB(name: string, modelId: string): { paramsB: number | null; activeB: number | null; method: string } {
  const s = `${name} ${modelId}`;
  const moe = MOE_RE.exec(s);
  if (moe) {
    const tb = parseFloat(moe[1]);
    const ab = parseFloat(moe[2]);
    if (Number.isFinite(tb) && Number.isFinite(ab)) return { paramsB: tb, activeB: ab, method: "regex_moe" };
  }
  const mm = MIL_RE.exec(s);
  if (mm) {
    const v = parseFloat(mm[1]) / 1000;
    if (Number.isFinite(v)) return { paramsB: v, activeB: null, method: "regex" };
  }
  const bm = BIL_RE.exec(s);
  if (bm) {
    const v = parseFloat(bm[1]);
    if (Number.isFinite(v)) return { paramsB: v, activeB: null, method: "regex" };
  }
  return { paramsB: null, activeB: null, method: "unknown" };
}

async function resolveParamsBWithHF(name: string, modelId: string): Promise<typeof resolveParamsB extends (...a: any) => infer R ? R : never> {
  const r = resolveParamsB(name, modelId);
  if (r.paramsB != null) return r;
  if (!modelId.includes("/")) return r;
  try {
    const resp = await fetch(`https://huggingface.co/api/models/${modelId}`, {
      headers: { "User-Agent": "Mozilla/5.0" },
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return r;
    const data = await resp.json();
    const total = data?.safetensors?.total;
    if (typeof total === "number" && total > 0) {
      return { paramsB: Math.round((total / 1_000_000_000) * 1e3) / 1e3, activeB: null, method: "hf" };
    }
  } catch {}
  return r;
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

function mergeKnownModels(fresh: KnownModel[], cached: KnownModel[]): KnownModel[] {
  const map = new Map<string, KnownModel>();
  for (const m of cached) map.set(m.name.toLowerCase(), m);
  for (const m of fresh) {
    const key = m.name.toLowerCase();
    const ex = map.get(key);
    map.set(key, ex
      ? { ...m, active_b: ex.active_b ?? m.active_b, source: m.source, last_seen: m.last_seen ?? ex.last_seen }
      : m);
  }
  return [...map.values()].sort((a, b) => (a.params_b ?? Infinity) - (b.params_b ?? Infinity));
}

// ---------------------------------------------------------------------------
// Firecrawl GPU scrape
// ---------------------------------------------------------------------------

async function firecrawlScrape(url: string): Promise<string> {
  if (!FIRECRAWL_KEY) throw new Error("FIRECRAWL_API_KEY not set");
  const resp = await fetch("https://api.firecrawl.dev/v1/scrape", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${FIRECRAWL_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url, formats: ["markdown"], onlyMainContent: true }),
  });
  if (!resp.ok) throw new Error(`Firecrawl HTTP ${resp.status} for ${url}`);
  const data = await resp.json();
  if (!data.success) throw new Error(`Firecrawl failed for ${url}: ${JSON.stringify(data)}`);
  return data.data?.markdown ?? "";
}

function extractGpuRate(gpuName: string, markdown: string): number | null {
  const raw = gpuName.replace(/^\d+x\s+/i, "").trim();
  const re = new RegExp(raw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  let best: number | null = null;
  let bestDist = Infinity;

  for (const match of markdown.matchAll(new RegExp(re.source, "gi"))) {
    const start = Math.max(0, match.index! - 200);
    const end = Math.min(markdown.length, match.index! + match[0].length + 300);
    const window = markdown.slice(start, end);
    for (const rm of window.matchAll(/\$\s*([\d,]+\.?\d*)\s*\/?\s*hr/gi)) {
      const rate = parseFloat(rm[1].replace(/,/g, ""));
      if (!Number.isFinite(rate)) continue;
      const dist = Math.abs(match.index! - (start + rm.index!));
      if (dist < bestDist) { bestDist = dist; best = rate; }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

interface GpuEntry {
  name: string; vram_gb: number;
  modal_per_hr: number; lambda_per_hr: number; runpod_per_hr: number;
}

interface ApiEntry {
  label: string; input_per_1m: number; output_per_1m: number;
}

async function main() {
  console.log("Fetching models.dev …");
  const payload = await fetchModelsDev();
  if (!payload) {
    console.error("models.dev fetch failed; exiting");
    process.exit(1);
  }

  // Closed APIs
  const closedApis = extractClosedApis(payload);
  console.log(`Extracted ${closedApis.length} closed API models`);

  // Open-weight models
  const rawModels = extractOpenWeightModels(payload);
  console.log(`Found ${rawModels.length} open-weight models. Resolving sizes…`);

  const freshKnown: KnownModel[] = [];
  for (const m of rawModels) {
    const r = await resolveParamsBWithHF(m.name, m.model_id);
    if (r.paramsB == null) continue;
    freshKnown.push({
      params_b: r.paramsB,
      active_b: r.activeB,
      name: m.name,
      arch: r.method === "regex_moe" ? "moe" : "dense",
      source: `models.dev:${r.method}`,
      last_seen: m.last_updated,
    });
  }
  console.log(`Resolved ${freshKnown.length} models`);

  // Load cached
  let cachedKnown: KnownModel[] = [];
  try { cachedKnown = JSON.parse(readFileSync(KNOWN_MODELS_PATH, "utf-8")); } catch {}

  const mergedModels = mergeKnownModels(freshKnown, cachedKnown);
  writeFileSync(KNOWN_MODELS_PATH, JSON.stringify(mergedModels, null, 2) + "\n");
  console.log(`Wrote ${mergedModels.length} models to knownModels.json`);

  // GPU scrape
  const gpuRows: GpuEntry[] = [];
  if (FIRECRAWL_KEY) {
    const markdowns: Record<string, string> = {};
    for (const [label, url] of Object.entries(GPU_URLS)) {
      try {
        markdowns[label] = await firecrawlScrape(url);
        console.log(`Scraped ${label}`);
      } catch (e) {
        console.error(`Firecrawl ${label} failed:`, e);
        markdowns[label] = "";
      }
    }

    // These are the GPU entries we ship; extract rates from the scraped markdown
    const gpuDefs = [
      { name: "L4 24GB", vram: 24 },
      { name: "L40S 48GB", vram: 48 },
      { name: "A100 40GB", vram: 40 },
      { name: "A100 80GB", vram: 80 },
      { name: "H100 80GB", vram: 80 },
      { name: "2xH100 160GB", vram: 160 },
      { name: "4xH100 320GB", vram: 320 },
      { name: "8xH100 640GB", vram: 640 },
      { name: "8xH200 1128GB", vram: 1128 },
    ];

    // Load cached rows so we can fall back when regex extraction returns 0/null.
    let cachedGpus: GpuEntry[] = [];
    try {
      const existing = JSON.parse(readFileSync(PRICING_PATH, "utf-8"));
      cachedGpus = existing.gpus ?? [];
    } catch {}
    const cachedByName = new Map(cachedGpus.map((g) => [g.name, g]));

    let extracted = 0;
    let preserved = 0;
    for (const gd of gpuDefs) {
      const cached = cachedByName.get(gd.name);
      const modalFresh = extractGpuRate(gd.name, markdowns["Modal"] ?? "");
      const lambdaFresh = extractGpuRate(gd.name, markdowns["Lambda"] ?? "");
      const runpodFresh = extractGpuRate(gd.name, markdowns["Runpod"] ?? "");

      const modal = modalFresh && modalFresh > 0 ? modalFresh : cached?.modal_per_hr ?? 0;
      const lambda = lambdaFresh && lambdaFresh > 0 ? lambdaFresh : cached?.lambda_per_hr ?? 0;
      const runpod = runpodFresh && runpodFresh > 0 ? runpodFresh : cached?.runpod_per_hr ?? 0;

      if (modalFresh && lambdaFresh && runpodFresh) extracted++;
      else preserved++;

      gpuRows.push({ name: gd.name, vram_gb: gd.vram, modal_per_hr: modal, lambda_per_hr: lambda, runpod_per_hr: runpod });
    }
    console.log(`GPU rates: ${extracted} fully extracted, ${preserved} partially-or-fully preserved from cache`);
  } else {
    console.log("FIRECRAWL_API_KEY not set; keeping cached GPU prices");
    try {
      const existing = JSON.parse(readFileSync(PRICING_PATH, "utf-8"));
      for (const g of existing.gpus ?? []) gpuRows.push(g);
    } catch {}
  }

  // Build final pricing.json
  const today = new Date().toISOString().slice(0, 10);
  const apiMap: Record<string, ApiEntry> = {};
  for (const a of closedApis) {
    const key = a.label.toLowerCase().replace(/[^a-z0-9]/g, "_");
    apiMap[key] = { label: a.label, input_per_1m: a.input_per_1m, output_per_1m: a.output_per_1m };
  }
  // Always keep custom
  apiMap["custom"] = { label: "Custom (enter rates)", input_per_1m: 1.0, output_per_1m: 3.0 };

  const pricing = {
    last_updated: today,
    gpu_last_updated: today,
    gpus: gpuRows,
    apis: apiMap,
  };

  writeFileSync(PRICING_PATH, JSON.stringify(pricing, null, 2) + "\n");
  console.log("Wrote pricing.json");
}

main().catch((e) => {
  console.error("Refresh failed:", e);
  process.exit(1);
});