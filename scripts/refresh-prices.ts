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

// Family-based filter — much simpler than regex whack-a-mole.
// Any model whose family is in this blocklist is rejected immediately.
const NON_LLM_FAMILIES = new Set([
  "text-embedding", "cohere-embed", "bge",
  "whisper", "speech-to-text", "text-to-speech",
  "flux", "stable-diffusion",
  "pixtral", "osmosis", "voxtral",
  "codestral", "devstral",         // Mistral code-only models
  "codellama",                     // code-only
]);

// A tiny fallback for the ~329 entries where models.dev has family: undefined
// and the name still clearly indicates a non-LLM.
// This list is short and rarely needs changes.
const NON_LLM_NAME_WORDS = new Set([
  "embedding", "embed", "retriever", "retrieval", "reranker", "rerank",
  "guard", "moderation", "toxic", "content-safety", "safety",
  "vision", "paddleocr", "ocr", "vl",
  "whisper", "tts", "asr",
  "flux", "sdxl", "stable-diffusion", "imagen",
  "esm", "protein", "biomed",
  "codellama", "pixtral",
]);

function deriveFamily(name: string, modelId: string, rawFamily?: string): string {
  if (rawFamily) return rawFamily;
  const tokens = (`${name} ${modelId}`.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
  const families = new Map([
    ["llama", ["llama", "l3"]],
    ["qwen", ["qwen", "qwq"]],
    ["gemma", ["gemma"]],
    ["mistral", ["mistral", "mixtral", "magistral", "ministral"]],
    ["phi", ["phi"]],
    ["deepseek", ["deepseek"]],
    ["kimi", ["kimi"]],
    ["glm", ["glm"]],
    ["nemotron", ["nemotron"]],
    ["falcon", ["falcon"]],
    ["yi", ["yi"]],
    ["command", ["command", "command-r", "command-a"]],
    ["internlm", ["internlm"]],
    ["gpt-oss", ["gpt-oss"]],
    ["hermes", ["hermes"]],
    ["granite", ["granite"]],
    ["olmo", ["olmo"]],
    ["solar", ["solar"]],
    ["baichuan", ["baichuan"]],
    ["jais", ["jais"]],
    ["mpt", ["mpt"]],
    ["dbrx", ["dbrx"]],
    ["openchat", ["openchat"]],
    ["stablelm", ["stablelm"]],
    ["jamba", ["jamba"]],
  ]);
  for (const [fam, aliases] of families) {
    for (const alias of aliases) {
      if (tokens.includes(alias)) return fam;
    }
  }
  return "other";
}

function isTextLLM(name: string, modelId: string, rawFamily?: string): boolean {
  const family = deriveFamily(name, modelId, rawFamily);
  if (NON_LLM_FAMILIES.has(family)) return false;
  const haystack = `${name} ${modelId}`.toLowerCase();
  const tokens = haystack.split(/[^a-z0-9]+/).filter(Boolean);
  for (const w of NON_LLM_NAME_WORDS) {
    // exact-token match OR substring within a token (catches qwen3guard, etc.)
    for (const t of tokens) {
      if (t === w || t.includes(w)) return false;
    }
  }
  return true;
}

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
      if (!isTextLLM(model.name ?? mid, mid, model.family)) continue;
      results.push({
        provider_id: pid,
        model_id: mid,
        name: model.name ?? mid,
        last_updated: model.last_updated ?? "unknown",
        family: model.family,
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
      ? { ...m, active_b: ex.active_b ?? m.active_b, family: m.family ?? ex.family, source: m.source, last_seen: m.last_seen ?? ex.last_seen }
      : m);
  }
  return [...map.values()].sort((a, b) => (a.params_b ?? Infinity) - (b.params_b ?? Infinity));
}

/**
 * Deprecation: for every (family, size-in-B) bucket, keep only the newest
 * model by last_seen. This eliminates stale versions like Llama 3.1 when
 * Llama 4 is present.
 */
function deprecateByFamilySize(models: KnownModel[]): KnownModel[] {
  const bucket = new Map<string, KnownModel[]>(); // "family|size" -> models[]

  for (const m of models) {
    const fam = m.family || "other";
    const size = Math.round(m.params_b); // round to nearest integer B
    const key = `${fam}|${size}`;
    const arr = bucket.get(key) ?? [];
    arr.push(m);
    bucket.set(key, arr);
  }

  const kept: KnownModel[] = [];
  for (const arr of bucket.values()) {
    arr.sort((a, b) => (b.last_seen ?? "0").localeCompare(a.last_seen ?? "0"));
    const newest = arr[0];
    kept.push(newest);
    if (arr.length > 1) {
      console.log(`  Deprecated ${arr.length - 1} older ${newest.family ?? "other"} ${Math.round(newest.params_b)}B models, keeping "${newest.name}" (${newest.last_seen})`);
    }
  }

  return kept.sort((a, b) => a.params_b - b.params_b);
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
  // Optional FT-engine fields, hand-maintained in pricing.json; preserved
  // across refreshes (this script never scrapes them).
  bf16_tflops?: number;
  single_gpu_vram_gb?: number;
  gpus_per_node?: number;
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
      family: deriveFamily(m.name, m.model_id, m.family),
    });
  }
  console.log(`Resolved ${freshKnown.length} models`);

  // Load cached — drop stale non-LLMs, keep manual overrides
  let cachedKnown: KnownModel[] = [];
  try { cachedKnown = JSON.parse(readFileSync(KNOWN_MODELS_PATH, "utf-8")); } catch {}
  const cleanCached = cachedKnown.filter(
    (m) => isTextLLM(m.name, "", m.family) || m.manual === true
  );

  let mergedModels = mergeKnownModels(freshKnown, cleanCached);
  console.log(`Merged ${mergedModels.length} models; deprecating stale family/size variants…`);
  mergedModels = deprecateByFamilySize(mergedModels);
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

      // Preserve hand-maintained FT-engine fields (bf16_tflops,
      // single_gpu_vram_gb, gpus_per_node) — these don't change with vendor
      // pricing and would silently disappear if we only wrote the four
      // scraped fields.
      const row: GpuEntry = {
        name: gd.name,
        vram_gb: gd.vram,
        modal_per_hr: modal,
        lambda_per_hr: lambda,
        runpod_per_hr: runpod,
      };
      if (cached?.bf16_tflops !== undefined) row.bf16_tflops = cached.bf16_tflops;
      if (cached?.single_gpu_vram_gb !== undefined)
        row.single_gpu_vram_gb = cached.single_gpu_vram_gb;
      if (cached?.gpus_per_node !== undefined)
        row.gpus_per_node = cached.gpus_per_node;
      gpuRows.push(row);
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