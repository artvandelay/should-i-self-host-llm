/** Live data from models.dev -- free, public, no-key-needed JSON endpoint. */

export interface ModelsDevProvider {
  name: string;
  env: string[];
  npm: string;
  doc: string;
  models: Record<string, ModelsDevModel>;
}

export interface ModelsDevModel {
  id: string;
  name: string;
  family: string;
  open_weights: boolean;
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
  };
  last_updated?: string;
}

export type ModelsDevPayload = Record<string, ModelsDevProvider>;

export interface ClosedApi {
  label: string;
  input_per_1m: number;
  output_per_1m: number;
  last_seen: string;
}

export interface OpenWeightModel {
  provider_id: string;
  model_id: string;
  name: string;
  last_updated: string;
}

export interface KnownModel {
  params_b: number;
  active_b: number | null;
  name: string;
  arch: "dense" | "moe";
  source: string;
  last_seen: string;
}

export type ParamsMethod = "regex_moe" | "regex" | "hf" | "unknown";

export interface ResolvedParams {
  paramsB: number | null;
  method: ParamsMethod;
  activeB?: number | null;
}

// First-party providers for the closed-API dropdown.
// models.dev lists 100+ providers (wrappers, gateways, etc.).
// We only surface actual labs so the dropdown is clean.
const FIRST_PARTY_PROVIDERS = new Set([
  "openai", "anthropic", "google", "deepseek", "mistral", "xai",
  "cohere", "cerebras", "groq", "perplexity", "nvidia", "alibaba",
  "zhipuai", "moonshotai", "minimax", "stepfun-ai", "upstage",
  "tencent-tokenhub", "sarvam", "fireworks-ai",
]);

const MODELS_DEV_URL = (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_MODELS_DEV_URL)
  ? (import.meta as any).env.VITE_MODELS_DEV_URL
  : "https://models.dev/api.json";

// =============================================================================
// fetchModelsDev
// =============================================================================

export async function fetchModelsDev(timeoutMs = 15_000): Promise<ModelsDevPayload | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(MODELS_DEV_URL, {
      signal: controller.signal,
      headers: { "User-Agent": "Mozilla/5.0 (AI-PH-FT-calculation/1.0)" },
    });
    if (!resp.ok) return null;
    return (await resp.json()) as ModelsDevPayload;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// =============================================================================
// extractClosedApis
// =============================================================================

export function extractClosedApis(payload: ModelsDevPayload): ClosedApi[] {
  const results: ClosedApi[] = [];
  const seen = new Set<string>();

  for (const [providerId, provider] of Object.entries(payload)) {
    if (!FIRST_PARTY_PROVIDERS.has(providerId)) continue;
    const providerName = provider.name ?? providerId;
    for (const [modelId, model] of Object.entries(provider.models ?? {})) {
      if (model.open_weights) continue;
      const cost = model.cost;
      if (!cost || cost.input == null || cost.output == null) continue;
      const label = `${providerName} ${model.name ?? modelId}`;
      if (seen.has(label)) continue;
      seen.add(label);
      results.push({
        label,
        input_per_1m: cost.input,
        output_per_1m: cost.output,
        last_seen: model.last_updated ?? "unknown",
      });
    }
  }

  results.sort((a, b) => a.label.localeCompare(b.label));
  return results;
}

// =============================================================================
// extractOpenWeightModels
// =============================================================================

export function extractOpenWeightModels(payload: ModelsDevPayload): OpenWeightModel[] {
  const results: OpenWeightModel[] = [];
  const seen = new Set<string>();

  for (const [providerId, provider] of Object.entries(payload)) {
    for (const [modelId, model] of Object.entries(provider.models ?? {})) {
      if (!model.open_weights) continue;
      if (seen.has(modelId)) continue;
      seen.add(modelId);
      results.push({
        provider_id: providerId,
        model_id: modelId,
        name: model.name ?? modelId,
        last_updated: model.last_updated ?? "unknown",
      });
    }
  }

  return results;
}

// =============================================================================
// resolveParamsB -- hybrid: MoE regex -> M/B regex -> unknown.
// HF fallback only in Node (the GH Action pre-resolves and caches).
// =============================================================================

const MOE_RE = /(\d+(?:\.\d+)?)\s*[Bb]-?A\s*(\d+(?:\.\d+)?)\s*[Bb]?/i;
const MILLION_RE = /(\d+(?:\.\d+)?)\s*[Mm]\b/;
const BILLION_RE = /(\d+(?:\.\d+)?)\s*[Bb]\b/;

export function resolveParamsB(name: string, modelId: string): ResolvedParams {
  const search = `${name} ${modelId}`;

  const moe = MOE_RE.exec(search);
  if (moe) {
    const totalB = parseFloat(moe[1]);
    const activeB = parseFloat(moe[2]);
    if (Number.isFinite(totalB) && Number.isFinite(activeB)) {
      return { paramsB: totalB, activeB, method: "regex_moe" };
    }
  }

  const mMatch = MILLION_RE.exec(search);
  if (mMatch) {
    const val = parseFloat(mMatch[1]) / 1000;
    if (Number.isFinite(val)) return { paramsB: val, method: "regex" };
  }

  const bMatch = BILLION_RE.exec(search);
  if (bMatch) {
    const val = parseFloat(bMatch[1]);
    if (Number.isFinite(val)) return { paramsB: val, method: "regex" };
  }

  return { paramsB: null, method: "unknown" };
}

// Only callable from Node (GH Action / CLI scripts).
export async function resolveParamsBNode(name: string, modelId: string): Promise<ResolvedParams> {
  const regexResult = resolveParamsB(name, modelId);
  if (regexResult.paramsB != null) return regexResult;
  if (!modelId.includes("/")) return regexResult;

  try {
    const resp = await fetch(`https://huggingface.co/api/models/${modelId}`, {
      headers: { "User-Agent": "Mozilla/5.0" },
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return regexResult;
    const data = await resp.json();
    const total = data?.safetensors?.total;
    if (typeof total === "number" && total > 0) {
      return { paramsB: Math.round((total / 1_000_000_000) * 1e3) / 1e3, method: "hf" };
    }
  } catch {
    // silently degrade
  }

  return regexResult;
}

// =============================================================================
// mergeKnownModels -- union by name; fresh overrides stale; manual kept.
// =============================================================================

export function mergeKnownModels(fresh: KnownModel[], cached: KnownModel[]): KnownModel[] {
  const map = new Map<string, KnownModel>();

  for (const m of cached) map.set(m.name.toLowerCase(), m);

  for (const m of fresh) {
    if (m.params_b == null) continue;
    const key = m.name.toLowerCase();
    const existing = map.get(key);
    if (existing) {
      map.set(key, {
        ...m,
        active_b: existing.active_b ?? m.active_b,
        source: m.source,
        last_seen: m.last_seen ?? existing.last_seen,
      });
    } else {
      map.set(key, m);
    }
  }

  return [...map.values()].sort((a, b) => {
    if (a.params_b == null) return 1;
    if (b.params_b == null) return -1;
    return a.params_b - b.params_b;
  });
}