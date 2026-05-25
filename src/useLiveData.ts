import { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchModelsDev,
  extractClosedApis,
  extractOpenWeightModels,
  resolveParamsB,
  mergeKnownModels,
  type ClosedApi,
  type KnownModel,
} from "./modelsDev";

export interface DataStatus {
  apis: "idle" | "loading" | "live" | "cached";
  apisError?: string;
  apisFetchedAt?: string;
  models: "idle" | "loading" | "live" | "cached";
  modelsError?: string;
  modelsFetchedAt?: string;
  gpus: string; // e.g. "bundled 3h ago" or "cached"
}

export interface LiveData {
  /** Merged closed-API list (bundled + live). */
  apis: ClosedApi[];
  /** Merged known models (bundled + live). */
  knownModels: KnownModel[];
  /** Refresh status for the footer / banner. */
  status: DataStatus;
  /** Manual trigger. */
  refresh: () => void;
}

/** Default (bundled) data read from static imports. Caller passes them in. */
export function useLiveData(
  bundledApis: ClosedApi[],
  bundledKnownModels: KnownModel[],
  gpuLastUpdated: string
): LiveData {
  const [apis, setApis] = useState<ClosedApi[]>(bundledApis);
  const [knownModels, setKnownModels] = useState<KnownModel[]>(bundledKnownModels);
  const [status, setStatus] = useState<DataStatus>({
    apis: "idle",
    models: "idle",
    gpus: `bundled (${gpuLastUpdated})`,
  });
  const refreshId = useRef(0);

  const refresh = useCallback(async () => {
    const id = ++refreshId.current;
    setStatus((s) => ({ ...s, apis: "loading", models: "loading" }));

    const payload = await fetchModelsDev();
    if (payload) {
      const now = new Date().toISOString().slice(0, 19).replace("T", " ");

      // Closed APIs
      try {
        const freshApis = extractClosedApis(payload);
        const mergedApis = mergeApiLists(bundledApis, freshApis);
        if (id === refreshId.current) {
          setApis(mergedApis);
          setStatus((s) => ({
            ...s,
            apis: "live",
            apisFetchedAt: now,
            apisError: undefined,
          }));
        }
      } catch (e: any) {
        if (id === refreshId.current) {
          setApis(bundledApis);
          setStatus((s) => ({
            ...s,
            apis: "cached",
            apisError: e?.message ?? "Extraction failed",
          }));
        }
      }

      // Open-weight models — regex only from browser (HF is rate-limited)
      try {
        const raw = extractOpenWeightModels(payload);
        const resolved = raw
          .map((m) => resolveParamsB(m.name, m.model_id))
          .filter((r) => r.paramsB != null);

        const freshKnown: KnownModel[] = [];
        let i = 0;
        for (const r of resolved) {
          const src = raw[i];
          i++;
          freshKnown.push({
            params_b: r.paramsB!,
            active_b: r.activeB ?? null,
            name: src.name,
            arch: r.method === "regex_moe" ? "moe" : "dense",
            source: `models.dev:${r.method}`,
            last_seen: src.last_updated ?? "unknown",
          });
        }

        const mergedModels = mergeKnownModels(freshKnown, bundledKnownModels);
        if (id === refreshId.current) {
          setKnownModels(mergedModels);
          setStatus((s) => ({
            ...s,
            models: "live",
            modelsFetchedAt: now,
            modelsError: undefined,
          }));
        }
      } catch (e: any) {
        if (id === refreshId.current) {
          setKnownModels(bundledKnownModels);
          setStatus((s) => ({
            ...s,
            models: "cached",
            modelsError: e?.message ?? "Model extraction failed",
          }));
        }
      }
    } else {
      if (id === refreshId.current) {
        setApis(bundledApis);
        setKnownModels(bundledKnownModels);
        setStatus((s) => ({
          ...s,
          apis: "cached",
          apisError: "Connection to models.dev failed",
          models: "cached",
          modelsError: "Connection to models.dev failed",
        }));
      }
    }
  }, [bundledApis, bundledKnownModels]);

  // Auto-fetch on mount
  useEffect(() => {
    refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { apis, knownModels, status, refresh };
}

/** Merge bundled + fresh API lists, preferring fresh entries. */
function mergeApiLists(bundled: ClosedApi[], fresh: ClosedApi[]): ClosedApi[] {
  const map = new Map<string, ClosedApi>();
  for (const a of bundled) map.set(a.label, a);
  for (const a of fresh) map.set(a.label, a);
  return [...map.values()].sort((a, b) => a.label.localeCompare(b.label));
}