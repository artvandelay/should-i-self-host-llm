import { describe, it, expect } from "vitest";
import {
  extractClosedApis,
  resolveParamsB,
  mergeKnownModels,
  type ModelsDevPayload,
  type KnownModel,
} from "../src/modelsDev";

describe("extractClosedApis", () => {
  it("extracts closed models from first-party providers", () => {
    const payload: ModelsDevPayload = {
      openai: {
        name: "OpenAI",
        env: [],
        npm: "",
        doc: "",
        models: {
          "gpt-4o-mini": {
            id: "gpt-4o-mini",
            name: "GPT-4o Mini",
            family: "",
            open_weights: false,
            cost: { input: 0.15, output: 0.60 },
            last_updated: "2026-05-20",
          },
        },
      },
      helicone: {
        name: "Helicone",
        env: [],
        npm: "",
        doc: "",
        models: {
          "gpt-5": {
            id: "gpt-5",
            name: "GPT-5 via Helicone",
            family: "",
            open_weights: false,
            cost: { input: 2.50, output: 10.0 },
          },
        },
      },
    };
    const extracted = extractClosedApis(payload);
    expect(extracted).toHaveLength(1);
    expect(extracted[0].label).toBe("OpenAI GPT-4o Mini");
    expect(extracted[0].input_per_1m).toBe(0.15);
    expect(extracted[0].output_per_1m).toBe(0.60);
  });

  it("skips models missing cost fields", () => {
    const payload: ModelsDevPayload = {
      openai: {
        name: "OpenAI",
        env: [],
        npm: "",
        doc: "",
        models: {
          "no-cost": {
            id: "no-cost",
            name: "No Cost Model",
            family: "",
            open_weights: false,
            last_updated: "2026-05-20",
          },
        },
      },
    };
    expect(extractClosedApis(payload)).toHaveLength(0);
  });

  it("skips open-weight models", () => {
    const payload: ModelsDevPayload = {
      openai: {
        name: "OpenAI",
        env: [],
        npm: "",
        doc: "",
        models: {
          "llama-3.1-8b": {
            id: "llama-3.1-8b",
            name: "Llama 3.1 8B",
            family: "",
            open_weights: true,
            cost: { input: 0, output: 0 },
          },
        },
      },
    };
    expect(extractClosedApis(payload)).toHaveLength(0);
  });
});

describe("resolveParamsB", () => {
  it("resolves millions", () => {
    const r = resolveParamsB("SmolLM2 135M", "smollm2-135m");
    expect(r.paramsB).toBe(0.135);
    expect(r.method).toBe("regex");
  });

  it("resolves billions", () => {
    const r = resolveParamsB("Llama 3.3 70B", "llama-3.3-70b");
    expect(r.paramsB).toBe(70);
    expect(r.method).toBe("regex");
  });

  it("detects MoE with active params", () => {
    const r = resolveParamsB("Qwen3-235B-A22B (MoE)", "qwen3-235b-a22b");
    expect(r.paramsB).toBe(235);
    expect(r.activeB).toBe(22);
    expect(r.method).toBe("regex_moe");
  });

  it("returns unknown for names without param counts", () => {
    const r = resolveParamsB("GPT-4o", "gpt-4o");
    expect(r.paramsB).toBeNull();
    expect(r.method).toBe("unknown");
  });

  it("returns unknown for model IDs with colons", () => {
    const r = resolveParamsB("Some Model:v2", "some-model:v2");
    expect(r.paramsB).toBeNull();
    expect(r.method).toBe("unknown");
  });

  it("returns unknown for model IDs without slashes", () => {
    const r = resolveParamsB("Kimi K2.5", "kimi-k2.5");
    expect(r.paramsB).toBeNull();
    expect(r.method).toBe("unknown");
  });
});

describe("mergeKnownModels", () => {
  it("merges fresh over cached, preserving manual active_b", () => {
    const cached: KnownModel[] = [
      { params_b: 70, active_b: null, name: "Llama 3.3 70B", arch: "dense", source: "manual", last_seen: "2026-05-20" },
      { params_b: 671, active_b: 37, name: "DeepSeek V3 671B-A37B", arch: "moe", source: "manual", last_seen: "2026-05-20" },
    ];
    const fresh: KnownModel[] = [
      { params_b: 70, active_b: null, name: "Llama 3.3 70B", arch: "dense", source: "models.dev:regex", last_seen: "2026-05-23" },
      { params_b: 8, active_b: null, name: "Llama 3.1 8B", arch: "dense", source: "models.dev:regex", last_seen: "2026-05-23" },
    ];
    const merged = mergeKnownModels(fresh, cached);
    expect(merged).toHaveLength(3);

    const llama70 = merged.find((m) => m.name === "Llama 3.3 70B");
    expect(llama70?.source).toBe("models.dev:regex");
    expect(llama70?.last_seen).toBe("2026-05-23");

    const deepseek = merged.find((m) => m.name === "DeepSeek V3 671B-A37B");
    expect(deepseek?.active_b).toBe(37);
    expect(deepseek?.source).toBe("manual");

    const llama8 = merged.find((m) => m.name === "Llama 3.1 8B");
    expect(llama8?.params_b).toBe(8);
  });

  it("preserves cached entries with no fresh counterpart", () => {
    const cached: KnownModel[] = [
      { params_b: 5, active_b: null, name: "Some Legacy Model", arch: "dense", source: "manual", last_seen: "2025-01-01" },
    ];
    expect(mergeKnownModels([], cached)).toHaveLength(1);
    expect(mergeKnownModels([], cached)[0].name).toBe("Some Legacy Model");
  });
});