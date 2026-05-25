import type { Pattern } from "./engine";

export interface PresetValues {
  queries_per_week: number;
  input_tokens: number;
  output_tokens: number;
  pattern: Pattern;
}

export interface Preset extends PresetValues {
  id: string;
  label: string;
  description: string;
}

export const PRESETS: Preset[] = [
  {
    id: "rag-chatbot",
    label: "RAG chatbot (small team)",
    description: "10k qpw · 20k in · 1k out · business hours",
    queries_per_week: 10_000,
    input_tokens: 20_000,
    output_tokens: 1_000,
    pattern: "business",
  },
  {
    id: "support-bot",
    label: "Customer support bot",
    description: "50k qpw · 800 in · 300 out · business hours",
    queries_per_week: 50_000,
    input_tokens: 800,
    output_tokens: 300,
    pattern: "business",
  },
  {
    id: "code-assistant",
    label: "Code assistant (internal)",
    description: "100k qpw · 4k in · 800 out · business hours",
    queries_per_week: 100_000,
    input_tokens: 4_000,
    output_tokens: 800,
    pattern: "business",
  },
  {
    id: "batch-docs",
    label: "Batch document processing",
    description: "200k qpw · 5k in · 1k out · uniform",
    queries_per_week: 200_000,
    input_tokens: 5_000,
    output_tokens: 1_000,
    pattern: "uniform",
  },
  {
    id: "high-volume-api",
    label: "High-volume API replacement",
    description: "1M qpw · 1.5k in · 400 out · uniform",
    queries_per_week: 1_000_000,
    input_tokens: 1_500,
    output_tokens: 400,
    pattern: "uniform",
  },
  {
    id: "always-on-agent",
    label: "Always-on agent",
    description: "25k qpw · 8k in · 1.5k out · always warm",
    queries_per_week: 25_000,
    input_tokens: 8_000,
    output_tokens: 1_500,
    pattern: "always_warm",
  },
  {
    id: "bursty-consumer",
    label: "Bursty consumer app",
    description: "500k qpw · 1k in · 400 out · bursty",
    queries_per_week: 500_000,
    input_tokens: 1_000,
    output_tokens: 400,
    pattern: "bursty",
  },
];

interface PresetsProps {
  current: PresetValues;
  onApply: (p: PresetValues) => void;
}

export function Presets({ current, onApply }: PresetsProps) {
  const activeId = PRESETS.find(
    (p) =>
      p.queries_per_week === current.queries_per_week &&
      p.input_tokens === current.input_tokens &&
      p.output_tokens === current.output_tokens &&
      p.pattern === current.pattern
  )?.id;

  return (
    <div className="mb-5">
      <div className="text-sm font-semibold uppercase tracking-wide text-slate-500 mb-2">
        Common scenarios
      </div>
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => {
          const active = p.id === activeId;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() =>
                onApply({
                  queries_per_week: p.queries_per_week,
                  input_tokens: p.input_tokens,
                  output_tokens: p.output_tokens,
                  pattern: p.pattern,
                })
              }
              title={p.description}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                active
                  ? "bg-indigo-600 border-indigo-600 text-white shadow-sm"
                  : "bg-white border-slate-300 text-slate-700 hover:bg-indigo-50 hover:border-indigo-300"
              }`}
            >
              {p.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
