import { describe, it, expect } from "vitest";
import { matchElo, attachElo, normaliseName, type EloEntry, type EloSnapshot } from "../src/eloMatch";

const entries: EloEntry[] = [
  { rank: 1, model: "claude-opus-4-6-thinking", vendor: "Anthropic", license: "proprietary", score: 1502 },
  { rank: 2, model: "claude-opus-4-6", vendor: "Anthropic", license: "proprietary", score: 1498 },
  { rank: 3, model: "claude-opus-4-7", vendor: "Anthropic", license: "proprietary", score: 1492 },
  { rank: 4, model: "claude-sonnet-4-6", vendor: "Anthropic", license: "proprietary", score: 1468 },
  { rank: 5, model: "gpt-5.5", vendor: "OpenAI", license: "proprietary", score: 1478 },
  { rank: 6, model: "gpt-5.4", vendor: "OpenAI", license: "proprietary", score: 1467 },
  { rank: 7, model: "gpt-5.2-chat-latest-20260210", vendor: "OpenAI", license: "proprietary", score: 1477 },
  { rank: 8, model: "gemini-3.5-flash", vendor: "Google", license: "proprietary", score: 1480 },
  { rank: 9, model: "gemini-3-pro", vendor: "Google", license: "proprietary", score: 1486 },
  { rank: 10, model: "glm-5.1", vendor: "Z.ai", license: "open", score: 1472 },
  { rank: 11, model: "kimi-k2.6", vendor: "Moonshot", license: "open", score: 1462 },
  { rank: 12, model: "deepseek-v4-pro-thinking", vendor: "DeepSeek", license: "open", score: 1461 },
];

describe("normaliseName", () => {
  it("strips non-alnum and lowercases", () => {
    expect(normaliseName("Claude-Opus 4.6")).toBe("claudeopus46");
    expect(normaliseName("GPT-5.5 Pro")).toBe("gpt55pro");
  });
});

describe("matchElo", () => {
  it("matches exact normalised names across separator differences", () => {
    expect(matchElo("Claude Opus 4.6", entries)?.model).toBe("claude-opus-4-6");
    expect(matchElo("GPT-5.5", entries)?.model).toBe("gpt-5.5");
    expect(matchElo("Gemini 3.5 Flash", entries)?.model).toBe("gemini-3.5-flash");
    expect(matchElo("GLM-5.1", entries)?.model).toBe("glm-5.1");
  });

  it("rejects mismatched versions even when family overlaps", () => {
    // GPT-5.2 must not silently match gpt-5.5 / gpt-5.4
    expect(matchElo("OpenAI GPT-5.2", entries)?.model).toBeUndefined();
    // Claude Opus 4.1 has no 4.1 entry; 4.6 and 4.7 are versions away
    expect(matchElo("Anthropic Claude Opus 4.1", entries)).toBeNull();
    // GPT-4 vs gpt-5.x
    expect(matchElo("OpenAI GPT-4", entries)).toBeNull();
  });

  it("matches dated/build-suffixed arena entries", () => {
    // arena ID has trailing date — should still match the bare label
    expect(matchElo("OpenAI GPT-5.2 Chat", entries)?.model).toBe(
      "gpt-5.2-chat-latest-20260210"
    );
  });

  it("matches with provider-prefixed labels", () => {
    expect(matchElo("Anthropic Claude Sonnet 4.6", entries)?.model).toBe("claude-sonnet-4-6");
    expect(matchElo("Zhipu AI GLM-5.1", entries)?.model).toBe("glm-5.1");
  });

  it("returns null for clearly unrelated names", () => {
    expect(matchElo("Llama 3.1 8B", entries)).toBeNull();
    expect(matchElo("text-embedding-3-large", entries)).toBeNull();
    expect(matchElo("", entries)).toBeNull();
  });

  it("returns null on an empty snapshot", () => {
    expect(matchElo("GPT-5.5", [])).toBeNull();
  });
});

describe("attachElo", () => {
  const snapshot: EloSnapshot = {
    last_updated: "test",
    source: "test",
    entries,
  };

  it("attaches elo + eloRank to matched items and lists unmatched", () => {
    const items = [
      { name: "Claude Opus 4.6" },
      { name: "Claude Sonnet 4.6" },
      { name: "Llama 3.1 8B" },
      { label: "GLM-5.1" },
    ] as { name?: string; label?: string; elo?: number; eloRank?: number }[];

    const r = attachElo(items, snapshot);

    expect(items[0].elo).toBe(1498);
    expect(items[0].eloRank).toBe(2);
    expect(items[1].elo).toBe(1468);
    expect(items[2].elo).toBeUndefined();
    expect(items[3].elo).toBe(1472);
    expect(r.matched).toBe(3);
    expect(r.unmatched).toEqual(["Llama 3.1 8B"]);
  });

  it("doesn't crash on items with neither name nor label", () => {
    const items: any[] = [{}];
    const r = attachElo(items, snapshot);
    expect(r.matched).toBe(0);
    expect(items[0].elo).toBeUndefined();
  });
});
