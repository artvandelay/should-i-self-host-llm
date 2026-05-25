import { describe, it, expect } from "vitest";

import { cumulativeProjection } from "../src/engine";
import {
  buildPaybackSentence,
  queriesToAmortize,
} from "../src/ftPayback";

describe("buildPaybackSentence", () => {
  it("branch 3 good payback — cumulativeProjection(2000, 200, 5000, 24)", () => {
    const proj = cumulativeProjection(2000, 200, 5000, 24);
    const sentence = buildPaybackSentence(5000, 2000, 200, proj);
    expect(sentence.tone).toBe("good");
    expect(proj.crossover_month).not.toBeNull();
    expect(sentence.headline).toMatch(/pays for itself in \d+ month/);
    expect(sentence.supporting).toMatch(/12 months/);
    expect(sentence.supporting).toMatch(/24 months/);
  });

  it("branch 1 bad — selfhost weekly > api weekly", () => {
    const proj = cumulativeProjection(100, 200, 5000, 24);
    const sentence = buildPaybackSentence(5000, 100, 200, proj);
    expect(sentence.tone).toBe("bad");
    expect(sentence.headline).toMatch(/won't pay back/i);
  });

  it("branch 1 bad — equal weekly costs", () => {
    const proj = cumulativeProjection(100, 100, 5000, 24);
    const sentence = buildPaybackSentence(5000, 100, 100, proj);
    expect(sentence.tone).toBe("bad");
    expect(sentence.headline).toMatch(/won't pay back/i);
  });

  it("branch 2 neutral — huge capex, weekly savings but no crossover", () => {
    const proj = cumulativeProjection(200, 100, 10_000_000, 24);
    expect(proj.crossover_month).toBeNull();
    const sentence = buildPaybackSentence(10_000_000, 200, 100, proj);
    expect(sentence.tone).toBe("neutral");
    expect(sentence.headline).toMatch(/won't pay back within 24 months/i);
  });

  it('uses singular "month" when crossover_month === 1', () => {
    const proj = cumulativeProjection(1000, 100, 3500, 24);
    expect(proj.crossover_month).toBe(1);
    const sentence = buildPaybackSentence(3500, 1000, 100, proj);
    expect(sentence.tone).toBe("good");
    expect(sentence.headline).toBe("Fine-tuning pays for itself in 1 month");
    expect(sentence.headline).not.toMatch(/months/);
  });
});

describe("queriesToAmortize", () => {
  it("returns positive queries when savings exist", () => {
    const result = queriesToAmortize(5000, 2000, 200, 500_000);
    expect(result).not.toBeNull();
    expect(result!.queries).toBeGreaterThan(0);
    expect(result!.savings_per_query).toBeCloseTo((2000 - 200) / 500_000, 10);
  });

  it("capex 0 returns queries 0", () => {
    const result = queriesToAmortize(0, 2000, 200, 500_000);
    expect(result).toEqual({
      queries: 0,
      savings_per_query: (2000 - 200) / 500_000,
    });
  });

  it("returns null when api weekly <= selfhost weekly", () => {
    expect(queriesToAmortize(5000, 100, 200, 500_000)).toBeNull();
    expect(queriesToAmortize(5000, 100, 100, 500_000)).toBeNull();
  });

  it("returns null when queries_per_week is 0", () => {
    expect(queriesToAmortize(5000, 2000, 200, 0)).toBeNull();
  });

  it("returns null for NaN / Infinity inputs", () => {
    expect(queriesToAmortize(NaN, 2000, 200, 500_000)).toBeNull();
    expect(queriesToAmortize(5000, Infinity, 200, 500_000)).toBeNull();
    expect(queriesToAmortize(5000, 2000, 200, Infinity)).toBeNull();
  });
});
