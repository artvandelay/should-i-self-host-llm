import { describe, it, expect } from "vitest";

import {
  breakEvenWeeks,
  costForView,
  costViewLabel,
  costViewSuffix,
  projectCost,
  WEEKS_PER_MONTH,
  WEEKS_PER_YEAR,
} from "../src/engine";

describe("projectCost", () => {
  it("derives monthly and annual from weekly", () => {
    const p = projectCost(100);
    expect(p.weekly).toBe(100);
    expect(p.monthly).toBeCloseTo(100 * WEEKS_PER_MONTH, 6);
    expect(p.annual).toBe(100 * WEEKS_PER_YEAR);
  });

  it("monthly * 12 should be approximately annual", () => {
    const p = projectCost(123.45);
    expect(p.monthly * 12).toBeCloseTo(p.annual, 6);
  });

  it("handles zero", () => {
    const p = projectCost(0);
    expect(p.weekly).toBe(0);
    expect(p.monthly).toBe(0);
    expect(p.annual).toBe(0);
  });
});

describe("costForView", () => {
  it("returns weekly unchanged", () => {
    expect(costForView(50, "weekly")).toBe(50);
  });
  it("scales monthly by 52/12", () => {
    expect(costForView(50, "monthly")).toBeCloseTo(50 * (52 / 12), 6);
  });
  it("scales annual by 52", () => {
    expect(costForView(50, "annual")).toBe(50 * 52);
  });
});

describe("costView labels and suffixes", () => {
  it("formats suffixes", () => {
    expect(costViewSuffix("weekly")).toBe("/wk");
    expect(costViewSuffix("monthly")).toBe("/mo");
    expect(costViewSuffix("annual")).toBe("/yr");
  });
  it("formats axis labels", () => {
    expect(costViewLabel("weekly")).toMatch(/[Ww]eekly/);
    expect(costViewLabel("monthly")).toMatch(/[Mm]onthly/);
    expect(costViewLabel("annual")).toMatch(/[Aa]nnual/);
  });
});

describe("breakEvenWeeks", () => {
  it("computes break-even with positive weekly savings", () => {
    // $5000 setup, save $100/wk -> 50 weeks
    expect(breakEvenWeeks(200, 100, 5000)).toBe(50);
  });

  it("returns 0 when setup cost is zero", () => {
    expect(breakEvenWeeks(200, 100, 0)).toBe(0);
  });

  it("rounds up to the next whole week", () => {
    // 5000 / 99 = 50.50... -> 51
    expect(breakEvenWeeks(199, 100, 5000)).toBe(51);
  });

  it("returns null when self-host is not cheaper", () => {
    expect(breakEvenWeeks(100, 100, 5000)).toBeNull();
    expect(breakEvenWeeks(100, 150, 5000)).toBeNull();
  });

  it("returns null when break-even exceeds cap (10 years default)", () => {
    // save $1/wk, $1,000,000 setup -> 1,000,000 weeks, far above 520 cap
    expect(breakEvenWeeks(101, 100, 1_000_000)).toBeNull();
  });

  it("respects custom cap", () => {
    // 5000 / 100 = 50 weeks; cap at 10 -> null; cap at 100 -> 50
    expect(breakEvenWeeks(200, 100, 5000, 10)).toBeNull();
    expect(breakEvenWeeks(200, 100, 5000, 100)).toBe(50);
  });

  it("rejects NaN / negative setup", () => {
    expect(breakEvenWeeks(NaN, 100, 5000)).toBeNull();
    expect(breakEvenWeeks(200, NaN, 5000)).toBeNull();
    expect(breakEvenWeeks(200, 100, -100)).toBeNull();
  });
});
