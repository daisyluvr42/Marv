import { describe, expect, it } from "vitest";
import {
  FORGET_CONFIDENCE_THRESHOLD,
  FORGET_STREAK_HALF_LIVES,
  P0_CLARITY_HALF_LIFE_DAYS,
  P1_CLARITY_HALF_LIFE_DAYS,
  P2_CLARITY_HALF_LIFE_DAYS,
  P3_CLARITY_HALF_LIFE_DAYS,
  clarityDecayFactor,
  computeCurrentClarity,
  computeFusionSemanticMatch,
  computeWeightedScore,
  shouldPruneMemoryItem,
} from "./salience-compute.js";

describe("salience-compute", () => {
  it("computes five-signal fusion and normalizes by total weight", () => {
    const score = computeFusionSemanticMatch({
      vectorScore: 0.8,
      lexicalScore: 0.6,
      bm25Score: 0.4,
      graphScore: 0.5,
      clusterScore: 0.2,
    });
    expect(score).toBeCloseTo(0.5928571, 5);
  });

  it("returns zero when all fusion weights are zero", () => {
    const score = computeFusionSemanticMatch(
      {
        vectorScore: 1,
        lexicalScore: 1,
        bm25Score: 1,
        graphScore: 1,
        clusterScore: 1,
      },
      {
        vector: 0,
        lexical: 0,
        bm25: 0,
        graph: 0,
        cluster: 0,
      },
    );
    expect(score).toBe(0);
  });

  it("handles weighted score extremes", () => {
    expect(computeWeightedScore(0.5, 0.5)).toBeGreaterThan(0.5);
    expect(computeWeightedScore(0.5, 2)).toBeLessThan(0.5);
    expect(computeWeightedScore(0.5, Number.NaN)).toBeCloseTo(0.5);
  });

  it("decay and pruning are disabled (all items are P3, no decay)", () => {
    const config = {
      p0ClarityHalfLifeDays: P0_CLARITY_HALF_LIFE_DAYS,
      p1ClarityHalfLifeDays: P1_CLARITY_HALF_LIFE_DAYS,
      p2ClarityHalfLifeDays: P2_CLARITY_HALF_LIFE_DAYS,
      p3ClarityHalfLifeDays: P3_CLARITY_HALF_LIFE_DAYS,
      forgetConfidenceThreshold: FORGET_CONFIDENCE_THRESHOLD,
      forgetStreakHalfLives: FORGET_STREAK_HALF_LIVES,
    };

    // All tiers return 1 (no decay)
    expect(clarityDecayFactor("P0", 30, config)).toBe(1);
    expect(clarityDecayFactor("P2", 30, config)).toBe(1);
    expect(clarityDecayFactor("P3", 365, config)).toBe(1);

    // Clarity = confidence (no decay applied)
    const item = { confidence: 0.2, tier: "P2" as const };
    expect(computeCurrentClarity(item, 120, config)).toBe(0.2);

    // Pruning always returns false
    expect(shouldPruneMemoryItem(item, 120, config)).toBe(false);
    expect(shouldPruneMemoryItem({ confidence: 0.01, tier: "P3" as const }, 999, config)).toBe(
      false,
    );
  });
});
