import { describe, expect, it } from "vitest";
import {
  FORGET_CONFIDENCE_THRESHOLD,
  FORGET_STREAK_HALF_LIVES,
  P0_CLARITY_HALF_LIFE_DAYS,
  P1_CLARITY_HALF_LIFE_DAYS,
  P2_CLARITY_HALF_LIFE_DAYS,
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

  it("applies clarity decay by tier and prunes stale low-confidence memory", () => {
    const config = {
      p0ClarityHalfLifeDays: P0_CLARITY_HALF_LIFE_DAYS,
      p1ClarityHalfLifeDays: P1_CLARITY_HALF_LIFE_DAYS,
      p2ClarityHalfLifeDays: P2_CLARITY_HALF_LIFE_DAYS,
      forgetConfidenceThreshold: FORGET_CONFIDENCE_THRESHOLD,
      forgetStreakHalfLives: FORGET_STREAK_HALF_LIVES,
    };

    const p0Decay = clarityDecayFactor("P0", 30, config);
    const p2Decay = clarityDecayFactor("P2", 30, config);
    expect(p0Decay).toBeGreaterThan(p2Decay);

    const p2Item = { confidence: 0.2, tier: "P2" as const };
    const p0Item = { confidence: 0.95, tier: "P0" as const };
    expect(computeCurrentClarity(p2Item, 120, config)).toBeLessThan(
      config.forgetConfidenceThreshold,
    );
    expect(shouldPruneMemoryItem(p2Item, 120, config)).toBe(true);
    expect(shouldPruneMemoryItem(p0Item, 3650, config)).toBe(false);
  });
});
