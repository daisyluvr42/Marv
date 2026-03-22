import { describe, expect, it } from "vitest";
import { computeFusionSemanticMatch, computeWeightedScore } from "./salience-compute.js";

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
});
