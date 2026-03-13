import { afterEach, describe, expect, it } from "vitest";
import { assertMediaRoutingShadowClean, detectMediaRoutingShadowMismatches } from "./apply.js";

const previousAssert = process.env.MARV_MEDIA_ROUTING_SHADOW_ASSERT;

afterEach(() => {
  if (previousAssert === undefined) {
    delete process.env.MARV_MEDIA_ROUTING_SHADOW_ASSERT;
    return;
  }
  process.env.MARV_MEDIA_ROUTING_SHADOW_ASSERT = previousAssert;
});

describe("detectMediaRoutingShadowMismatches", () => {
  it("returns no mismatches when routing snapshots agree", () => {
    expect(
      detectMediaRoutingShadowMismatches({
        expected: {
          promptMedia: [{ kind: "image", source: "native", path: "/tmp/a.png" }],
          derivedText: { transcript: "hello" },
          decisions: [],
          settled: true,
        },
        actual: {
          promptMedia: [{ kind: "image", source: "native", path: "/tmp/a.png" }],
          derivedText: { transcript: "hello" },
          decisions: [],
          settled: true,
        },
      }),
    ).toEqual([]);
  });

  it("reports the fields that diverge", () => {
    expect(
      detectMediaRoutingShadowMismatches({
        expected: {
          promptMedia: [],
          derivedText: { transcript: "hello" },
          decisions: [],
          settled: true,
        },
        actual: {
          promptMedia: [{ kind: "audio", source: "derived", path: "/tmp/a.ogg" }],
          derivedText: { transcript: "different" },
          decisions: [{ capability: "audio", outcome: "success", attachments: [] }],
          settled: false,
        },
      }),
    ).toEqual(["promptMedia", "derivedText", "decisions", "settled"]);
  });

  it("can enforce a hard failure when shadow assert mode is enabled", () => {
    process.env.MARV_MEDIA_ROUTING_SHADOW_ASSERT = "1";
    expect(() => assertMediaRoutingShadowClean(["promptMedia", "derivedText"])).toThrowError(
      /routing shadow mismatch fields=promptMedia,derivedText/,
    );
  });

  it("does not throw when shadow assert mode is disabled", () => {
    delete process.env.MARV_MEDIA_ROUTING_SHADOW_ASSERT;
    expect(() => assertMediaRoutingShadowClean(["promptMedia"])).not.toThrow();
  });
});
