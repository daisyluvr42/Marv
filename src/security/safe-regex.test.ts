import { describe, expect, it } from "vitest";
import { compileSafeRegex, hasNestedRepetition } from "./safe-regex.js";

describe("safe-regex", () => {
  it("detects nested repetition patterns", () => {
    expect(hasNestedRepetition("(a+)+$")).toBe(true);
    expect(hasNestedRepetition("(ab|cd)+$")).toBe(false);
  });

  it("rejects unsafe regexes", () => {
    expect(compileSafeRegex("(a+)+$", "g")).toBeNull();
  });

  it("compiles safe regexes", () => {
    const regex = compileSafeRegex("token=([A-Za-z0-9]+)", "gi");
    expect(regex).not.toBeNull();
    expect(regex?.test("TOKEN=abc123")).toBe(true);
  });
});
