import { describe, expect, it } from "vitest";
import { extractGeminiResponse, extractLastJsonObject } from "./output-extract.js";

describe("media understanding output extraction", () => {
  it("parses the last top-level JSON object instead of nested objects", () => {
    const raw = '{"response":"ok","meta":{"tokens":1}}';
    expect(extractLastJsonObject(raw)).toEqual({
      response: "ok",
      meta: { tokens: 1 },
    });
  });

  it("extracts Gemini response from nested JSON payloads", () => {
    const raw = 'info\n{"response":"hello","meta":{"tokens":1,"finish_reason":"stop"}}';
    expect(extractGeminiResponse(raw)).toBe("hello");
  });
});
