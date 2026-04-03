import { describe, expect, it } from "vitest";
import { isContextOverflowError, isLikelyContextOverflowError } from "./errors.js";

describe("isContextOverflowError", () => {
  it("recognises n_keep >= n_ctx local backend error", () => {
    const msg =
      "The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 22278 >= n_ctx: 4096)";
    expect(isContextOverflowError(msg)).toBe(true);
  });

  it("recognises lowercase n_keep/n_ctx variant", () => {
    expect(isContextOverflowError("error: n_keep exceeds n_ctx")).toBe(true);
  });

  it("does not false-positive on unrelated messages", () => {
    expect(isContextOverflowError("rate limit exceeded")).toBe(false);
    expect(isContextOverflowError("authentication failed")).toBe(false);
  });

  it("still matches existing overflow patterns", () => {
    expect(isContextOverflowError("context length exceeded")).toBe(true);
    expect(isContextOverflowError("request_too_large")).toBe(true);
    expect(isContextOverflowError("prompt is too long")).toBe(true);
  });
});

describe("isLikelyContextOverflowError", () => {
  it("recognises n_keep >= n_ctx via delegation to isContextOverflowError", () => {
    const msg =
      "The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 22278 >= n_ctx: 4096)";
    expect(isLikelyContextOverflowError(msg)).toBe(true);
  });

  it("does not match rate limit errors", () => {
    expect(isLikelyContextOverflowError("rate limit exceeded")).toBe(false);
  });
});
