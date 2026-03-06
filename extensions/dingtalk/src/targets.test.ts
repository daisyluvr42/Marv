import { describe, expect, it } from "vitest";
import { looksLikeDingTalkId, normalizeDingTalkTarget, parseDingTalkTarget } from "./targets.js";

describe("dingtalk targets", () => {
  it("normalizes prefixed user and group targets", () => {
    expect(normalizeDingTalkTarget("dingtalk:user:alice")).toBe("user:alice");
    expect(normalizeDingTalkTarget("group:cid123")).toBe("group:cid123");
  });

  it("parses user and group targets", () => {
    expect(parseDingTalkTarget("user:alice")).toEqual({ kind: "user", value: "alice" });
    expect(parseDingTalkTarget("conversation:cid123")).toEqual({
      kind: "group",
      value: "cid123",
    });
  });

  it("detects id-like targets", () => {
    expect(looksLikeDingTalkId("user:alice")).toBe(true);
    expect(looksLikeDingTalkId("alice")).toBe(false);
  });
});
