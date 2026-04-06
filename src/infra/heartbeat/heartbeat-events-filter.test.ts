import { describe, expect, it } from "vitest";
import { buildCronEventPrompt } from "./heartbeat-events-filter.js";

describe("buildCronEventPrompt", () => {
  it("includes the event content", () => {
    const prompt = buildCronEventPrompt(["Morning report failed: timeout"]);
    expect(prompt).toContain("Morning report failed: timeout");
  });

  it("forbids hallucinated follow-up actions", () => {
    const prompt = buildCronEventPrompt(["Cron job error: provider unreachable"]);
    expect(prompt).toContain("Do NOT promise");
    expect(prompt).toContain("tool call in this same turn");
  });

  it("returns heartbeat ack for empty events", () => {
    const prompt = buildCronEventPrompt([]);
    expect(prompt).toContain("HEARTBEAT_OK");
    expect(prompt).not.toContain("Do NOT promise");
  });
});
