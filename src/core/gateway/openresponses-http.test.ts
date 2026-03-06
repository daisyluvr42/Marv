import { describe, expect, it } from "vitest";
import { buildAgentPrompt } from "./openresponses-http.js";

describe("buildAgentPrompt", () => {
  it("includes function_call items in the reconstructed conversation", () => {
    const prompt = buildAgentPrompt([
      {
        type: "message",
        role: "user",
        content: "What's the weather in SF?",
      },
      {
        type: "function_call",
        call_id: "call_weather",
        name: "get_weather",
        arguments: '{"location":"SF"}',
      },
      {
        type: "function_call_output",
        call_id: "call_weather",
        output: '{"temp":"70F"}',
      },
      {
        type: "message",
        role: "user",
        content: "Summarize that result.",
      },
    ]);

    expect(prompt.message).toContain('[function_call:call_weather] get_weather({"location":"SF"})');
    expect(prompt.message).toContain('Tool:call_weather: {"temp":"70F"}');
  });

  it("preserves reasoning summaries in extra system prompt", () => {
    const prompt = buildAgentPrompt([
      {
        type: "reasoning",
        summary: "The user wants the shortest safe answer.",
      },
      {
        type: "message",
        role: "user",
        content: "Reply briefly.",
      },
    ]);

    expect(prompt.extraSystemPrompt).toContain("Previous reasoning summary");
    expect(prompt.extraSystemPrompt).toContain("The user wants the shortest safe answer.");
  });
});
