import { describe, expect, it } from "vitest";
import {
  buildAuthChoiceGroups,
  buildAuthChoiceOptions,
  formatAuthChoiceChoicesForCli,
} from "./auth-choice-options.js";

function getOptions(includeSkip = false) {
  return buildAuthChoiceOptions({
    includeSkip,
  });
}

describe("buildAuthChoiceOptions", () => {
  it("includes GitHub Copilot", () => {
    const options = getOptions();

    expect(options.find((opt) => opt.value === "github-copilot")).toBeDefined();
  });

  it("includes Anthropic token option", () => {
    const options = getOptions();

    expect(options.some((opt) => opt.value === "token")).toBe(true);
  });

  it.each([
    ["Z.AI (GLM) auth choice", ["zai-api-key"]],
    ["Xiaomi auth choice", ["xiaomi-api-key"]],
    ["MiniMax auth choice", ["minimax-api", "minimax-api-key-cn", "minimax-api-lightning"]],
    [
      "Moonshot auth choice",
      ["moonshot-api-key", "moonshot-api-key-cn", "kimi-code-api-key", "together-api-key"],
    ],
    ["Vercel AI Gateway auth choice", ["ai-gateway-api-key"]],
    ["Cloudflare AI Gateway auth choice", ["cloudflare-ai-gateway-api-key"]],
    ["Together AI auth choice", ["together-api-key"]],
    ["Synthetic auth choice", ["synthetic-api-key"]],
    ["Chutes OAuth auth choice", ["chutes"]],
    ["Qwen auth choice", ["qwen-portal"]],
    ["xAI auth choice", ["xai-api-key"]],
    ["vLLM auth choice", ["vllm"]],
  ])("includes %s", (_label, expectedValues) => {
    const options = getOptions();

    for (const value of expectedValues) {
      expect(options.some((opt) => opt.value === value)).toBe(true);
    }
  });

  it("builds cli help choices from the same catalog", () => {
    const options = getOptions(true);
    const cliChoices = formatAuthChoiceChoicesForCli({
      includeSkip: true,
    }).split("|");

    for (const option of options) {
      expect(cliChoices).toContain(option.value);
    }
  });

  it("omits removed legacy aliases from cli help choices", () => {
    const cliChoices = formatAuthChoiceChoicesForCli({
      includeSkip: true,
    }).split("|");

    expect(cliChoices).not.toContain("setup-token");
    expect(cliChoices).not.toContain("oauth");
    expect(cliChoices).not.toContain("claude-cli");
    expect(cliChoices).not.toContain("codex-cli");
  });

  it("shows Chutes in grouped provider selection", () => {
    const { groups } = buildAuthChoiceGroups({
      includeSkip: false,
    });
    const chutesGroup = groups.find((group) => group.value === "chutes");

    expect(chutesGroup).toBeDefined();
    expect(chutesGroup?.options.some((opt) => opt.value === "chutes")).toBe(true);
  });
});
