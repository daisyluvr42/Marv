import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import type { WizardPrompter } from "../wizard/prompts.js";
import { applyDefaultModelChoice } from "./auth-choice.default-model.js";
import {
  applyGoogleGeminiModelDefault,
  GOOGLE_GEMINI_DEFAULT_MODEL,
} from "./google-gemini-model-default.js";
import {
  applyOpenAICodexModelDefault,
  OPENAI_CODEX_DEFAULT_MODEL,
} from "./openai-codex-model-default.js";
import {
  applyOpenAIConfig,
  applyOpenAIProviderConfig,
  OPENAI_DEFAULT_MODEL,
} from "./openai-model-default.js";
import {
  applyOpencodeZenModelDefault,
  OPENCODE_ZEN_DEFAULT_MODEL,
} from "./opencode-zen-model-default.js";

const loadModelCatalog = vi.hoisted(() => vi.fn(async () => []));
vi.mock("../agents/model/model-catalog.js", () => ({
  loadModelCatalog,
}));

function makePrompter(): WizardPrompter {
  return {
    intro: async () => {},
    outro: async () => {},
    note: async () => {},
    select: (async <T>() => "" as T) as WizardPrompter["select"],
    multiselect: (async <T>() => [] as T[]) as WizardPrompter["multiselect"],
    text: async () => "",
    confirm: async () => false,
    progress: () => ({ update: () => {}, stop: () => {} }),
  };
}

function expectPrimaryModelChanged(
  applied: { changed: boolean; next: MarvConfig },
  primary: string,
) {
  expect(applied.changed).toBe(true);
  expect(applied.next.agents?.defaults?.model).toEqual({ primary });
}

function expectConfigUnchanged(applied: { changed: boolean; next: MarvConfig }, cfg: MarvConfig) {
  expect(applied.changed).toBe(false);
  expect(applied.next).toEqual(cfg);
}

describe("applyDefaultModelChoice", () => {
  it("syncs provider selections when returning an agent override", async () => {
    const defaultModel = "vercel-ai-gateway/anthropic/claude-opus-4.6";
    loadModelCatalog.mockResolvedValueOnce([
      {
        provider: "vercel-ai-gateway",
        id: "anthropic/claude-opus-4.6",
        name: "Claude Opus 4.6",
      },
      {
        provider: "vercel-ai-gateway",
        id: "openai/gpt-5.4",
        name: "GPT-5.4",
      },
    ]);
    const noteAgentModel = vi.fn(async () => {});
    const applied = await applyDefaultModelChoice({
      config: {},
      setDefaultModel: false,
      defaultModel,
      // Simulate a provider function that does not explicitly add the entry.
      applyProviderConfig: (config: MarvConfig) => config,
      applyDefaultConfig: (config: MarvConfig) => config,
      noteAgentModel,
      prompter: makePrompter(),
    });

    expect(noteAgentModel).toHaveBeenCalledWith(defaultModel);
    expect(applied.agentModelOverride).toBe(defaultModel);
    expect(applied.config.models?.selections?.["vercel-ai-gateway"]).toContain(defaultModel);
  });

  it("adds canonical provider selections for anthropic aliases", async () => {
    const defaultModel = "anthropic/opus-4.6";
    loadModelCatalog.mockResolvedValueOnce([
      {
        provider: "anthropic",
        id: "claude-opus-4-6",
        name: "Claude Opus 4.6",
      },
    ]);
    const applied = await applyDefaultModelChoice({
      config: {},
      setDefaultModel: false,
      defaultModel,
      applyProviderConfig: (config: MarvConfig) => config,
      applyDefaultConfig: (config: MarvConfig) => config,
      noteAgentModel: async () => {},
      prompter: makePrompter(),
    });

    expect(applied.config.models?.selections?.anthropic).toContain("anthropic/claude-opus-4-6");
  });

  it("uses applyDefaultConfig path when setDefaultModel is true", async () => {
    const defaultModel = "openai/gpt-5.1-codex";
    const applied = await applyDefaultModelChoice({
      config: {},
      setDefaultModel: true,
      defaultModel,
      applyProviderConfig: (config: MarvConfig) => config,
      applyDefaultConfig: () => ({
        agents: {
          defaults: {
            model: { primary: defaultModel },
          },
        },
      }),
      noteDefault: defaultModel,
      noteAgentModel: async () => {},
      prompter: makePrompter(),
    });

    expect(applied.agentModelOverride).toBeUndefined();
    expect(applied.config.agents?.defaults?.model).toEqual({ primary: defaultModel });
  });
});

describe("applyGoogleGeminiModelDefault", () => {
  it("sets gemini default when model is unset", () => {
    const cfg: MarvConfig = { agents: { defaults: {} } };
    const applied = applyGoogleGeminiModelDefault(cfg);
    expectPrimaryModelChanged(applied, GOOGLE_GEMINI_DEFAULT_MODEL);
  });

  it("overrides existing model", () => {
    const cfg: MarvConfig = {
      agents: { defaults: { model: { primary: "anthropic/claude-opus-4-5" } } },
    };
    const applied = applyGoogleGeminiModelDefault(cfg);
    expectPrimaryModelChanged(applied, GOOGLE_GEMINI_DEFAULT_MODEL);
  });

  it("no-ops when already gemini default", () => {
    const cfg: MarvConfig = {
      agents: { defaults: { model: { primary: GOOGLE_GEMINI_DEFAULT_MODEL } } },
    };
    const applied = applyGoogleGeminiModelDefault(cfg);
    expectConfigUnchanged(applied, cfg);
  });
});

describe("applyOpenAIProviderConfig", () => {
  it("adds metadata entry for default model", () => {
    const next = applyOpenAIProviderConfig({});
    expect(Object.keys(next.models?.metadata ?? {})).toContain(OPENAI_DEFAULT_MODEL);
  });

  it("preserves existing alias for default model", () => {
    const next = applyOpenAIProviderConfig({
      models: {
        metadata: {
          [OPENAI_DEFAULT_MODEL]: { alias: "My GPT" },
        },
      },
    });
    expect(next.models?.metadata?.[OPENAI_DEFAULT_MODEL]?.alias).toBe("My GPT");
  });
});

describe("applyOpenAIConfig", () => {
  it("sets default when model is unset", () => {
    const next = applyOpenAIConfig({});
    expect(next.agents?.defaults?.model).toEqual({ primary: OPENAI_DEFAULT_MODEL });
  });

  it("overrides model.primary when model object already exists", () => {
    const next = applyOpenAIConfig({
      agents: { defaults: { model: { primary: "anthropic/claude-opus-4-6", fallbacks: [] } } },
    });
    expect(next.agents?.defaults?.model).toEqual({ primary: OPENAI_DEFAULT_MODEL, fallbacks: [] });
  });
});

describe("applyOpenAICodexModelDefault", () => {
  it("sets openai-codex default when model is unset", () => {
    const cfg: MarvConfig = { agents: { defaults: {} } };
    const applied = applyOpenAICodexModelDefault(cfg);
    expectPrimaryModelChanged(applied, OPENAI_CODEX_DEFAULT_MODEL);
  });

  it("sets openai-codex default when model is openai/*", () => {
    const cfg: MarvConfig = {
      agents: { defaults: { model: { primary: OPENAI_DEFAULT_MODEL } } },
    };
    const applied = applyOpenAICodexModelDefault(cfg);
    expectPrimaryModelChanged(applied, OPENAI_CODEX_DEFAULT_MODEL);
  });

  it("does not override openai-codex/*", () => {
    const cfg: MarvConfig = {
      agents: { defaults: { model: { primary: OPENAI_CODEX_DEFAULT_MODEL } } },
    };
    const applied = applyOpenAICodexModelDefault(cfg);
    expectConfigUnchanged(applied, cfg);
  });

  it("does not override non-openai models", () => {
    const cfg: MarvConfig = {
      agents: { defaults: { model: { primary: "anthropic/claude-opus-4-5" } } },
    };
    const applied = applyOpenAICodexModelDefault(cfg);
    expectConfigUnchanged(applied, cfg);
  });
});

describe("applyOpencodeZenModelDefault", () => {
  it("sets opencode default when model is unset", () => {
    const cfg: MarvConfig = { agents: { defaults: {} } };
    const applied = applyOpencodeZenModelDefault(cfg);
    expectPrimaryModelChanged(applied, OPENCODE_ZEN_DEFAULT_MODEL);
  });

  it("overrides existing model", () => {
    const cfg = {
      agents: { defaults: { model: "anthropic/claude-opus-4-5" } },
    } as MarvConfig;
    const applied = applyOpencodeZenModelDefault(cfg);
    expectPrimaryModelChanged(applied, OPENCODE_ZEN_DEFAULT_MODEL);
  });

  it("no-ops when already opencode-zen default", () => {
    const cfg = {
      agents: { defaults: { model: OPENCODE_ZEN_DEFAULT_MODEL } },
    } as MarvConfig;
    const applied = applyOpencodeZenModelDefault(cfg);
    expectConfigUnchanged(applied, cfg);
  });

  it("no-ops when already legacy opencode-zen default", () => {
    const cfg = {
      agents: { defaults: { model: "opencode-zen/claude-opus-4-5" } },
    } as MarvConfig;
    const applied = applyOpencodeZenModelDefault(cfg);
    expectConfigUnchanged(applied, cfg);
  });

  it("preserves fallbacks when setting primary", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          model: {
            primary: "anthropic/claude-opus-4-5",
            fallbacks: ["google/gemini-3-pro"],
          },
        },
      },
    };
    const applied = applyOpencodeZenModelDefault(cfg);
    expect(applied.changed).toBe(true);
    expect(applied.next.agents?.defaults?.model).toEqual({
      primary: OPENCODE_ZEN_DEFAULT_MODEL,
      fallbacks: ["google/gemini-3-pro"],
    });
  });
});
