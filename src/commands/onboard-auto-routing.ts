import type { OpenClawConfig } from "../config/config.js";
import type { AutoRoutingConfig, AutoRoutingRule } from "../config/types.agent-defaults.js";
import type { WizardPrompter } from "../wizard/prompts.js";

type ModelOption = { value: string; label: string };

const WELL_KNOWN_FAST_MODELS: ModelOption[] = [
  { value: "anthropic/claude-haiku-4-5", label: "Claude Haiku 4.5 (Anthropic)" },
  { value: "openai/gpt-4o-mini", label: "GPT-4o Mini (OpenAI)" },
  { value: "google/gemini-2.0-flash", label: "Gemini 2.0 Flash (Google)" },
];

const WELL_KNOWN_POWERFUL_MODELS: ModelOption[] = [
  { value: "anthropic/claude-opus-4-6", label: "Claude Opus 4.6 (Anthropic)" },
  { value: "anthropic/claude-sonnet-4-6", label: "Claude Sonnet 4.6 (Anthropic)" },
  { value: "openai/gpt-5.2", label: "GPT-5.2 (OpenAI)" },
  { value: "google/gemini-3-pro-preview", label: "Gemini 3 Pro (Google)" },
];

function buildRules(fastModel: string, powerfulModel: string): AutoRoutingRule[] {
  return [
    { complexity: "simple", model: fastModel, thinking: "off" },
    { complexity: "moderate", model: fastModel, thinking: "off" },
    { complexity: "complex", model: powerfulModel, thinking: "low" },
    { complexity: "expert", model: powerfulModel, thinking: "medium" },
  ];
}

function applyAutoRouting(cfg: OpenClawConfig, routing: AutoRoutingConfig): OpenClawConfig {
  return {
    ...cfg,
    agents: {
      ...cfg.agents,
      defaults: {
        ...cfg.agents?.defaults,
        autoRouting: routing,
      },
    },
  };
}

/**
 * Onboarding step: prompt user to configure auto model routing.
 * Returns the updated config (unchanged if user declines).
 */
export async function promptAutoRouting(params: {
  config: OpenClawConfig;
  prompter: WizardPrompter;
}): Promise<OpenClawConfig> {
  const { prompter } = params;
  let config = params.config;

  const enable = await prompter.confirm({
    message: "Enable auto model routing? (routes simpler messages to faster/cheaper models)",
    initialValue: false,
  });

  if (!enable) {
    return config;
  }

  const fastModel = await prompter.select<string>({
    message: "Pick a fast model for simple/moderate messages:",
    options: [
      ...WELL_KNOWN_FAST_MODELS.map((m) => ({ value: m.value, label: m.label })),
      { value: "_custom", label: "Custom (enter provider/model)" },
    ],
  });

  let resolvedFast = fastModel;
  if (fastModel === "_custom") {
    resolvedFast = await prompter.text({
      message: "Enter fast model (provider/model):",
      placeholder: "anthropic/claude-haiku-4-5",
    });
  }

  const powerfulModel = await prompter.select<string>({
    message: "Pick a powerful model for complex/expert tasks:",
    options: [
      ...WELL_KNOWN_POWERFUL_MODELS.map((m) => ({ value: m.value, label: m.label })),
      { value: "_custom", label: "Custom (enter provider/model)" },
    ],
  });

  let resolvedPowerful = powerfulModel;
  if (powerfulModel === "_custom") {
    resolvedPowerful = await prompter.text({
      message: "Enter powerful model (provider/model):",
      placeholder: "anthropic/claude-opus-4-6",
    });
  }

  const rules = buildRules(resolvedFast, resolvedPowerful);
  config = applyAutoRouting(config, {
    enabled: true,
    classifier: "rules",
    rules,
  });

  await prompter.note(
    [
      "Auto routing configured:",
      `  Simple/Moderate → ${resolvedFast}`,
      `  Complex/Expert  → ${resolvedPowerful}`,
      "",
      "Edit config.yaml to fine-tune thresholds or switch to LLM classifier.",
    ].join("\n"),
    "Auto Routing",
  );

  return config;
}
