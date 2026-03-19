import type { MarvConfig } from "../core/config/config.js";
import type { AutoRoutingConfig } from "../core/config/types.agent-defaults.js";
import type { WizardPrompter } from "../wizard/prompts.js";

function applyAutoRouting(cfg: MarvConfig, routing: AutoRoutingConfig): MarvConfig {
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
 * Onboarding step: prompt user to enable subagent auto model routing.
 *
 * When enabled, subagents automatically pick models from the configured
 * model pool based on task complexity (simple tasks → cheaper/faster models,
 * complex tasks → more capable models). The main session model is not
 * affected — users choose/switch it manually.
 *
 * Returns the updated config (unchanged if user declines).
 */
export async function promptAutoRouting(params: {
  config: MarvConfig;
  prompter: WizardPrompter;
}): Promise<MarvConfig> {
  const { prompter } = params;
  let config = params.config;

  const enable = await prompter.confirm({
    message:
      "Enable subagent auto routing? (subagents pick models from your pool based on task complexity)",
    initialValue: false,
  });

  if (!enable) {
    // Explicitly persist the user's choice so that sessions_spawn
    // (which checks `autoRouting.enabled === true`) does not auto-route.
    config = applyAutoRouting(config, {
      enabled: false,
    });
    return config;
  }

  // No model selection needed — subagent auto-routing uses the model pool
  // directly, reordering candidates by tier based on complexity.
  config = applyAutoRouting(config, {
    enabled: true,
    classifier: "rules",
  });

  await prompter.note(
    [
      "Subagent auto routing enabled.",
      "",
      "Subagents will automatically pick models from your model pool:",
      "  Simple/Moderate tasks → faster, lower-tier models",
      "  Complex/Expert tasks  → more capable, higher-tier models",
      "",
      "Your main session model is not affected by auto routing.",
      "Edit config to fine-tune thresholds or switch to LLM classifier.",
    ].join("\n"),
    "Subagent Auto Routing",
  );

  return config;
}
