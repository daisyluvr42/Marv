import type { ApplyAuthChoiceParams, ApplyAuthChoiceResult } from "./auth-choice.apply.js";
import { promptCustomApiConfig } from "./onboard-custom.js";

export async function applyAuthChoiceCustomApi(
  params: ApplyAuthChoiceParams,
): Promise<ApplyAuthChoiceResult | null> {
  if (params.authChoice !== "custom-api-key") {
    return null;
  }

  const result = await promptCustomApiConfig({
    prompter: params.prompter,
    runtime: params.runtime,
    config: params.config,
  });

  return { config: result.config };
}
