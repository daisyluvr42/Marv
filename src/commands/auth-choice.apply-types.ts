import type { MarvConfig } from "../core/config/config.js";
import type { RuntimeEnv } from "../runtime.js";
import type { WizardPrompter } from "../wizard/prompts.js";
import type { AuthChoice } from "./onboard-types.js";

export type ApplyAuthChoiceParams = {
  authChoice: AuthChoice;
  config: MarvConfig;
  prompter: WizardPrompter;
  runtime: RuntimeEnv;
  agentDir?: string;
  setDefaultModel: boolean;
  agentId?: string;
  opts?: {
    tokenProvider?: string;
    token?: string;
    cloudflareAiGatewayAccountId?: string;
    cloudflareAiGatewayGatewayId?: string;
    cloudflareAiGatewayApiKey?: string;
    xaiApiKey?: string;
  };
};

export type ApplyAuthChoiceResult = {
  config: MarvConfig;
  agentModelOverride?: string;
};
