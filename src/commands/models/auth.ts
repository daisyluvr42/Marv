import { confirm as clackConfirm, text as clackText } from "@clack/prompts";
import {
  resolveAgentDir,
  resolveAgentWorkspaceDir,
  resolveDefaultAgentId,
} from "../../agents/agent-scope.js";
import { ensureAuthProfileStore } from "../../agents/auth-profiles.js";
import { upsertAuthProfile } from "../../agents/auth-profiles.js";
import type { AuthProfileCredential } from "../../agents/auth-profiles/types.js";
import { normalizeProviderId } from "../../agents/model/model-selection.js";
import { resolveDefaultAgentWorkspaceDir } from "../../agents/workspace.js";
import { formatCliCommand } from "../../cli/command-format.js";
import { parseDurationMs } from "../../cli/parse-duration.js";
import type { MarvConfig } from "../../core/config/config.js";
import { logConfigUpdated } from "../../core/config/logging.js";
import { resolvePluginProviders } from "../../plugins/providers.js";
import type { ProviderAuthResult, ProviderPlugin } from "../../plugins/types.js";
import type { RuntimeEnv } from "../../runtime.js";
import { stylePromptMessage } from "../../terminal/prompt-style.js";
import { createClackPrompter } from "../../wizard/clack-prompter.js";
import { promptAuthChoiceGrouped } from "../auth-choice-prompt.js";
import { applyAuthChoice } from "../auth-choice.js";
import { validateAnthropicSetupToken } from "../auth-token.js";
import { isRemoteEnvironment } from "../oauth-env.js";
import { createVpsAwareOAuthHandlers } from "../oauth-flow.js";
import { applyAuthProfileConfig } from "../onboard-auth.js";
import { openUrl } from "../onboard-helpers.js";
import { applyNonInteractiveAuthChoice } from "../onboard-non-interactive/local/auth-choice.js";
import { type OnboardOptions } from "../onboard-types.js";
import {
  applyDefaultModel,
  mergeConfigPatch,
  pickAuthMethod,
  resolveProviderMatch,
} from "../provider-auth-helpers.js";
import { resolveModelsAuthChoice } from "./auth-choice.js";
import { loadValidConfigOrThrow, resolveKnownAgentId, updateConfig } from "./shared.js";

const confirm = (params: Parameters<typeof clackConfirm>[0]) =>
  clackConfirm({
    ...params,
    message: stylePromptMessage(params.message),
  });
const text = (params: Parameters<typeof clackText>[0]) =>
  clackText({
    ...params,
    message: stylePromptMessage(params.message),
  });
type TokenProvider = "anthropic";

function resolveTokenProvider(raw?: string): TokenProvider | "custom" | null {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = normalizeProviderId(trimmed);
  if (normalized === "anthropic") {
    return "anthropic";
  }
  return "custom";
}

function resolveDefaultTokenProfileId(provider: string): string {
  return `${normalizeProviderId(provider)}:manual`;
}

export async function modelsAuthSetupTokenCommand(
  opts: { provider?: string; yes?: boolean },
  runtime: RuntimeEnv,
) {
  const provider = resolveTokenProvider(opts.provider ?? "anthropic");
  if (provider !== "anthropic") {
    throw new Error("Only --provider anthropic is supported for setup-token.");
  }

  if (!process.stdin.isTTY) {
    throw new Error("setup-token requires an interactive TTY.");
  }

  if (!opts.yes) {
    const proceed = await confirm({
      message: "Have you run `claude setup-token` and copied the token?",
      initialValue: true,
    });
    if (!proceed) {
      return;
    }
  }

  const tokenInput = await text({
    message: "Paste Anthropic setup-token",
    validate: (value) => validateAnthropicSetupToken(String(value ?? "")),
  });
  const token = String(tokenInput ?? "").trim();
  const profileId = resolveDefaultTokenProfileId(provider);

  upsertAuthProfile({
    profileId,
    credential: {
      type: "token",
      provider,
      token,
    },
  });

  await updateConfig((cfg) =>
    applyAuthProfileConfig(cfg, {
      profileId,
      provider,
      mode: "token",
    }),
  );

  logConfigUpdated(runtime);
  runtime.log(`Auth profile: ${profileId} (${provider}/token)`);
}

export async function modelsAuthPasteTokenCommand(
  opts: {
    provider?: string;
    profileId?: string;
    expiresIn?: string;
  },
  runtime: RuntimeEnv,
) {
  const rawProvider = opts.provider?.trim();
  if (!rawProvider) {
    throw new Error("Missing --provider.");
  }
  const provider = normalizeProviderId(rawProvider);
  const profileId = opts.profileId?.trim() || resolveDefaultTokenProfileId(provider);

  const tokenInput = await text({
    message: `Paste token for ${provider}`,
    validate: (value) => (value?.trim() ? undefined : "Required"),
  });
  const token = String(tokenInput ?? "").trim();

  const expires =
    opts.expiresIn?.trim() && opts.expiresIn.trim().length > 0
      ? Date.now() + parseDurationMs(String(opts.expiresIn ?? "").trim(), { defaultUnit: "d" })
      : undefined;

  upsertAuthProfile({
    profileId,
    credential: {
      type: "token",
      provider,
      token,
      ...(expires ? { expires } : {}),
    },
  });

  await updateConfig((cfg) => applyAuthProfileConfig(cfg, { profileId, provider, mode: "token" }));

  logConfigUpdated(runtime);
  runtime.log(`Auth profile: ${profileId} (${provider}/token)`);
}

type AddOptions = {
  provider?: string;
  method?: string;
  setDefault?: boolean;
  agent?: string;
};

type SetOptions = {
  provider: string;
  method?: string;
  apiKey?: string;
  token?: string;
  profileId?: string;
  expiresIn?: string;
  baseUrl?: string;
  model?: string;
  compatibility?: "openai" | "anthropic";
  providerId?: string;
  accountId?: string;
  gatewayId?: string;
  setDefault?: boolean;
  agent?: string;
};

function preserveDefaultModelSelection(original: MarvConfig, next: MarvConfig): MarvConfig {
  const originalModel = original.agents?.defaults?.model;
  const nextDefaults = {
    ...next.agents?.defaults,
  };
  if (originalModel === undefined) {
    delete nextDefaults.model;
  } else {
    nextDefaults.model = originalModel;
  }
  return {
    ...next,
    agents: {
      ...next.agents,
      defaults: nextDefaults,
    },
  };
}

function buildModelsAuthSetOnboardOptions(params: {
  authChoice: string;
  opts: SetOptions;
}): OnboardOptions {
  const { authChoice, opts } = params;
  const apiKey = opts.apiKey;
  const token = opts.token;
  const baseUrl = opts.baseUrl;
  const model = opts.model;
  const compatibility = opts.compatibility;
  const providerId = opts.providerId;
  const accountId = opts.accountId;
  const gatewayId = opts.gatewayId;

  switch (authChoice) {
    case "apiKey":
      return { authChoice: "apiKey", anthropicApiKey: apiKey };
    case "token":
      return {
        authChoice: "token",
        tokenProvider: "anthropic",
        token,
        tokenProfileId: opts.profileId,
        tokenExpiresIn: opts.expiresIn,
      };
    case "openai-api-key":
      return { authChoice: "openai-api-key", openaiApiKey: apiKey };
    case "gemini-api-key":
      return { authChoice: "gemini-api-key", geminiApiKey: apiKey };
    case "openrouter-api-key":
      return { authChoice: "openrouter-api-key", openrouterApiKey: apiKey };
    case "litellm-api-key":
      return { authChoice: "litellm-api-key", litellmApiKey: apiKey };
    case "ai-gateway-api-key":
      return { authChoice: "ai-gateway-api-key", aiGatewayApiKey: apiKey };
    case "cloudflare-ai-gateway-api-key":
      return {
        authChoice: "cloudflare-ai-gateway-api-key",
        cloudflareAiGatewayApiKey: apiKey,
        cloudflareAiGatewayAccountId: accountId,
        cloudflareAiGatewayGatewayId: gatewayId,
      };
    case "moonshot-api-key":
      return { authChoice: "moonshot-api-key", moonshotApiKey: apiKey };
    case "moonshot-api-key-cn":
      return { authChoice: "moonshot-api-key-cn", moonshotApiKey: apiKey };
    case "kimi-code-api-key":
      return { authChoice: "kimi-code-api-key", kimiCodeApiKey: apiKey };
    case "synthetic-api-key":
      return { authChoice: "synthetic-api-key", syntheticApiKey: apiKey };
    case "venice-api-key":
      return { authChoice: "venice-api-key", veniceApiKey: apiKey };
    case "together-api-key":
      return { authChoice: "together-api-key", togetherApiKey: apiKey };
    case "huggingface-api-key":
      return { authChoice: "huggingface-api-key", huggingfaceApiKey: apiKey };
    case "zai-api-key":
    case "zai-coding-global":
    case "zai-coding-cn":
    case "zai-global":
    case "zai-cn":
      return { authChoice: authChoice as OnboardOptions["authChoice"], zaiApiKey: apiKey };
    case "xiaomi-api-key":
      return { authChoice: "xiaomi-api-key", xiaomiApiKey: apiKey };
    case "xai-api-key":
      return { authChoice: "xai-api-key", xaiApiKey: apiKey };
    case "qianfan-api-key":
      return { authChoice: "qianfan-api-key", qianfanApiKey: apiKey };
    case "minimax-cloud":
    case "minimax-api":
    case "minimax-api-key-cn":
    case "minimax-api-lightning":
      return { authChoice: authChoice as OnboardOptions["authChoice"], minimaxApiKey: apiKey };
    case "opencode-zen":
      return { authChoice: "opencode-zen", opencodeZenApiKey: apiKey };
    case "custom-api-key":
      return {
        authChoice: "custom-api-key",
        customBaseUrl: baseUrl,
        customApiKey: apiKey,
        customModelId: model,
        customProviderId: providerId,
        customCompatibility: compatibility,
      };
    default:
      return { authChoice: authChoice as OnboardOptions["authChoice"] };
  }
}

export async function modelsAuthAddCommand(opts: AddOptions, runtime: RuntimeEnv) {
  const config = await loadValidConfigOrThrow();
  const agentId = resolveKnownAgentId({ cfg: config, rawAgentId: opts.agent });
  const agentDir = agentId ? resolveAgentDir(config, agentId) : undefined;
  const prompter = createClackPrompter();

  const resolvedChoice = resolveModelsAuthChoice({
    provider: opts.provider,
    method: opts.method,
  });
  let authChoice = resolvedChoice.choice;

  if (opts.provider?.trim() && opts.method?.trim() && !authChoice) {
    throw new Error(`Unknown auth method "${opts.method}" for provider "${opts.provider}".`);
  }

  if (!authChoice && opts.provider?.trim()) {
    const { options } = resolvedChoice;
    if (options.length === 0) {
      throw new Error(`Unknown provider "${opts.provider}".`);
    }
    authChoice =
      options.length === 1
        ? options[0].value
        : await prompter.select({
            message: `${opts.provider.trim()} auth method`,
            options: options.map((option) => ({
              value: option.value,
              label: option.label,
              hint: option.hint,
            })),
          });
  }

  if (!authChoice) {
    authChoice = await promptAuthChoiceGrouped({
      prompter,
      store: ensureAuthProfileStore(agentDir, {
        allowKeychainPrompt: false,
      }),
      includeSkip: false,
    });
  }

  const result = await applyAuthChoice({
    authChoice,
    config,
    prompter,
    runtime,
    agentDir,
    agentId,
    setDefaultModel: Boolean(opts.setDefault),
  });

  await updateConfig(() => result.config);
  logConfigUpdated(runtime);
}

export async function modelsAuthSetCommand(opts: SetOptions, runtime: RuntimeEnv) {
  const config = await loadValidConfigOrThrow();
  const agentId = resolveKnownAgentId({ cfg: config, rawAgentId: opts.agent });
  const agentDir = agentId ? resolveAgentDir(config, agentId) : undefined;
  const resolvedChoice = resolveModelsAuthChoice({
    provider: opts.provider,
    method: opts.method,
  });
  const authChoice = resolvedChoice.choice;

  if (!authChoice) {
    if (resolvedChoice.options.length === 0) {
      throw new Error(`Unknown provider "${opts.provider}".`);
    }
    throw new Error(`Provider "${opts.provider}" has multiple auth methods. Use --method <id>.`);
  }

  const onboardOpts = buildModelsAuthSetOnboardOptions({
    authChoice,
    opts,
  });
  const nextConfig = await applyNonInteractiveAuthChoice({
    nextConfig: config,
    authChoice,
    opts: onboardOpts,
    runtime,
    baseConfig: config,
    agentDir,
  });
  if (!nextConfig) {
    return;
  }

  const finalConfig = opts.setDefault
    ? nextConfig
    : preserveDefaultModelSelection(config, nextConfig);
  await updateConfig(() => finalConfig);
  logConfigUpdated(runtime);
}

type LoginOptions = {
  provider?: string;
  method?: string;
  setDefault?: boolean;
};

export function resolveRequestedLoginProviderOrThrow(
  providers: ProviderPlugin[],
  rawProvider?: string,
): ProviderPlugin | null {
  const requested = rawProvider?.trim();
  if (!requested) {
    return null;
  }
  const matched = resolveProviderMatch(providers, requested);
  if (matched) {
    return matched;
  }
  const available = providers
    .map((provider) => provider.id)
    .filter(Boolean)
    .toSorted((a, b) => a.localeCompare(b));
  const availableText = available.length > 0 ? available.join(", ") : "(none)";
  throw new Error(
    `Unknown provider "${requested}". Loaded providers: ${availableText}. Verify plugins via \`${formatCliCommand("marv plugins list --json")}\`.`,
  );
}

function credentialMode(credential: AuthProfileCredential): "api_key" | "oauth" | "token" {
  if (credential.type === "api_key") {
    return "api_key";
  }
  if (credential.type === "token") {
    return "token";
  }
  return "oauth";
}

export async function modelsAuthLoginCommand(opts: LoginOptions, runtime: RuntimeEnv) {
  if (!process.stdin.isTTY) {
    throw new Error("models auth login requires an interactive TTY.");
  }

  const config = await loadValidConfigOrThrow();
  const defaultAgentId = resolveDefaultAgentId(config);
  const agentDir = resolveAgentDir(config, defaultAgentId);
  const workspaceDir =
    resolveAgentWorkspaceDir(config, defaultAgentId) ?? resolveDefaultAgentWorkspaceDir();

  const providers = resolvePluginProviders({ config, workspaceDir });
  if (providers.length === 0) {
    throw new Error(
      `No provider plugins found. Install one via \`${formatCliCommand("marv plugins install")}\`.`,
    );
  }

  const prompter = createClackPrompter();
  const requestedProvider = resolveRequestedLoginProviderOrThrow(providers, opts.provider);
  const selectedProvider =
    requestedProvider ??
    (await prompter
      .select({
        message: "Select a provider",
        options: providers.map((provider) => ({
          value: provider.id,
          label: provider.label,
          hint: provider.docsPath ? `Docs: ${provider.docsPath}` : undefined,
        })),
      })
      .then((id) => resolveProviderMatch(providers, String(id))));

  if (!selectedProvider) {
    throw new Error("Unknown provider. Use --provider <id> to pick a provider plugin.");
  }

  const chosenMethod =
    pickAuthMethod(selectedProvider, opts.method) ??
    (selectedProvider.auth.length === 1
      ? selectedProvider.auth[0]
      : await prompter
          .select({
            message: `Auth method for ${selectedProvider.label}`,
            options: selectedProvider.auth.map((method) => ({
              value: method.id,
              label: method.label,
              hint: method.hint,
            })),
          })
          .then((id) => selectedProvider.auth.find((method) => method.id === String(id))));

  if (!chosenMethod) {
    throw new Error("Unknown auth method. Use --method <id> to select one.");
  }

  const isRemote = isRemoteEnvironment();
  const result: ProviderAuthResult = await chosenMethod.run({
    config,
    agentDir,
    workspaceDir,
    prompter,
    runtime,
    isRemote,
    openUrl: async (url) => {
      await openUrl(url);
    },
    oauth: {
      createVpsAwareHandlers: (params) => createVpsAwareOAuthHandlers(params),
    },
  });

  for (const profile of result.profiles) {
    upsertAuthProfile({
      profileId: profile.profileId,
      credential: profile.credential,
      agentDir,
    });
  }

  await updateConfig((cfg) => {
    let next = cfg;
    if (result.configPatch) {
      next = mergeConfigPatch(next, result.configPatch);
    }
    for (const profile of result.profiles) {
      next = applyAuthProfileConfig(next, {
        profileId: profile.profileId,
        provider: profile.credential.provider,
        mode: credentialMode(profile.credential),
      });
    }
    if (opts.setDefault && result.defaultModel) {
      next = applyDefaultModel(next, result.defaultModel);
    }
    return next;
  });

  logConfigUpdated(runtime);
  for (const profile of result.profiles) {
    runtime.log(
      `Auth profile: ${profile.profileId} (${profile.credential.provider}/${credentialMode(profile.credential)})`,
    );
  }
  if (result.defaultModel) {
    runtime.log(
      opts.setDefault
        ? `Default model set to ${result.defaultModel}`
        : `Default model available: ${result.defaultModel} (use --set-default to apply)`,
    );
  }
  if (result.notes && result.notes.length > 0) {
    await prompter.note(result.notes.join("\n"), "Provider notes");
  }
}
