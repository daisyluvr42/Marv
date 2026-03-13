import { applyAgentDefaultPrimaryModel, resolvePrimaryModel } from "../commands/model-default.js";
import {
  normalizeGatewayTokenInput,
  randomToken,
  validateGatewayPasswordInput,
} from "../commands/onboard-helpers.js";
import type { MarvConfig } from "../core/config/config.js";
import { findTailscaleBinary } from "../infra/tailscale.js";
import { validateIPv4AddressInput } from "../shared/net/ipv4.js";
import type { GatewayWizardSettings } from "./onboarding.types.js";
import type { WizardPrompter } from "./prompts.js";

type WizardRevisionIntent =
  | { kind: "set-default-model"; model: string }
  | { kind: "set-gateway-auth"; authMode: GatewayWizardSettings["authMode"] }
  | {
      kind: "set-gateway-bind";
      bind: GatewayWizardSettings["bind"];
      customBindHost?: string;
    }
  | { kind: "set-tailscale-mode"; tailscaleMode: GatewayWizardSettings["tailscaleMode"] };

type ApplyWizardRevisionParams = {
  input: string;
  nextConfig: MarvConfig;
  settings: GatewayWizardSettings;
  prompter: WizardPrompter;
};

export type ApplyWizardRevisionResult = {
  recognized: boolean;
  changed: boolean;
  nextConfig: MarvConfig;
  settings: GatewayWizardSettings;
  restartGateway: boolean;
  notes: string[];
};

const MODEL_PATTERNS = [
  /\b(?:set|switch|change|use)?\s*(?:the\s+)?(?:default|primary)?\s*model(?:\s+(?:to|as))?\s+([a-z0-9._-]+\/[a-z0-9._:-]+)/i,
  /\bmodel\s+([a-z0-9._-]+\/[a-z0-9._:-]+)/i,
];

function matchFirstGroup(input: string, patterns: RegExp[]): string | undefined {
  for (const pattern of patterns) {
    const match = input.match(pattern);
    const value = match?.[1]?.trim();
    if (value) {
      return value;
    }
  }
  return undefined;
}

export function describeSupportedWizardRevisions(): string {
  return [
    "Try one of these:",
    '- "switch default model to openai/gpt-5.2"',
    '- "use token auth"',
    '- "set gateway bind to loopback"',
    '- "set gateway bind to custom 192.168.1.100"',
    '- "turn tailscale serve on"',
  ].join("\n");
}

export async function promptWizardRevisionFallback(params: {
  nextConfig: MarvConfig;
  settings: GatewayWizardSettings;
  prompter: WizardPrompter;
}): Promise<ApplyWizardRevisionResult | null> {
  const choice = await params.prompter.select({
    message: "Pick a setup area to change",
    options: [
      { value: "model", label: "Default model" },
      { value: "auth", label: "Gateway auth" },
      { value: "bind", label: "Gateway bind" },
      { value: "tailscale", label: "Tailscale" },
      { value: "cancel", label: "Never mind" },
    ],
    initialValue: "model",
  });

  if (choice === "cancel") {
    return null;
  }

  if (choice === "model") {
    const current = resolvePrimaryModel(params.nextConfig.agents?.defaults?.model)?.trim() ?? "";
    const model = await params.prompter.text({
      message: "Default model",
      initialValue: current,
      placeholder: "provider/model",
      validate: (value) => (value.trim() ? undefined : "Required"),
    });
    return applyWizardRevision({
      ...params,
      input: `switch default model to ${model.trim()}`,
    });
  }

  if (choice === "auth") {
    const authMode = await params.prompter.select<GatewayWizardSettings["authMode"]>({
      message: "Gateway auth",
      options: [
        { value: "token", label: "Token" },
        { value: "password", label: "Password" },
      ],
      initialValue: params.settings.authMode,
    });
    return applyWizardRevision({
      ...params,
      input: `use ${authMode} auth`,
    });
  }

  if (choice === "bind") {
    const bind = await params.prompter.select<GatewayWizardSettings["bind"]>({
      message: "Gateway bind",
      options: [
        { value: "loopback", label: "Loopback (127.0.0.1)" },
        { value: "lan", label: "LAN (0.0.0.0)" },
        { value: "tailnet", label: "Tailnet (Tailscale IP)" },
        { value: "auto", label: "Auto" },
        { value: "custom", label: "Custom IP" },
      ],
      initialValue: params.settings.bind,
    });
    if (bind === "custom") {
      const customBindHost = await params.prompter.text({
        message: "Custom IP address",
        placeholder: "192.168.1.100",
        initialValue: params.settings.customBindHost ?? "",
        validate: validateIPv4AddressInput,
      });
      return applyWizardRevision({
        ...params,
        input: `set gateway bind to custom ${customBindHost.trim()}`,
      });
    }
    return applyWizardRevision({
      ...params,
      input: `set gateway bind to ${bind}`,
    });
  }

  const tailscaleMode = await params.prompter.select<GatewayWizardSettings["tailscaleMode"]>({
    message: "Tailscale exposure",
    options: [
      { value: "off", label: "Off" },
      { value: "serve", label: "Serve" },
      { value: "funnel", label: "Funnel" },
    ],
    initialValue: params.settings.tailscaleMode,
  });
  return applyWizardRevision({
    ...params,
    input: `turn tailscale ${tailscaleMode}`,
  });
}

export function parseWizardRevisionInput(input: string): WizardRevisionIntent[] {
  const normalized = input.trim();
  if (!normalized) {
    return [];
  }

  const intents: WizardRevisionIntent[] = [];
  const model = matchFirstGroup(normalized, MODEL_PATTERNS);
  if (model) {
    intents.push({ kind: "set-default-model", model });
  }

  const lower = normalized.toLowerCase();
  if (
    /\b(?:gateway\s+)?(?:auth|authentication)\b/.test(lower) ||
    /\b(?:token|password)\s+auth\b/.test(lower)
  ) {
    if (/\bpassword\b/.test(lower)) {
      intents.push({ kind: "set-gateway-auth", authMode: "password" });
    } else if (/\btoken\b/.test(lower)) {
      intents.push({ kind: "set-gateway-auth", authMode: "token" });
    }
  }

  const customBindMatch = normalized.match(
    /\b(?:gateway\s+bind|bind|listen)\b(?:\s+(?:to|on|as))?\s+custom(?:\s+ip)?(?:\s+([0-9.]+))?/i,
  );
  if (customBindMatch) {
    intents.push({
      kind: "set-gateway-bind",
      bind: "custom",
      customBindHost: customBindMatch[1]?.trim(),
    });
  } else {
    const bindMatch = normalized.match(
      /\b(?:gateway\s+bind|bind|listen)\b(?:\s+(?:to|on|as))?\s+(loopback|lan|auto|tailnet)\b/i,
    );
    if (bindMatch) {
      intents.push({
        kind: "set-gateway-bind",
        bind: bindMatch[1].toLowerCase() as GatewayWizardSettings["bind"],
      });
    } else if (/\blocal only\b/i.test(normalized)) {
      intents.push({ kind: "set-gateway-bind", bind: "loopback" });
    }
  }

  const tailscaleMatch =
    normalized.match(
      /\btailscale(?:\s+(?:mode|exposure))?(?:\s+(?:to|as))?\s+(off|serve|funnel)\b/i,
    ) ?? normalized.match(/\bturn\s+tailscale\s+(off|serve|funnel)\b/i);
  if (tailscaleMatch) {
    intents.push({
      kind: "set-tailscale-mode",
      tailscaleMode: tailscaleMatch[1].toLowerCase() as GatewayWizardSettings["tailscaleMode"],
    });
  }

  return intents;
}

async function ensureGatewayPassword(
  params: ApplyWizardRevisionParams,
  nextConfig: MarvConfig,
): Promise<string> {
  const existing = String(nextConfig.gateway?.auth?.password ?? "").trim();
  if (existing) {
    return existing;
  }
  const password = await params.prompter.text({
    message: "Gateway password",
    validate: validateGatewayPasswordInput,
  });
  return String(password ?? "").trim();
}

async function ensureCustomBindHost(
  params: ApplyWizardRevisionParams,
  initialValue?: string,
): Promise<string> {
  const trimmed = initialValue?.trim();
  if (trimmed && validateIPv4AddressInput(trimmed) === undefined) {
    return trimmed;
  }
  const input = await params.prompter.text({
    message: "Custom IP address",
    placeholder: "192.168.1.100",
    initialValue: trimmed ?? "",
    validate: validateIPv4AddressInput,
  });
  return String(input ?? "").trim();
}

export async function applyWizardRevision(
  params: ApplyWizardRevisionParams,
): Promise<ApplyWizardRevisionResult> {
  const intents = parseWizardRevisionInput(params.input);
  let nextConfig = params.nextConfig;
  let settings = { ...params.settings };
  let changed = false;
  let restartGateway = false;
  const notes: string[] = [];

  if (intents.length === 0) {
    return {
      recognized: false,
      changed: false,
      nextConfig,
      settings,
      restartGateway,
      notes,
    };
  }

  for (const intent of intents) {
    if (intent.kind === "set-default-model") {
      const current = resolvePrimaryModel(nextConfig.agents?.defaults?.model)?.trim();
      const result = applyAgentDefaultPrimaryModel({
        cfg: nextConfig,
        model: intent.model,
      });
      if (result.changed) {
        nextConfig = result.next;
        changed = true;
        notes.push(`Primary model updated to ${intent.model}.`);
      } else if (current === intent.model) {
        notes.push(`Primary model already set to ${intent.model}.`);
      }
      continue;
    }

    if (intent.kind === "set-gateway-auth") {
      if (settings.authMode === intent.authMode) {
        notes.push(`Gateway auth already uses ${intent.authMode}.`);
        continue;
      }

      if (intent.authMode === "token") {
        const gatewayToken =
          settings.gatewayToken?.trim() ||
          normalizeGatewayTokenInput(nextConfig.gateway?.auth?.token) ||
          randomToken();
        nextConfig = {
          ...nextConfig,
          gateway: {
            ...nextConfig.gateway,
            auth: {
              ...nextConfig.gateway?.auth,
              mode: "token",
              token: gatewayToken,
            },
          },
        };
        settings = {
          ...settings,
          authMode: "token",
          gatewayToken,
        };
      } else {
        const password = await ensureGatewayPassword(params, nextConfig);
        nextConfig = {
          ...nextConfig,
          gateway: {
            ...nextConfig.gateway,
            auth: {
              ...nextConfig.gateway?.auth,
              mode: "password",
              password,
            },
          },
        };
        settings = {
          ...settings,
          authMode: "password",
          gatewayToken: undefined,
        };
      }
      changed = true;
      restartGateway = true;
      notes.push(`Gateway auth updated to ${intent.authMode}.`);
      continue;
    }

    if (intent.kind === "set-gateway-bind") {
      let customBindHost =
        intent.bind === "custom"
          ? await ensureCustomBindHost(params, intent.customBindHost ?? settings.customBindHost)
          : undefined;
      if (
        settings.bind === intent.bind &&
        (intent.bind !== "custom" || settings.customBindHost === customBindHost)
      ) {
        notes.push(
          intent.bind === "custom"
            ? `Gateway bind already uses custom ${customBindHost}.`
            : `Gateway bind already uses ${intent.bind}.`,
        );
        continue;
      }
      nextConfig = {
        ...nextConfig,
        gateway: {
          ...nextConfig.gateway,
          bind: intent.bind,
          ...(intent.bind === "custom" && customBindHost ? { customBindHost } : {}),
          ...(intent.bind !== "custom" ? { customBindHost: undefined } : {}),
        },
      };
      settings = {
        ...settings,
        bind: intent.bind,
        customBindHost,
      };
      changed = true;
      restartGateway = true;
      notes.push(
        intent.bind === "custom"
          ? `Gateway bind updated to custom ${customBindHost}.`
          : `Gateway bind updated to ${intent.bind}.`,
      );
      continue;
    }

    if (intent.kind === "set-tailscale-mode") {
      if (settings.tailscaleMode === intent.tailscaleMode) {
        notes.push(`Tailscale already set to ${intent.tailscaleMode}.`);
        continue;
      }

      if (intent.tailscaleMode !== "off") {
        const tailscaleBin = await findTailscaleBinary();
        if (!tailscaleBin) {
          notes.push("Tailscale binary not found. Continuing with config update anyway.");
        }
      }

      settings = {
        ...settings,
        tailscaleMode: intent.tailscaleMode,
      };
      nextConfig = {
        ...nextConfig,
        gateway: {
          ...nextConfig.gateway,
          tailscale: {
            ...nextConfig.gateway?.tailscale,
            mode: intent.tailscaleMode,
            resetOnExit: settings.tailscaleResetOnExit,
          },
        },
      };
      changed = true;
      restartGateway = true;
      notes.push(`Tailscale updated to ${intent.tailscaleMode}.`);

      if (intent.tailscaleMode !== "off" && settings.bind !== "loopback") {
        settings = {
          ...settings,
          bind: "loopback",
          customBindHost: undefined,
        };
        nextConfig = {
          ...nextConfig,
          gateway: {
            ...nextConfig.gateway,
            bind: "loopback",
            customBindHost: undefined,
          },
        };
        notes.push("Adjusted gateway bind to loopback because Tailscale exposure requires it.");
      }

      if (intent.tailscaleMode === "funnel" && settings.authMode !== "password") {
        const password = await ensureGatewayPassword(params, nextConfig);
        nextConfig = {
          ...nextConfig,
          gateway: {
            ...nextConfig.gateway,
            auth: {
              ...nextConfig.gateway?.auth,
              mode: "password",
              password,
            },
          },
        };
        settings = {
          ...settings,
          authMode: "password",
          gatewayToken: undefined,
        };
        notes.push("Adjusted gateway auth to password because Tailscale funnel requires it.");
      }
    }
  }

  return {
    recognized: true,
    changed,
    nextConfig,
    settings,
    restartGateway,
    notes,
  };
}
