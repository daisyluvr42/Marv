import { DEFAULT_PROVIDER } from "../agents/defaults.js";
import { parseModelRef } from "../agents/model/model-selection.js";
import { formatCliCommand } from "../cli/command-format.js";
import { resolvePrimaryModel } from "../commands/model-default.js";
import { describeProbeSummary, runAuthProbes } from "../commands/models/list.probe.js";
import type {
  GatewayAuthChoice,
  OnboardMode,
  OnboardOptions,
  ResetScope,
} from "../commands/onboard-types.js";
import type { MarvConfig } from "../core/config/config.js";
import {
  DEFAULT_GATEWAY_PORT,
  readConfigFileSnapshot,
  resolveGatewayPort,
  writeConfigFile,
} from "../core/config/config.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { resolveUserPath } from "../utils.js";
import { formatWizardMetrics, instrumentWizardPrompter } from "./metrics.js";
import {
  buildWizardStagePlan,
  createWizardStageController,
  presentWizardStage,
} from "./onboarding.stages.js";
import type { QuickstartGatewayDefaults, WizardFlow } from "./onboarding.types.js";
import { WizardCancelledError, type WizardPrompter } from "./prompts.js";

async function requireRiskAcknowledgement(params: {
  opts: OnboardOptions;
  prompter: WizardPrompter;
}) {
  if (params.opts.acceptRisk === true) {
    return;
  }

  await params.prompter.note(
    [
      "Security warning — please read.",
      "",
      "Marv is a hobby project and still in beta. Expect sharp edges.",
      "This bot can read files and run actions if tools are enabled.",
      "A bad prompt can trick it into doing unsafe things.",
      "",
      "If you’re not comfortable with basic security and access control, don’t run Marv.",
      "Ask someone experienced to help before enabling tools or exposing it to the internet.",
      "",
      "Recommended baseline:",
      "- Pairing/allowlists + mention gating.",
      "- Sandbox + least-privilege tools.",
      "- Keep secrets out of the agent’s reachable filesystem.",
      "- Use the strongest available model for any bot with tools or untrusted inboxes.",
      "",
      "Run regularly:",
      "marv security audit --deep",
      "marv security audit --fix",
      "",
      "Must read: /gateway/security",
    ].join("\n"),
    "Security",
  );

  const ok = await params.prompter.confirm({
    message: "I understand this is powerful and inherently risky. Continue?",
    initialValue: false,
  });
  if (!ok) {
    throw new WizardCancelledError("risk not accepted");
  }
}

async function validateConfiguredModelEarly(params: {
  config: MarvConfig;
  prompter: WizardPrompter;
}) {
  const configured = resolvePrimaryModel(params.config.agents?.defaults?.model)?.trim();
  if (!configured) {
    return;
  }

  const parsed = parseModelRef(configured, DEFAULT_PROVIDER);
  if (!parsed) {
    return;
  }

  const progress = params.prompter.progress("Model check");
  try {
    progress.update(`Checking ${parsed.provider}/${parsed.model}…`);
    const summary = await runAuthProbes({
      cfg: params.config,
      providers: [parsed.provider],
      modelCandidates: [`${parsed.provider}/${parsed.model}`],
      options: {
        provider: parsed.provider,
        timeoutMs: 8_000,
        concurrency: 1,
        maxTokens: 16,
      },
    });
    const ok = summary.results.find((result) => result.status === "ok");
    if (ok) {
      await params.prompter.note(
        [
          `Validated model: ${ok.model ?? `${parsed.provider}/${parsed.model}`}`,
          describeProbeSummary(summary),
        ].join("\n"),
        "Model ready",
      );
      return;
    }

    const firstFailure = summary.results[0];
    await params.prompter.note(
      [
        `Could not validate ${parsed.provider}/${parsed.model} yet.`,
        describeProbeSummary(summary),
        firstFailure?.error ? `Reason: ${firstFailure.error}` : undefined,
        `Check later: ${formatCliCommand("marv models list --probe")}`,
      ]
        .filter(Boolean)
        .join("\n"),
      "Model check",
    );
  } catch (error) {
    await params.prompter.note(
      [
        `Could not validate ${parsed.provider}/${parsed.model} yet.`,
        error instanceof Error ? error.message : String(error),
        `Check later: ${formatCliCommand("marv models list --probe")}`,
      ].join("\n"),
      "Model check",
    );
  } finally {
    progress.stop("Model check complete.");
  }
}

function inferExistingOnboardMode(config: MarvConfig): OnboardMode | undefined {
  const hasRemote = Boolean(config.gateway?.remote?.url?.trim());
  const hasLocalWorkspace = Boolean(config.agents?.defaults?.workspace?.trim());
  const hasLocalGateway =
    typeof config.gateway?.port === "number" ||
    config.gateway?.bind !== undefined ||
    config.gateway?.auth?.mode !== undefined ||
    Boolean(config.gateway?.auth?.token) ||
    Boolean(config.gateway?.auth?.password) ||
    config.gateway?.customBindHost !== undefined ||
    config.gateway?.tailscale?.mode !== undefined;
  const hasLocal = hasLocalWorkspace || hasLocalGateway;
  if (hasLocal && !hasRemote) {
    return "local";
  }
  if (hasRemote && !hasLocal) {
    return "remote";
  }
  return undefined;
}

export async function runOnboardingWizard(
  opts: OnboardOptions,
  runtime: RuntimeEnv = defaultRuntime,
  prompter: WizardPrompter,
) {
  const instrumented = instrumentWizardPrompter(prompter);
  prompter = instrumented.prompter;
  const onboardHelpers = await import("../commands/onboard-helpers.js");
  try {
    onboardHelpers.printWizardHeader(runtime);
    await prompter.intro("Marv onboarding");
    const explicitFlowRaw = opts.flow?.trim();
    const normalizedExplicitFlow = explicitFlowRaw === "manual" ? "advanced" : explicitFlowRaw;
    if (
      normalizedExplicitFlow &&
      normalizedExplicitFlow !== "quickstart" &&
      normalizedExplicitFlow !== "advanced"
    ) {
      runtime.error("Invalid --flow (use quickstart, manual, or advanced).");
      runtime.exit(1);
      return;
    }
    const explicitFlow: WizardFlow | undefined =
      normalizedExplicitFlow === "quickstart" || normalizedExplicitFlow === "advanced"
        ? normalizedExplicitFlow
        : undefined;
    await presentWizardStage(prompter, "environment", { flow: explicitFlow });
    await requireRiskAcknowledgement({ opts, prompter });

    const snapshot = await readConfigFileSnapshot();
    let baseConfig: MarvConfig = snapshot.valid ? snapshot.config : {};
    let reuseExistingLocalSetup = false;

    if (snapshot.exists && !snapshot.valid) {
      await prompter.note(onboardHelpers.summarizeExistingConfig(baseConfig), "Invalid config");
      if (snapshot.issues.length > 0) {
        await prompter.note(
          [
            ...snapshot.issues.map((iss) => `- ${iss.path}: ${iss.message}`),
            "",
            "Docs: /gateway/configuration",
          ].join("\n"),
          "Config issues",
        );
      }
      await prompter.outro(
        `Config invalid. Run \`${formatCliCommand("marv doctor")}\` to repair it, then re-run onboarding.`,
      );
      runtime.exit(1);
      return;
    }

    const quickstartHint = `Configure details later via ${formatCliCommand("marv configure")}.`;
    const manualHint = "Configure port, network, Tailscale, and auth options.";
    let flow: WizardFlow =
      explicitFlow ??
      (await prompter.select({
        message: "Onboarding mode",
        options: [
          { value: "quickstart", label: "QuickStart", hint: quickstartHint },
          { value: "advanced", label: "Manual", hint: manualHint },
        ],
        initialValue: "quickstart",
      }));

    if (opts.mode === "remote" && flow === "quickstart") {
      await prompter.note(
        "QuickStart only supports local gateways. Switching to Manual mode.",
        "QuickStart",
      );
      flow = "advanced";
    }

    if (snapshot.exists) {
      await prompter.note(
        onboardHelpers.summarizeExistingConfig(baseConfig),
        "Existing config detected",
      );

      const action = await prompter.select({
        message: "Config handling",
        options: [
          { value: "keep", label: "Use existing values" },
          { value: "modify", label: "Update values" },
          { value: "reset", label: "Reset" },
        ],
      });

      if (action === "reset") {
        const workspaceDefault =
          baseConfig.agents?.defaults?.workspace ?? onboardHelpers.DEFAULT_WORKSPACE;
        const resetScope = (await prompter.select({
          message: "Reset scope",
          options: [
            { value: "config", label: "Config only" },
            {
              value: "config+creds+sessions",
              label: "Config + creds + sessions",
            },
            {
              value: "full",
              label: "Full reset (config + creds + sessions + workspace)",
            },
          ],
        })) as ResetScope;
        await onboardHelpers.handleReset(resetScope, resolveUserPath(workspaceDefault), runtime);
        baseConfig = {};
      } else if (action === "keep") {
        reuseExistingLocalSetup = true;
      }
    }

    const quickstartGateway: QuickstartGatewayDefaults = (() => {
      const hasExisting =
        typeof baseConfig.gateway?.port === "number" ||
        baseConfig.gateway?.bind !== undefined ||
        baseConfig.gateway?.auth?.mode !== undefined ||
        baseConfig.gateway?.auth?.token !== undefined ||
        baseConfig.gateway?.auth?.password !== undefined ||
        baseConfig.gateway?.customBindHost !== undefined ||
        baseConfig.gateway?.tailscale?.mode !== undefined;

      const bindRaw = baseConfig.gateway?.bind;
      const bind =
        bindRaw === "loopback" ||
        bindRaw === "lan" ||
        bindRaw === "auto" ||
        bindRaw === "custom" ||
        bindRaw === "tailnet"
          ? bindRaw
          : "loopback";

      let authMode: GatewayAuthChoice = "token";
      if (
        baseConfig.gateway?.auth?.mode === "token" ||
        baseConfig.gateway?.auth?.mode === "password"
      ) {
        authMode = baseConfig.gateway.auth.mode;
      } else if (baseConfig.gateway?.auth?.token) {
        authMode = "token";
      } else if (baseConfig.gateway?.auth?.password) {
        authMode = "password";
      }

      const tailscaleRaw = baseConfig.gateway?.tailscale?.mode;
      const tailscaleMode =
        tailscaleRaw === "off" || tailscaleRaw === "serve" || tailscaleRaw === "funnel"
          ? tailscaleRaw
          : "off";

      return {
        hasExisting,
        port: resolveGatewayPort(baseConfig),
        bind,
        authMode,
        tailscaleMode,
        token: baseConfig.gateway?.auth?.token,
        password: baseConfig.gateway?.auth?.password,
        customBindHost: baseConfig.gateway?.customBindHost,
        tailscaleResetOnExit: baseConfig.gateway?.tailscale?.resetOnExit ?? false,
      };
    })();

    if (flow === "quickstart") {
      const formatBind = (value: "loopback" | "lan" | "auto" | "custom" | "tailnet") => {
        if (value === "loopback") {
          return "Loopback (127.0.0.1)";
        }
        if (value === "lan") {
          return "LAN";
        }
        if (value === "custom") {
          return "Custom IP";
        }
        if (value === "tailnet") {
          return "Tailnet (Tailscale IP)";
        }
        return "Auto";
      };
      const formatAuth = (value: GatewayAuthChoice) => {
        if (value === "token") {
          return "Token (default)";
        }
        return "Password";
      };
      const formatTailscale = (value: "off" | "serve" | "funnel") => {
        if (value === "off") {
          return "Off";
        }
        if (value === "serve") {
          return "Serve";
        }
        return "Funnel";
      };
      const quickstartLines = quickstartGateway.hasExisting
        ? [
            "Keeping your current gateway settings:",
            `Gateway port: ${quickstartGateway.port}`,
            `Gateway bind: ${formatBind(quickstartGateway.bind)}`,
            ...(quickstartGateway.bind === "custom" && quickstartGateway.customBindHost
              ? [`Gateway custom IP: ${quickstartGateway.customBindHost}`]
              : []),
            `Gateway auth: ${formatAuth(quickstartGateway.authMode)}`,
            `Tailscale exposure: ${formatTailscale(quickstartGateway.tailscaleMode)}`,
            "Direct to chat channels.",
          ]
        : [
            `Gateway port: ${DEFAULT_GATEWAY_PORT}`,
            "Gateway bind: Loopback (127.0.0.1)",
            "Gateway auth: Token (default)",
            "Tailscale exposure: Off",
            "Direct to chat channels.",
          ];
      await prompter.note(quickstartLines.join("\n"), "QuickStart");
    }

    const localPort = resolveGatewayPort(baseConfig);
    const localUrl = `ws://127.0.0.1:${localPort}`;
    const localProbe = await onboardHelpers.probeGatewayReachable({
      url: localUrl,
      token:
        baseConfig.gateway?.auth?.token ??
        process.env.MARV_GATEWAY_TOKEN ??
        process.env.MARV_GATEWAY_TOKEN,
      password:
        baseConfig.gateway?.auth?.password ??
        process.env.MARV_GATEWAY_PASSWORD ??
        process.env.MARV_GATEWAY_PASSWORD,
    });
    const remoteUrl = baseConfig.gateway?.remote?.url?.trim() ?? "";
    const remoteProbe = remoteUrl
      ? await onboardHelpers.probeGatewayReachable({
          url: remoteUrl,
          token: baseConfig.gateway?.remote?.token,
        })
      : null;

    const inferredMode =
      reuseExistingLocalSetup && opts.mode === undefined
        ? inferExistingOnboardMode(baseConfig)
        : undefined;
    if (reuseExistingLocalSetup && inferredMode) {
      await prompter.note(
        inferredMode === "local"
          ? "Your current setup already looks local, so onboarding will keep using this machine."
          : "Your current setup already looks remote, so onboarding will keep using that gateway.",
        "Setup target",
      );
    }

    const mode =
      opts.mode ??
      inferredMode ??
      (flow === "quickstart"
        ? "local"
        : ((await prompter.select({
            message: "What do you want to set up?",
            options: [
              {
                value: "local",
                label: "Local gateway (this machine)",
                hint: localProbe.ok
                  ? `Gateway reachable (${localUrl})`
                  : `No gateway detected (${localUrl})`,
              },
              {
                value: "remote",
                label: "Remote gateway (info-only)",
                hint: !remoteUrl
                  ? "No remote URL configured yet"
                  : remoteProbe?.ok
                    ? `Gateway reachable (${remoteUrl})`
                    : `Configured but unreachable (${remoteUrl})`,
              },
            ],
          })) as OnboardMode));
    const stageController = createWizardStageController({
      prompter,
      plan: buildWizardStagePlan({ mode }),
      initialStage: "environment",
    });

    if (mode === "remote") {
      await stageController.enter("setup", { flow, mode });
      const { promptRemoteGatewayConfig } = await import("../commands/onboard-remote.js");
      const { logConfigUpdated } = await import("../core/config/logging.js");
      let nextConfig = await promptRemoteGatewayConfig(baseConfig, prompter);
      await stageController.enter("review", { flow, mode });
      nextConfig = onboardHelpers.applyWizardMetadata(nextConfig, { command: "onboard", mode });
      await writeConfigFile(nextConfig);
      logConfigUpdated(runtime);
      await prompter.outro("Remote gateway configured.");
      return;
    }

    let nextConfig: MarvConfig = baseConfig;
    let workspaceDir = resolveUserPath(
      baseConfig.agents?.defaults?.workspace ?? onboardHelpers.DEFAULT_WORKSPACE,
    );
    let settings;

    if (reuseExistingLocalSetup) {
      await stageController.enter("model", {
        flow,
        mode,
        reuseExistingLocalSetup,
      });
      await prompter.note(
        "Reusing your existing local model, workspace, gateway, and channel settings.",
        "Using existing setup",
      );
      settings = {
        port: quickstartGateway.port,
        bind: quickstartGateway.bind,
        customBindHost: quickstartGateway.customBindHost,
        authMode: quickstartGateway.authMode,
        gatewayToken: quickstartGateway.token,
        tailscaleMode: quickstartGateway.tailscaleMode,
        tailscaleResetOnExit: quickstartGateway.tailscaleResetOnExit,
      };
      await validateConfiguredModelEarly({ config: nextConfig, prompter });
    } else {
      await stageController.enter("model", { flow, mode });
      const { ensureAuthProfileStore } = await import("../agents/auth-profiles.js");
      const { promptAuthChoiceGrouped } = await import("../commands/auth-choice-prompt.js");
      const { promptCustomApiConfig } = await import("../commands/onboard-custom.js");
      const { applyAuthChoice, resolvePreferredProviderForAuthChoice, warnIfModelConfigLooksOff } =
        await import("../commands/auth-choice.js");
      const { applyPrimaryModel, promptDefaultModel } = await import("../commands/model-picker.js");

      const authStore = ensureAuthProfileStore(undefined, {
        allowKeychainPrompt: false,
      });
      const authChoiceFromPrompt = opts.authChoice === undefined;
      const authChoice =
        opts.authChoice ??
        (await promptAuthChoiceGrouped({
          prompter,
          store: authStore,
          includeSkip: true,
        }));

      if (authChoice === "custom-api-key") {
        const customResult = await promptCustomApiConfig({
          prompter,
          runtime,
          config: nextConfig,
        });
        nextConfig = customResult.config;
      } else {
        const authResult = await applyAuthChoice({
          authChoice,
          config: nextConfig,
          prompter,
          runtime,
          setDefaultModel: true,
          opts: {
            tokenProvider: opts.tokenProvider,
            token: opts.authChoice === "apiKey" && opts.token ? opts.token : undefined,
          },
        });
        nextConfig = authResult.config;
      }

      if (authChoiceFromPrompt && authChoice !== "custom-api-key") {
        const modelSelection = await promptDefaultModel({
          config: nextConfig,
          prompter,
          allowKeep: true,
          ignoreAllowlist: true,
          includeVllm: true,
          preferredProvider: resolvePreferredProviderForAuthChoice(authChoice),
        });
        if (modelSelection.config) {
          nextConfig = modelSelection.config;
        }
        if (modelSelection.model) {
          nextConfig = applyPrimaryModel(nextConfig, modelSelection.model);
        }
      }

      await warnIfModelConfigLooksOff(nextConfig, prompter);
      if (authChoice !== "skip") {
        await validateConfiguredModelEarly({ config: nextConfig, prompter });
      }

      await stageController.enter("setup", { flow, mode });
      const workspaceInput =
        opts.workspace ??
        (flow === "quickstart"
          ? (baseConfig.agents?.defaults?.workspace ?? onboardHelpers.DEFAULT_WORKSPACE)
          : await prompter.text({
              message: "Workspace directory",
              initialValue:
                baseConfig.agents?.defaults?.workspace ?? onboardHelpers.DEFAULT_WORKSPACE,
            }));

      workspaceDir = resolveUserPath(workspaceInput.trim() || onboardHelpers.DEFAULT_WORKSPACE);

      const { applyOnboardingLocalWorkspaceConfig } = await import("../commands/onboard-config.js");
      nextConfig = applyOnboardingLocalWorkspaceConfig(nextConfig, workspaceDir);

      // Auto model routing (advanced/guided flows only).
      if (flow !== "quickstart") {
        const { promptAutoRouting } = await import("../commands/onboard-auto-routing.js");
        nextConfig = await promptAutoRouting({ config: nextConfig, prompter });
      }

      const { configureGatewayForOnboarding } = await import("./onboarding.gateway-config.js");
      const gateway = await configureGatewayForOnboarding({
        flow,
        baseConfig,
        nextConfig,
        localPort,
        quickstartGateway,
        prompter,
        runtime,
      });
      nextConfig = gateway.nextConfig;
      settings = gateway.settings;

      if (opts.skipChannels ?? opts.skipProviders) {
        await prompter.note("Skipping channel setup.", "Channels");
      } else {
        const { listChannelPlugins } = await import("../channels/plugins/index.js");
        const { setupChannels } = await import("../commands/onboard-channels.js");
        const quickstartAllowFromChannels =
          flow === "quickstart"
            ? listChannelPlugins()
                .filter((plugin) => plugin.meta.quickstartAllowFrom)
                .map((plugin) => plugin.id)
            : [];
        nextConfig = await setupChannels(nextConfig, runtime, prompter, {
          allowSignalInstall: true,
          forceAllowFromChannels: quickstartAllowFromChannels,
          skipDmPolicyPrompt: flow === "quickstart",
          skipConfirm: flow === "quickstart",
          quickstartDefaults: flow === "quickstart",
        });
      }
    }

    if (reuseExistingLocalSetup) {
      await stageController.enter("setup", {
        flow,
        mode,
        reuseExistingLocalSetup,
      });
    }

    await writeConfigFile(nextConfig);
    const { logConfigUpdated } = await import("../core/config/logging.js");
    logConfigUpdated(runtime);
    await onboardHelpers.ensureWorkspaceAndSessions(workspaceDir, runtime, {
      skipBootstrap: Boolean(nextConfig.agents?.defaults?.skipBootstrap),
    });

    if (opts.skipSkills || reuseExistingLocalSetup) {
      await prompter.note(
        reuseExistingLocalSetup ? "Keeping your existing skills setup." : "Skipping skills setup.",
        "Skills",
      );
    } else {
      const { setupSkills } = await import("../commands/onboard-skills.js");
      nextConfig = await setupSkills(nextConfig, workspaceDir, runtime, prompter);
    }

    // Setup hooks (session memory on /new)
    if (reuseExistingLocalSetup) {
      await prompter.note("Keeping your existing internal hooks setup.", "Hooks");
    } else {
      const { setupInternalHooks } = await import("../commands/onboard-hooks.js");
      nextConfig = await setupInternalHooks(nextConfig, runtime, prompter);
    }

    nextConfig = onboardHelpers.applyWizardMetadata(nextConfig, { command: "onboard", mode });
    await writeConfigFile(nextConfig);

    await stageController.enter("review", { flow, mode });
    const { finalizeOnboardingWizard } = await import("./onboarding.finalize.js");
    const { launchedTui } = await finalizeOnboardingWizard({
      flow,
      opts,
      baseConfig,
      nextConfig,
      workspaceDir,
      settings,
      prompter,
      runtime,
    });
    if (launchedTui) {
      return;
    }
  } finally {
    if (process.env.MARV_WIZARD_METRICS === "1") {
      runtime.log(formatWizardMetrics(instrumented.getMetrics()));
    }
  }
}
