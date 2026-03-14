import fs from "node:fs/promises";
import path from "node:path";
import { DEFAULT_BOOTSTRAP_FILENAME } from "../agents/workspace.js";
import { formatCliCommand } from "../cli/command-format.js";
import {
  buildGatewayInstallPlan,
  gatewayInstallErrorHint,
} from "../commands/daemon-install-helpers.js";
import {
  DEFAULT_GATEWAY_DAEMON_RUNTIME,
  GATEWAY_DAEMON_RUNTIME_OPTIONS,
} from "../commands/daemon-runtime.js";
import { formatHealthCheckFailure } from "../commands/health-format.js";
import { healthCommand } from "../commands/health.js";
import { resolvePrimaryModel } from "../commands/model-default.js";
import {
  detectBrowserOpenSupport,
  formatControlUiSshHint,
  openUrl,
  probeGatewayReachable,
  waitForGatewayReachable,
  resolveControlUiLinks,
} from "../commands/onboard-helpers.js";
import type { OnboardOptions } from "../commands/onboard-types.js";
import { writeConfigFile, type MarvConfig } from "../core/config/config.js";
import { logConfigUpdated } from "../core/config/logging.js";
import { ensureControlUiAssetsBuilt } from "../infra/control-ui-assets.js";
import { resolveGatewayService } from "../infra/daemon/service.js";
import { isSystemdUserServiceAvailable } from "../infra/daemon/systemd.js";
import type { RuntimeEnv } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { restoreTerminalState } from "../terminal/restore.js";
import { runTui } from "../tui/tui.js";
import { resolveUserPath } from "../utils.js";
import { setupOnboardingShellCompletion } from "./onboarding.completion.js";
import type { GatewayWizardSettings, WizardFlow } from "./onboarding.types.js";
import type { WizardPrompter } from "./prompts.js";
import {
  applyWizardRevision,
  describeSupportedWizardRevisions,
  promptWizardRevisionFallback,
} from "./revision.js";

type FinalizeOnboardingOptions = {
  flow: WizardFlow;
  opts: OnboardOptions;
  baseConfig: MarvConfig;
  nextConfig: MarvConfig;
  workspaceDir: string;
  settings: GatewayWizardSettings;
  prompter: WizardPrompter;
  runtime: RuntimeEnv;
};

function listOnboardingCapabilities(params: { nextConfig: MarvConfig; channelsEnabled: number }) {
  const capabilities: string[] = [];
  const controlUiEnabled = params.nextConfig.gateway?.controlUi?.enabled ?? true;
  if (controlUiEnabled) {
    capabilities.push("Control UI");
  }
  const webSearchKey = (params.nextConfig.tools?.web?.search?.apiKey ?? "").trim();
  const webSearchEnv = (process.env.BRAVE_API_KEY ?? "").trim();
  if (webSearchKey || webSearchEnv) {
    capabilities.push("Web search");
  }
  if (params.channelsEnabled > 0) {
    capabilities.push(
      `${params.channelsEnabled} chat channel${params.channelsEnabled === 1 ? "" : "s"} configured`,
    );
  }
  return capabilities;
}

export function formatOnboardingSetupSummary(params: {
  nextConfig: MarvConfig;
  workspaceDir: string;
  settings: GatewayWizardSettings;
}) {
  const primaryModel = resolvePrimaryModel(params.nextConfig.agents?.defaults?.model);
  const channelsEnabled = Object.entries(params.nextConfig.channels ?? {}).filter(
    ([, value]) => value && typeof value === "object",
  ).length;
  const capabilities = listOnboardingCapabilities({
    nextConfig: params.nextConfig,
    channelsEnabled,
  });
  const summaryLines = [
    "Setup summary:",
    `Primary model: ${primaryModel?.trim() || "not configured yet"}`,
    `Workspace: ${resolveUserPath(params.workspaceDir)}`,
    `Gateway: ${params.settings.bind}:${params.settings.port} (${params.settings.authMode})`,
    `Tailscale: ${params.settings.tailscaleMode}`,
    `Capabilities: ${capabilities.join(", ") || "Core local setup"}`,
  ];
  return summaryLines.join("\n");
}

export async function finalizeOnboardingWizard(
  options: FinalizeOnboardingOptions,
): Promise<{ launchedTui: boolean }> {
  const { flow, opts, baseConfig, prompter, runtime } = options;
  let { nextConfig, settings, workspaceDir } = options;

  const withWizardProgress = async <T>(
    label: string,
    options: { doneMessage?: string },
    work: (progress: { update: (message: string) => void }) => Promise<T>,
  ): Promise<T> => {
    const progress = prompter.progress(label);
    try {
      return await work(progress);
    } finally {
      progress.stop(options.doneMessage);
    }
  };

  const systemdAvailable =
    process.platform === "linux" ? await isSystemdUserServiceAvailable() : true;
  if (process.platform === "linux" && !systemdAvailable) {
    await prompter.note(
      "Systemd user services are unavailable. Skipping lingering checks and service install.",
      "Systemd",
    );
  }

  if (process.platform === "linux" && systemdAvailable) {
    const { ensureSystemdUserLingerInteractive } = await import("../commands/systemd-linger.js");
    await ensureSystemdUserLingerInteractive({
      runtime,
      prompter: {
        confirm: prompter.confirm,
        note: prompter.note,
      },
      reason:
        "Linux installs use a systemd user service by default. Without lingering, systemd stops the user session on logout/idle and kills the Gateway.",
      requireConfirm: false,
    });
  }

  const explicitInstallDaemon =
    typeof opts.installDaemon === "boolean" ? opts.installDaemon : undefined;
  let installDaemon: boolean;
  if (explicitInstallDaemon !== undefined) {
    installDaemon = explicitInstallDaemon;
  } else if (process.platform === "linux" && !systemdAvailable) {
    installDaemon = false;
  } else if (flow === "quickstart") {
    installDaemon = true;
  } else {
    installDaemon = await prompter.confirm({
      message: "Install Gateway service (recommended)",
      initialValue: true,
    });
  }

  if (process.platform === "linux" && !systemdAvailable && installDaemon) {
    await prompter.note(
      "Systemd user services are unavailable; skipping service install. Use your container supervisor or `docker compose up -d`.",
      "Gateway service",
    );
    installDaemon = false;
  }

  if (installDaemon) {
    const daemonRuntime =
      flow === "quickstart"
        ? DEFAULT_GATEWAY_DAEMON_RUNTIME
        : await prompter.select({
            message: "Gateway service runtime",
            options: GATEWAY_DAEMON_RUNTIME_OPTIONS,
            initialValue: opts.daemonRuntime ?? DEFAULT_GATEWAY_DAEMON_RUNTIME,
          });
    if (flow === "quickstart") {
      await prompter.note(
        "QuickStart uses Node for the Gateway service (stable + supported).",
        "Gateway service runtime",
      );
    }
    const service = resolveGatewayService();
    const loaded = await service.isLoaded({ env: process.env });
    if (loaded) {
      const action = await prompter.select({
        message: "Gateway service already installed",
        options: [
          { value: "restart", label: "Restart" },
          { value: "reinstall", label: "Reinstall" },
          { value: "skip", label: "Skip" },
        ],
      });
      if (action === "restart") {
        await withWizardProgress(
          "Gateway service",
          { doneMessage: "Gateway service restarted." },
          async (progress) => {
            progress.update("Restarting Gateway service…");
            await service.restart({
              env: process.env,
              stdout: process.stdout,
            });
          },
        );
      } else if (action === "reinstall") {
        await withWizardProgress(
          "Gateway service",
          { doneMessage: "Gateway service uninstalled." },
          async (progress) => {
            progress.update("Uninstalling Gateway service…");
            await service.uninstall({ env: process.env, stdout: process.stdout });
          },
        );
      }
    }

    if (!loaded || (loaded && !(await service.isLoaded({ env: process.env })))) {
      const progress = prompter.progress("Gateway service");
      let installError: string | null = null;
      try {
        progress.update("Preparing Gateway service…");
        const { programArguments, workingDirectory, environment } = await buildGatewayInstallPlan({
          env: process.env,
          port: settings.port,
          token: settings.gatewayToken,
          runtime: daemonRuntime,
          warn: (message, title) => prompter.note(message, title),
          config: nextConfig,
        });

        progress.update("Installing Gateway service…");
        await service.install({
          env: process.env,
          stdout: process.stdout,
          programArguments,
          workingDirectory,
          environment,
        });
      } catch (err) {
        installError = err instanceof Error ? err.message : String(err);
      } finally {
        progress.stop(
          installError ? "Gateway service install failed." : "Gateway service installed.",
        );
      }
      if (installError) {
        await prompter.note(`Gateway service install failed: ${installError}`, "Gateway");
        await prompter.note(gatewayInstallErrorHint(), "Gateway");
      }
    }
  }

  if (!opts.skipHealth) {
    const probeLinks = resolveControlUiLinks({
      bind: nextConfig.gateway?.bind ?? "loopback",
      port: settings.port,
      customBindHost: nextConfig.gateway?.customBindHost,
      basePath: undefined,
    });
    // Daemon install/restart can briefly flap the WS; wait a bit so health check doesn't false-fail.
    await waitForGatewayReachable({
      url: probeLinks.wsUrl,
      token: settings.gatewayToken,
      deadlineMs: 15_000,
    });
    try {
      await healthCommand({ json: false, timeoutMs: 10_000 }, runtime);
    } catch (err) {
      runtime.error(formatHealthCheckFailure(err));
      await prompter.note(
        ["Docs:", "/gateway/health", "/gateway/troubleshooting"].join("\n"),
        "Health check help",
      );
    }
  }

  const controlUiEnabled =
    nextConfig.gateway?.controlUi?.enabled ?? baseConfig.gateway?.controlUi?.enabled ?? true;
  if (!opts.skipUi && controlUiEnabled) {
    const controlUiAssets = await ensureControlUiAssetsBuilt(runtime);
    if (!controlUiAssets.ok && controlUiAssets.message) {
      runtime.error(controlUiAssets.message);
    }
  }

  await prompter.note(
    [
      "Add nodes for extra features:",
      "- macOS app (system + notifications)",
      "- iOS app (camera/canvas)",
      "- Android app (camera/canvas)",
    ].join("\n"),
    "Optional apps",
  );

  const controlUiBasePath =
    nextConfig.gateway?.controlUi?.basePath ?? baseConfig.gateway?.controlUi?.basePath;
  let controlUiOpened = false;
  let controlUiOpenHint: string | undefined;
  let seededInBackground = false;
  let hatchChoice: "tui" | "web" | "later" | "revise" | null = null;
  let launchedTui = false;

  if (opts.skipUi) {
    await prompter.note("Skipping Control UI/TUI prompts.", "Control UI");
  } else {
    // Keep the default interaction budget stable: revision support lives behind the
    // existing post-setup choice instead of adding a new mandatory prompt.
    while (true) {
      const links = resolveControlUiLinks({
        bind: settings.bind,
        port: settings.port,
        customBindHost: settings.customBindHost,
        basePath: controlUiBasePath,
      });
      const authedUrl =
        settings.authMode === "token" && settings.gatewayToken
          ? `${links.httpUrl}#token=${encodeURIComponent(settings.gatewayToken)}`
          : links.httpUrl;
      const gatewayProbe = await probeGatewayReachable({
        url: links.wsUrl,
        token: settings.authMode === "token" ? settings.gatewayToken : undefined,
        password: settings.authMode === "password" ? nextConfig.gateway?.auth?.password : "",
      });
      const gatewayStatusLine = gatewayProbe.ok
        ? "Gateway: reachable"
        : `Gateway: not detected${gatewayProbe.detail ? ` (${gatewayProbe.detail})` : ""}`;
      const bootstrapPath = path.join(resolveUserPath(workspaceDir), DEFAULT_BOOTSTRAP_FILENAME);
      const hasBootstrap = await fs
        .access(bootstrapPath)
        .then(() => true)
        .catch(() => false);

      await prompter.note(
        [
          formatOnboardingSetupSummary({
            nextConfig,
            workspaceDir,
            settings,
          }),
          "",
          `Web UI: ${links.httpUrl}`,
          settings.authMode === "token" && settings.gatewayToken
            ? `Web UI (with token): ${authedUrl}`
            : undefined,
          `Gateway WS: ${links.wsUrl}`,
          gatewayStatusLine,
          "Docs: /web/control-ui",
        ]
          .filter(Boolean)
          .join("\n"),
        "Control UI",
      );

      if (gatewayProbe.ok) {
        if (hasBootstrap) {
          await prompter.note(
            [
              "This is the defining action that makes your agent you.",
              "Please take your time.",
              "The more you tell it, the better the experience will be.",
              'We will send: "Wake up, my friend!"',
            ].join("\n"),
            "Start TUI (best option!)",
          );
        }

        await prompter.note(
          [
            "Gateway token: shared auth for the Gateway + Control UI.",
            "Stored in: ~/.marv/marv.json (gateway.auth.token) or MARV_GATEWAY_TOKEN.",
            `View token: ${formatCliCommand("marv config get gateway.auth.token")}`,
            `Generate token: ${formatCliCommand("marv doctor --generate-gateway-token")}`,
            "Web UI stores a copy in this browser's localStorage (marv.control.settings.v1; legacy key also supported).",
            `Open the dashboard anytime: ${formatCliCommand("marv dashboard --no-open")}`,
            "If prompted: paste the token into Control UI settings (or use the tokenized dashboard URL).",
          ].join("\n"),
          "Token",
        );
      }

      hatchChoice = gatewayProbe.ok
        ? await prompter.select({
            message: "How do you want to hatch your bot?",
            options: [
              { value: "tui", label: "Hatch in TUI (recommended)" },
              { value: "web", label: "Open the Web UI" },
              { value: "revise", label: "Revise setup" },
              { value: "later", label: "Do this later" },
            ],
            initialValue: "tui",
          })
        : await prompter.select({
            message: "Gateway is not reachable yet. What do you want to do next?",
            options: [
              { value: "revise", label: "Revise setup" },
              { value: "later", label: "Do this later" },
            ],
            initialValue: "revise",
          });

      if (hatchChoice === "revise") {
        const revisionInput = await prompter.text({
          message: "What would you like to change?",
          placeholder: "e.g. switch default model to openai/gpt-5.2",
        });
        const revision = await applyWizardRevision({
          input: revisionInput,
          nextConfig,
          settings,
          prompter,
        });
        if (!revision.recognized) {
          await prompter.note(describeSupportedWizardRevisions(), "Supported setup changes");
          const fallbackRevision = await promptWizardRevisionFallback({
            nextConfig,
            settings,
            prompter,
          });
          if (!fallbackRevision) {
            await prompter.note("No setup changes applied.", "Setup unchanged");
            continue;
          }
          if (!fallbackRevision.changed) {
            await prompter.note(
              fallbackRevision.notes.join("\n") ||
                "That request does not change the current setup.",
              "Setup unchanged",
            );
            continue;
          }
          nextConfig = fallbackRevision.nextConfig;
          settings = fallbackRevision.settings;
          await writeConfigFile(nextConfig);
          logConfigUpdated(runtime);
          if (fallbackRevision.restartGateway) {
            const service = resolveGatewayService();
            if (await service.isLoaded({ env: process.env })) {
              await withWizardProgress(
                "Gateway service",
                { doneMessage: "Gateway service restarted." },
                async (progress) => {
                  progress.update("Restarting Gateway service…");
                  await service.restart({
                    env: process.env,
                    stdout: process.stdout,
                  });
                },
              );
            }
          }
          await prompter.note(fallbackRevision.notes.join("\n"), "Setup updated");
          continue;
        }
        if (!revision.changed) {
          await prompter.note(
            revision.notes.join("\n") || "That request does not change the current setup.",
            "Setup unchanged",
          );
          continue;
        }
        nextConfig = revision.nextConfig;
        settings = revision.settings;
        await writeConfigFile(nextConfig);
        logConfigUpdated(runtime);
        if (revision.restartGateway) {
          const service = resolveGatewayService();
          if (await service.isLoaded({ env: process.env })) {
            await withWizardProgress(
              "Gateway service",
              { doneMessage: "Gateway service restarted." },
              async (progress) => {
                progress.update("Restarting Gateway service…");
                await service.restart({
                  env: process.env,
                  stdout: process.stdout,
                });
              },
            );
          }
        }
        await prompter.note(revision.notes.join("\n"), "Setup updated");
        continue;
      }

      if (hatchChoice === "tui") {
        restoreTerminalState("pre-onboarding tui", { resumeStdinIfPaused: true });
        await runTui({
          url: links.wsUrl,
          token: settings.authMode === "token" ? settings.gatewayToken : undefined,
          password: settings.authMode === "password" ? nextConfig.gateway?.auth?.password : "",
          // Safety: onboarding TUI should not auto-deliver to lastProvider/lastTo.
          deliver: false,
          message: hasBootstrap ? "Wake up, my friend!" : undefined,
        });
        launchedTui = true;
      } else if (hatchChoice === "web") {
        const browserSupport = await detectBrowserOpenSupport();
        if (browserSupport.ok) {
          controlUiOpened = await openUrl(authedUrl);
          if (!controlUiOpened) {
            controlUiOpenHint = formatControlUiSshHint({
              port: settings.port,
              basePath: controlUiBasePath,
              token: settings.authMode === "token" ? settings.gatewayToken : undefined,
            });
          }
        } else {
          controlUiOpenHint = formatControlUiSshHint({
            port: settings.port,
            basePath: controlUiBasePath,
            token: settings.authMode === "token" ? settings.gatewayToken : undefined,
          });
        }
        await prompter.note(
          [
            `Dashboard link (with token): ${authedUrl}`,
            controlUiOpened
              ? "Opened in your browser. Keep that tab to control Marv."
              : "Copy/paste this URL in a browser on this machine to control Marv.",
            controlUiOpenHint,
          ]
            .filter(Boolean)
            .join("\n"),
          "Dashboard ready",
        );
      } else {
        await prompter.note(
          `When you're ready: ${formatCliCommand("marv dashboard --no-open")}`,
          "Later",
        );
      }
      break;
    }
  }

  await prompter.note(
    ["Back up your agent workspace.", "Docs: /concepts/agent-workspace"].join("\n"),
    "Workspace backup",
  );

  await prompter.note(
    "Running agents on your computer is risky — harden your setup: /security",
    "Security",
  );

  await setupOnboardingShellCompletion({ flow, prompter });

  const shouldOpenControlUi =
    !opts.skipUi &&
    settings.authMode === "token" &&
    Boolean(settings.gatewayToken) &&
    hatchChoice === null;
  if (shouldOpenControlUi) {
    const browserSupport = await detectBrowserOpenSupport();
    if (browserSupport.ok) {
      controlUiOpened = await openUrl(authedUrl);
      if (!controlUiOpened) {
        controlUiOpenHint = formatControlUiSshHint({
          port: settings.port,
          basePath: controlUiBasePath,
          token: settings.gatewayToken,
        });
      }
    } else {
      controlUiOpenHint = formatControlUiSshHint({
        port: settings.port,
        basePath: controlUiBasePath,
        token: settings.gatewayToken,
      });
    }

    await prompter.note(
      [
        `Dashboard link (with token): ${authedUrl}`,
        controlUiOpened
          ? "Opened in your browser. Keep that tab to control Marv."
          : "Copy/paste this URL in a browser on this machine to control Marv.",
        controlUiOpenHint,
      ]
        .filter(Boolean)
        .join("\n"),
      "Dashboard ready",
    );
  }

  const webSearchKey = (nextConfig.tools?.web?.search?.apiKey ?? "").trim();
  const webSearchEnv = (process.env.BRAVE_API_KEY ?? "").trim();
  const hasWebSearchKey = Boolean(webSearchKey || webSearchEnv);
  await prompter.note(
    hasWebSearchKey
      ? [
          "Web search is enabled, so your agent can look things up online when needed.",
          "",
          webSearchKey
            ? "API key: stored in config (tools.web.search.apiKey)."
            : "API key: provided via BRAVE_API_KEY env var (Gateway environment).",
          "Docs: /tools/web",
        ].join("\n")
      : [
          "If you want your agent to be able to search the web, you’ll need an API key.",
          "",
          "Marv uses Brave Search for the `web_search` tool. Without a Brave Search API key, web search won’t work.",
          "",
          "Set it up interactively:",
          `- Run: ${formatCliCommand("marv configure --section web")}`,
          "- Enable web_search and paste your Brave Search API key",
          "",
          "Alternative: set BRAVE_API_KEY in the Gateway environment (no config changes).",
          "Docs: /tools/web",
        ].join("\n"),
    "Web search (optional)",
  );

  await prompter.note(`What now: ${formatDocsLink("/start/showcase", "showcase")}`, "What now");

  await prompter.outro(
    controlUiOpened
      ? "Onboarding complete. Dashboard opened; keep that tab to control Marv."
      : seededInBackground
        ? "Onboarding complete. Web UI seeded in the background; open it anytime with the dashboard link above."
        : "Onboarding complete. Use the dashboard link above to control Marv.",
  );

  return { launchedTui };
}
