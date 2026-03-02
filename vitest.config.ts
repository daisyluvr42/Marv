import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, defineProject } from "vitest/config";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));
const isCI = process.env.CI === "true" || process.env.GITHUB_ACTIONS === "true";
const isWindows = process.platform === "win32";
const localWorkers = Math.max(4, Math.min(16, os.cpus().length));
const ciWorkers = isWindows ? 2 : 3;
const defaultMaxWorkers = isCI ? ciWorkers : localWorkers;
const liveEnabled = process.env.MARV_LIVE_TEST === "1" || process.env.CLAWDBOT_LIVE_TEST === "1";

const baseExclude = [
  "dist/**",
  "apps/**",
  "**/node_modules/**",
  "**/vendor/**",
  "dist/Marv.app/**",
  "**/*.e2e.test.ts",
] as const;

const unitFastExtraExclude = [
  "src/plugins/loader.test.ts",
  "src/plugins/tools.optional.test.ts",
  "src/agents/session-tool-result-guard.tool-result-persist-hook.test.ts",
  "src/security/fix.test.ts",
  "src/security/audit.test.ts",
  "src/utils.test.ts",
  "src/auto-reply/tool-meta.test.ts",
  "src/auto-reply/envelope.test.ts",
  "src/commands/auth-choice.test.ts",
  "src/media/store.test.ts",
  "src/media/store.header-ext.test.ts",
  "src/channels/web/media.test.ts",
  "src/channels/web/auto-reply.web-auto-reply.falls-back-text-media-send-fails.test.ts",
  "src/browser/server.covers-additional-endpoint-branches.test.ts",
  "src/browser/server.post-tabs-open-profile-unknown-returns-404.test.ts",
  "src/browser/server.agent-contract-snapshot-endpoints.test.ts",
  "src/browser/server.agent-contract-form-layout-act-commands.test.ts",
  "src/browser/server.skips-default-maxchars-explicitly-set-zero.test.ts",
  "src/browser/server.auth-token-gates-http.test.ts",
  "src/auto-reply/reply.block-streaming.test.ts",
  "src/hooks/install.test.ts",
  "src/channels/telegram/bot.create-telegram-bot.test.ts",
  "src/channels/telegram/bot.test.ts",
  "src/channels/slack/monitor/slash.test.ts",
  "src/channels/imessage/monitor.shutdown.unhandled-rejection.test.ts",
  "src/process/exec.test.ts",
] as const;

const sharedProjectTest = {
  testTimeout: 120_000,
  hookTimeout: isWindows ? 180_000 : 120_000,
  unstubEnvs: true,
  unstubGlobals: true,
  setupFiles: ["test/setup.ts"],
  maxWorkers: defaultMaxWorkers,
} as const;

const pluginSdkAliases = [
  {
    find: "marv/plugin-sdk/account-id",
    replacement: path.join(repoRoot, "src", "plugin-sdk", "account-id.ts"),
  },
  {
    find: "marv/plugin-sdk",
    replacement: path.join(repoRoot, "src", "plugin-sdk", "index.ts"),
  },
] as const;

const pluginSdkProjectConfig = {
  resolve: { alias: pluginSdkAliases },
  server: { deps: { inline: [/^marv\/plugin-sdk(?:\/.*)?$/] } },
} as const;

export default defineConfig({
  resolve: {
    // Keep this ordered: the base `marv/plugin-sdk` alias is a prefix match.
    alias: pluginSdkAliases,
  },
  server: {
    deps: {
      inline: [/^marv\/plugin-sdk(?:\/.*)?$/],
    },
  },
  test: {
    alias: pluginSdkAliases,
    projects: [
      defineProject({
        ...pluginSdkProjectConfig,
        test: {
          ...sharedProjectTest,
          name: "unit-fast",
          pool: "vmForks",
          include: [
            "src/**/*.test.ts",
            "test/**/*.test.ts",
            "ui/src/ui/views/usage-render-details.test.ts",
          ],
          exclude: [
            ...baseExclude,
            "**/*.live.test.ts",
            "src/core/gateway/**",
            ...unitFastExtraExclude,
          ],
        },
      }),
      defineProject({
        ...pluginSdkProjectConfig,
        test: {
          ...sharedProjectTest,
          name: "unit",
          pool: "forks",
          include: [
            "src/**/*.test.ts",
            "test/**/*.test.ts",
            "ui/src/ui/views/usage-render-details.test.ts",
          ],
          exclude: [...baseExclude, "**/*.live.test.ts", "src/core/gateway/**"],
        },
      }),
      defineProject({
        ...pluginSdkProjectConfig,
        test: {
          ...sharedProjectTest,
          name: "extensions",
          pool: "vmForks",
          include: ["extensions/**/*.test.ts"],
          exclude: [...baseExclude, "**/*.live.test.ts"],
        },
      }),
      defineProject({
        ...pluginSdkProjectConfig,
        test: {
          ...sharedProjectTest,
          name: "gateway",
          pool: "forks",
          include: ["src/core/gateway/**/*.test.ts"],
          exclude: [...baseExclude, "**/*.live.test.ts"],
        },
      }),
      defineProject({
        ...pluginSdkProjectConfig,
        test: {
          ...sharedProjectTest,
          name: "live",
          pool: "forks",
          maxWorkers: 1,
          include: liveEnabled ? ["src/**/*.live.test.ts"] : [],
          exclude: baseExclude,
        },
      }),
    ],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      // Keep coverage stable without an ever-growing exclude list:
      // only count files actually exercised by the test suite.
      all: false,
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 55,
        statements: 70,
      },
      // Anchor to repo-root `src/` only. Without this, coverage globs can
      // unintentionally match nested `*/src/**` folders (extensions, apps, etc).
      include: ["./src/**/*.ts"],
      exclude: [
        // Never count workspace packages/apps toward core coverage thresholds.
        "extensions/**",
        "apps/**",
        "ui/**",
        "test/**",
        "src/**/*.test.ts",
        // Entrypoints and wiring (covered by CI smoke + manual/e2e flows).
        "src/entry.ts",
        "src/index.ts",
        "src/runtime.ts",
        "src/channel-web.ts",
        "src/extensionAPI.ts",
        "src/logging.ts",
        "src/cli/**",
        "src/commands/**",
        "src/infra/daemon/**",
        "src/hooks/**",
        "src/macos/**",

        // Large integration surfaces; validated via e2e/manual/contract tests.
        "src/acp/**",
        "src/agents/**",
        "src/channels/**",
        "src/gateway/**",
        "src/line/**",
        "src/media-understanding/**",
        "src/node-host/**",
        "src/plugins/**",
        "src/providers/**",

        // Some agent integrations are intentionally validated via manual/e2e runs.
        "src/agents/model-scan.ts",
        "src/agents/pi-embedded-runner.ts",
        "src/agents/sandbox-paths.ts",
        "src/agents/sandbox.ts",
        "src/agents/skills-install.ts",
        "src/agents/pi-tool-definition-adapter.ts",
        "src/agents/tools/discord-actions*.ts",
        "src/agents/tools/slack-actions.ts",

        // Hard-to-unit-test modules; exercised indirectly by integration tests.
        "src/infra/state-migrations.ts",
        "src/infra/skills-remote.ts",
        "src/infra/update/update-check.ts",
        "src/infra/ports-inspect.ts",
        "src/infra/outbound/outbound-session.ts",
        "src/memory/batch-gemini.ts",

        // Gateway server integration surfaces are intentionally validated via manual/e2e runs.
        "src/gateway/control-ui.ts",
        "src/gateway/server-bridge.ts",
        "src/gateway/server-channels.ts",
        "src/gateway/server-methods/config.ts",
        "src/gateway/server-methods/send.ts",
        "src/gateway/server-methods/skills.ts",
        "src/gateway/server-methods/talk.ts",
        "src/gateway/server-methods/web.ts",
        "src/gateway/server-methods/wizard.ts",

        // Process bridges are hard to unit-test in isolation.
        "src/gateway/call.ts",
        "src/process/tau-rpc.ts",
        "src/process/exec.ts",
        // Interactive UIs/flows are intentionally validated via manual/e2e runs.
        "src/tui/**",
        "src/wizard/**",
        // Channel surfaces are largely integration-tested (or manually validated).
        "src/discord/**",
        "src/imessage/**",
        "src/signal/**",
        "src/slack/**",
        "src/browser/**",
        "src/channels/web/**",
        "src/telegram/index.ts",
        "src/telegram/proxy.ts",
        "src/telegram/webhook-set.ts",
        "src/telegram/**",
        "src/webchat/**",
        "src/gateway/server.ts",
        "src/gateway/client.ts",
        "src/gateway/protocol/**",
        "src/infra/tailscale.ts",
      ],
    },
  },
});
