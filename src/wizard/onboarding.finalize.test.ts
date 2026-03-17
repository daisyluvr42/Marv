import { describe, expect, it } from "vitest";
import { formatOnboardingSetupSummary } from "./onboarding.finalize.js";

describe("formatOnboardingSetupSummary", () => {
  it("includes the primary model and gateway summary when configured", () => {
    expect(
      formatOnboardingSetupSummary({
        nextConfig: {
          agents: {
            defaults: {
              model: { primary: "openai/gpt-5.2" },
            },
          },
          channels: {
            telegram: { botToken: "123:ABC" },
          },
          tools: {
            web: {
              search: {
                apiKey: "brave-key",
              },
            },
          },
        },
        workspaceDir: "~/marv-workspace",
        settings: {
          port: 4242,
          bind: "loopback",
          authMode: "token",
          gatewayToken: "token",
          tailscaleMode: "off",
          tailscaleResetOnExit: false,
        },
      }),
    ).toContain("Primary model: openai/gpt-5.2");
    expect(
      formatOnboardingSetupSummary({
        nextConfig: {
          agents: {
            defaults: {
              model: { primary: "openai/gpt-5.2" },
            },
          },
          channels: {
            telegram: { botToken: "123:ABC" },
          },
          tools: {
            web: {
              search: {
                apiKey: "brave-key",
              },
            },
          },
        },
        workspaceDir: "~/marv-workspace",
        settings: {
          port: 4242,
          bind: "loopback",
          authMode: "token",
          gatewayToken: "token",
          tailscaleMode: "off",
          tailscaleResetOnExit: false,
        },
      }),
    ).toContain("Capabilities: Control UI, Web search, 1 chat channel configured");
  });

  it("falls back cleanly when no model is configured yet", () => {
    expect(
      formatOnboardingSetupSummary({
        nextConfig: {},
        workspaceDir: "/tmp/marv-workspace",
        settings: {
          port: 4242,
          bind: "lan",
          authMode: "password",
          tailscaleMode: "serve",
          tailscaleResetOnExit: false,
        },
      }),
    ).toContain("Primary model: not configured yet");
    expect(
      formatOnboardingSetupSummary({
        nextConfig: {
          gateway: {
            controlUi: {
              enabled: false,
            },
          },
        },
        workspaceDir: "/tmp/marv-workspace",
        settings: {
          port: 4242,
          bind: "lan",
          authMode: "password",
          tailscaleMode: "serve",
          tailscaleResetOnExit: false,
        },
      }),
    ).toContain("Capabilities: Core local setup");
  });
});
