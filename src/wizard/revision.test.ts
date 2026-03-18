import { describe, expect, it, vi } from "vitest";
import type { WizardPrompter } from "./prompts.js";

const mocks = vi.hoisted(() => ({
  randomToken: vi.fn(),
  findTailscaleBinary: vi.fn(async (_opts?: unknown) => "/usr/bin/tailscale"),
}));

vi.mock("../commands/onboard-helpers.js", async (importActual) => {
  const actual = await importActual<typeof import("../commands/onboard-helpers.js")>();
  return {
    ...actual,
    randomToken: mocks.randomToken,
  };
});

vi.mock("../infra/tailscale.js", () => ({
  findTailscaleBinary: (...args: [unknown]) => mocks.findTailscaleBinary(...args),
}));

import {
  applyWizardRevision,
  describeSupportedWizardRevisions,
  parseWizardRevisionInput,
  promptWizardRevisionFallback,
} from "./revision.js";

function createPrompter(params?: {
  textQueue?: Array<string | undefined>;
  selectQueue?: Array<unknown>;
}): WizardPrompter {
  const queue = [...(params?.textQueue ?? [])];
  const selectQueue = [...(params?.selectQueue ?? [])];
  return {
    intro: vi.fn(async () => {}),
    outro: vi.fn(async () => {}),
    note: vi.fn(async () => {}),
    select: vi.fn(async () => selectQueue.shift()) as WizardPrompter["select"],
    multiselect: vi.fn(async () => []),
    text: vi.fn(async () => queue.shift() as string),
    confirm: vi.fn(async () => false),
    progress: vi.fn(() => ({ update: vi.fn(), stop: vi.fn() })),
  };
}

describe("parseWizardRevisionInput", () => {
  it("extracts multiple supported intents from one sentence", () => {
    expect(
      parseWizardRevisionInput(
        "switch default model to openai/gpt-5.2 and set gateway bind to loopback",
      ),
    ).toEqual([
      { kind: "set-default-model", model: "openai/gpt-5.2" },
      { kind: "set-gateway-bind", bind: "loopback" },
    ]);
  });

  it("returns an empty list for unsupported freeform text", () => {
    expect(parseWizardRevisionInput("make it smarter")).toEqual([]);
  });
});

describe("applyWizardRevision", () => {
  it("updates the default model without requiring a gateway restart", async () => {
    const result = await applyWizardRevision({
      input: "switch default model to openai/gpt-5.2",
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter(),
    });

    expect(result.recognized).toBe(true);
    expect(result.changed).toBe(true);
    expect(result.restartGateway).toBe(false);
    expect(result.nextConfig.agents?.defaults?.model).toEqual({ primary: "openai/gpt-5.2" });
    expect(result.notes).toContain("Primary model updated to openai/gpt-5.2.");
  });

  it("switches gateway auth to token and generates a token when needed", async () => {
    mocks.randomToken.mockReturnValueOnce("generated-token");

    const result = await applyWizardRevision({
      input: "use token auth",
      nextConfig: {
        gateway: {
          auth: {
            mode: "password",
            password: "secret-password",
          },
        },
      },
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "password",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter(),
    });

    expect(result.changed).toBe(true);
    expect(result.restartGateway).toBe(true);
    expect(result.settings.authMode).toBe("token");
    expect(result.settings.gatewayToken).toBe("generated-token");
    expect(result.nextConfig.gateway?.auth).toMatchObject({
      mode: "token",
      token: "generated-token",
    });
  });

  it("enforces funnel constraints by switching bind and auth", async () => {
    const prompter = createPrompter({ textQueue: ["secret-password"] });
    const result = await applyWizardRevision({
      input: "turn tailscale funnel",
      nextConfig: {
        gateway: {
          auth: {
            mode: "token",
            token: "old-token",
          },
        },
      },
      settings: {
        port: 4242,
        bind: "lan",
        authMode: "token",
        gatewayToken: "old-token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter,
    });

    expect(result.changed).toBe(true);
    expect(result.restartGateway).toBe(true);
    expect(result.settings.tailscaleMode).toBe("funnel");
    expect(result.settings.bind).toBe("loopback");
    expect(result.settings.authMode).toBe("password");
    expect(result.nextConfig.gateway?.auth).toMatchObject({
      mode: "password",
      password: "secret-password",
    });
    expect(result.notes).toContain(
      "Adjusted gateway bind to loopback because Tailscale exposure requires it.",
    );
    expect(result.notes).toContain(
      "Adjusted gateway auth to password because Tailscale funnel requires it.",
    );
  });

  it("prompts for a custom IP when one is not provided inline", async () => {
    const prompter = createPrompter({ textQueue: ["192.168.1.44"] });
    const result = await applyWizardRevision({
      input: "set gateway bind to custom",
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter,
    });

    expect(result.settings.bind).toBe("custom");
    expect(result.settings.customBindHost).toBe("192.168.1.44");
    expect(result.nextConfig.gateway).toMatchObject({
      bind: "custom",
      customBindHost: "192.168.1.44",
    });
  });

  it("reports unsupported requests cleanly", async () => {
    const result = await applyWizardRevision({
      input: "make it smarter",
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter(),
    });

    expect(result.recognized).toBe(false);
    expect(result.changed).toBe(false);
    expect(describeSupportedWizardRevisions()).toContain("switch default model");
  });
});

describe("promptWizardRevisionFallback", () => {
  it("offers a structured fallback for model changes", async () => {
    const result = await promptWizardRevisionFallback({
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter({
        selectQueue: ["model"],
        textQueue: ["openai/gpt-5.2"],
      }),
    });

    expect(result?.changed).toBe(true);
    expect(result?.nextConfig.agents?.defaults?.model).toEqual({ primary: "openai/gpt-5.2" });
  });

  it("supports a structured fallback for custom bind changes", async () => {
    const result = await promptWizardRevisionFallback({
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter({
        selectQueue: ["bind", "custom"],
        textQueue: ["192.168.1.77"],
      }),
    });

    expect(result?.changed).toBe(true);
    expect(result?.settings.bind).toBe("custom");
    expect(result?.settings.customBindHost).toBe("192.168.1.77");
  });

  it("allows cancelling the structured fallback", async () => {
    const result = await promptWizardRevisionFallback({
      nextConfig: {},
      settings: {
        port: 4242,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
      prompter: createPrompter({
        selectQueue: ["cancel"],
      }),
    });

    expect(result).toBeNull();
  });
});
