import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createExitThrowingRuntime,
  createWizardPrompter,
} from "../../../commands/test-wizard-helpers.js";
import type { MarvConfig } from "../../../core/config/config.js";
import type { WizardPrompter } from "../../../wizard/prompts.js";

const { probeTelegramMock } = vi.hoisted(() => ({
  probeTelegramMock: vi.fn(),
}));

vi.mock("../../telegram/probe.js", () => ({
  probeTelegram: probeTelegramMock,
}));

import { telegramOnboardingAdapter } from "./telegram.js";

function createPrompter(params: {
  texts?: string[];
  confirms?: boolean[];
  note?: WizardPrompter["note"];
}) {
  const texts = [...(params.texts ?? [])];
  const confirms = [...(params.confirms ?? [])];
  return createWizardPrompter({
    text: vi.fn(async () => texts.shift() ?? "") as unknown as WizardPrompter["text"],
    confirm: vi.fn(async () => confirms.shift() ?? false),
    note: params.note ?? (vi.fn(async () => {}) as WizardPrompter["note"]),
    progress: vi.fn(() => ({ update: vi.fn(), stop: vi.fn() })),
  });
}

describe("telegramOnboardingAdapter", () => {
  beforeEach(() => {
    probeTelegramMock.mockReset();
  });

  it("re-prompts until a valid Telegram token is entered", async () => {
    probeTelegramMock
      .mockResolvedValueOnce({
        ok: false,
        status: 401,
        error: "Unauthorized",
      })
      .mockResolvedValueOnce({
        ok: true,
        bot: { username: "valid_bot" },
      });

    const note = vi.fn(async () => {});
    const prompter = createPrompter({
      texts: ["bad-token", "good-token"],
      note,
    });

    const result = await telegramOnboardingAdapter.configure({
      cfg: {} as MarvConfig,
      runtime: createExitThrowingRuntime(),
      prompter,
      accountOverrides: {},
      shouldPromptAccountIds: false,
      forceAllowFrom: false,
    });

    expect(result.cfg.channels?.telegram?.botToken).toBe("good-token");
    expect(probeTelegramMock).toHaveBeenNthCalledWith(1, "bad-token", 8_000);
    expect(probeTelegramMock).toHaveBeenNthCalledWith(2, "good-token", 8_000);
    expect(note).toHaveBeenCalledWith(
      "Telegram bot token invalid: Unauthorized. Please re-enter it.",
      "Telegram",
    );
    expect(note).toHaveBeenCalledWith("Validated Telegram bot @valid_bot.", "Telegram");
  });

  it("clears shadowing tokenFile overrides when replacing the default account token", async () => {
    probeTelegramMock.mockResolvedValue({
      ok: true,
      bot: { username: "fresh_bot" },
    });

    const prompter = createPrompter({
      texts: ["fresh-token"],
      confirms: [false],
    });

    const result = await telegramOnboardingAdapter.configure({
      cfg: {
        channels: {
          telegram: {
            tokenFile: "/tmp/old-token",
            accounts: {
              default: {
                botToken: "shadow-token",
                tokenFile: "/tmp/shadow-token",
              },
            },
          },
        },
      } as MarvConfig,
      runtime: createExitThrowingRuntime(),
      prompter,
      accountOverrides: {},
      shouldPromptAccountIds: false,
      forceAllowFrom: false,
    });

    expect(result.cfg.channels?.telegram?.botToken).toBe("fresh-token");
    expect(result.cfg.channels?.telegram?.tokenFile).toBeUndefined();
    expect(result.cfg.channels?.telegram?.accounts?.default?.botToken).toBeUndefined();
    expect(result.cfg.channels?.telegram?.accounts?.default?.tokenFile).toBeUndefined();
  });
});
