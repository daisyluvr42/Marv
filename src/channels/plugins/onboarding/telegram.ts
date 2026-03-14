import { formatCliCommand } from "../../../cli/command-format.js";
import type { MarvConfig } from "../../../core/config/config.js";
import type { DmPolicy } from "../../../core/config/types.js";
import { DEFAULT_ACCOUNT_ID, normalizeAccountId } from "../../../routing/session-key.js";
import { formatDocsLink } from "../../../terminal/links.js";
import type { WizardPrompter } from "../../../wizard/prompts.js";
import {
  listTelegramAccountIds,
  resolveDefaultTelegramAccountId,
  resolveTelegramAccount,
} from "../../telegram/accounts.js";
import { fetchTelegramChatId } from "../../telegram/api.js";
import { probeTelegram } from "../../telegram/probe.js";
import type { ChannelOnboardingAdapter, ChannelOnboardingDmPolicy } from "../onboarding-types.js";
import { addWildcardAllowFrom, mergeAllowFromEntries, promptAccountId } from "./helpers.js";

const channel = "telegram" as const;
const TELEGRAM_ONBOARDING_PROBE_TIMEOUT_MS = 8_000;

function setTelegramDmPolicy(cfg: MarvConfig, dmPolicy: DmPolicy) {
  const allowFrom =
    dmPolicy === "open" ? addWildcardAllowFrom(cfg.channels?.telegram?.allowFrom) : undefined;
  return {
    ...cfg,
    channels: {
      ...cfg.channels,
      telegram: {
        ...cfg.channels?.telegram,
        dmPolicy,
        ...(allowFrom ? { allowFrom } : {}),
      },
    },
  };
}

async function noteTelegramTokenHelp(prompter: WizardPrompter): Promise<void> {
  await prompter.note(
    [
      "1) Open Telegram and chat with @BotFather",
      "2) Run /newbot (or /mybots)",
      "3) Copy the token (looks like 123456:ABC...)",
      "Tip: you can also set TELEGRAM_BOT_TOKEN in your env.",
      `Docs: ${formatDocsLink("/telegram")}`,
      "Project: https://github.com/daisyluvr42/Marv",
    ].join("\n"),
    "Telegram bot token",
  );
}

async function noteTelegramUserIdHelp(prompter: WizardPrompter): Promise<void> {
  await prompter.note(
    [
      `1) DM your bot, then read from.id in \`${formatCliCommand("marv logs --follow")}\` (safest)`,
      "2) Or call https://api.telegram.org/bot<bot_token>/getUpdates and read message.from.id",
      "3) Third-party: DM @userinfobot or @getidsbot",
      `Docs: ${formatDocsLink("/telegram")}`,
      "Project: https://github.com/daisyluvr42/Marv",
    ].join("\n"),
    "Telegram user id",
  );
}

function formatTelegramProbeError(params: {
  error?: string | null;
  status?: number | null;
}): string {
  if (params.error?.trim()) {
    return params.error.trim();
  }
  if (typeof params.status === "number") {
    return `HTTP ${params.status}`;
  }
  return "unknown Telegram API error";
}

async function validateTelegramBotToken(params: {
  token: string;
  prompter: WizardPrompter;
  sourceLabel: string;
}): Promise<{ ok: true; username?: string | null } | { ok: false; error: string }> {
  const progress = params.prompter.progress("Telegram");
  try {
    progress.update(`Validating ${params.sourceLabel}…`);
    const probe = await probeTelegram(params.token, TELEGRAM_ONBOARDING_PROBE_TIMEOUT_MS);
    if (!probe.ok) {
      return {
        ok: false,
        error: formatTelegramProbeError({
          error: probe.error,
          status: probe.status,
        }),
      };
    }
    return { ok: true, username: probe.bot?.username ?? null };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    progress.stop();
  }
}

async function promptValidatedTelegramBotToken(params: {
  prompter: WizardPrompter;
  initialValue?: string;
  invalidReason?: string;
}): Promise<string> {
  let initialValue = params.initialValue;
  let invalidReason = params.invalidReason;
  while (true) {
    if (invalidReason) {
      await params.prompter.note(
        `Telegram bot token invalid: ${invalidReason}. Please re-enter it.`,
        "Telegram",
      );
    }
    const token = String(
      await params.prompter.text({
        message: "Enter Telegram bot token",
        initialValue,
        validate: (value) => (value?.trim() ? undefined : "Required"),
      }),
    ).trim();
    const validation = await validateTelegramBotToken({
      token,
      prompter: params.prompter,
      sourceLabel: "Telegram bot token",
    });
    if (validation.ok) {
      if (validation.username) {
        await params.prompter.note(`Validated Telegram bot @${validation.username}.`, "Telegram");
      } else {
        await params.prompter.note("Validated Telegram bot token.", "Telegram");
      }
      return token;
    }
    initialValue = token;
    invalidReason = validation.error;
  }
}

function clearDefaultAccountTokenOverrides(cfg: MarvConfig) {
  const defaultAccount = cfg.channels?.telegram?.accounts?.[DEFAULT_ACCOUNT_ID];
  if (!defaultAccount) {
    return cfg.channels?.telegram?.accounts;
  }
  return {
    ...cfg.channels?.telegram?.accounts,
    [DEFAULT_ACCOUNT_ID]: {
      ...defaultAccount,
      botToken: undefined,
      tokenFile: undefined,
    },
  };
}

function setTelegramManualToken(params: {
  cfg: MarvConfig;
  accountId: string;
  token: string;
}): MarvConfig {
  if (params.accountId === DEFAULT_ACCOUNT_ID) {
    return {
      ...params.cfg,
      channels: {
        ...params.cfg.channels,
        telegram: {
          ...params.cfg.channels?.telegram,
          enabled: true,
          botToken: params.token,
          tokenFile: undefined,
          accounts: clearDefaultAccountTokenOverrides(params.cfg),
        },
      },
    };
  }

  return {
    ...params.cfg,
    channels: {
      ...params.cfg.channels,
      telegram: {
        ...params.cfg.channels?.telegram,
        enabled: true,
        accounts: {
          ...params.cfg.channels?.telegram?.accounts,
          [params.accountId]: {
            ...params.cfg.channels?.telegram?.accounts?.[params.accountId],
            enabled: params.cfg.channels?.telegram?.accounts?.[params.accountId]?.enabled ?? true,
            botToken: params.token,
            tokenFile: undefined,
          },
        },
      },
    },
  };
}

function setTelegramEnvTokenSource(cfg: MarvConfig): MarvConfig {
  return {
    ...cfg,
    channels: {
      ...cfg.channels,
      telegram: {
        ...cfg.channels?.telegram,
        enabled: true,
        botToken: undefined,
        tokenFile: undefined,
        accounts: clearDefaultAccountTokenOverrides(cfg),
      },
    },
  };
}

async function promptTelegramAllowFrom(params: {
  cfg: MarvConfig;
  prompter: WizardPrompter;
  accountId: string;
}): Promise<MarvConfig> {
  const { cfg, prompter, accountId } = params;
  const resolved = resolveTelegramAccount({ cfg, accountId });
  const existingAllowFrom = resolved.config.allowFrom ?? [];
  await noteTelegramUserIdHelp(prompter);

  const token = resolved.token;
  if (!token) {
    await prompter.note("Telegram token missing; username lookup is unavailable.", "Telegram");
  }

  const resolveTelegramUserId = async (raw: string): Promise<string | null> => {
    const trimmed = raw.trim();
    if (!trimmed) {
      return null;
    }
    const stripped = trimmed.replace(/^(telegram|tg):/i, "").trim();
    if (/^\d+$/.test(stripped)) {
      return stripped;
    }
    if (!token) {
      return null;
    }
    const username = stripped.startsWith("@") ? stripped : `@${stripped}`;
    return await fetchTelegramChatId({ token, chatId: username });
  };

  const parseInput = (value: string) =>
    value
      .split(/[\n,;]+/g)
      .map((entry) => entry.trim())
      .filter(Boolean);

  let resolvedIds: string[] = [];
  while (resolvedIds.length === 0) {
    const entry = await prompter.text({
      message: "Telegram allowFrom (numeric sender id; @username resolves to id)",
      placeholder: "@username",
      initialValue: existingAllowFrom[0] ? String(existingAllowFrom[0]) : undefined,
      validate: (value) => (String(value ?? "").trim() ? undefined : "Required"),
    });
    const parts = parseInput(String(entry));
    const results = await Promise.all(parts.map((part) => resolveTelegramUserId(part)));
    const unresolved = parts.filter((_, idx) => !results[idx]);
    if (unresolved.length > 0) {
      await prompter.note(
        `Could not resolve: ${unresolved.join(", ")}. Use @username or numeric id.`,
        "Telegram allowlist",
      );
      continue;
    }
    resolvedIds = results.filter(Boolean) as string[];
  }

  const unique = mergeAllowFromEntries(existingAllowFrom, resolvedIds);

  if (accountId === DEFAULT_ACCOUNT_ID) {
    return {
      ...cfg,
      channels: {
        ...cfg.channels,
        telegram: {
          ...cfg.channels?.telegram,
          enabled: true,
          dmPolicy: "allowlist",
          allowFrom: unique,
        },
      },
    };
  }

  return {
    ...cfg,
    channels: {
      ...cfg.channels,
      telegram: {
        ...cfg.channels?.telegram,
        enabled: true,
        accounts: {
          ...cfg.channels?.telegram?.accounts,
          [accountId]: {
            ...cfg.channels?.telegram?.accounts?.[accountId],
            enabled: cfg.channels?.telegram?.accounts?.[accountId]?.enabled ?? true,
            dmPolicy: "allowlist",
            allowFrom: unique,
          },
        },
      },
    },
  };
}

async function promptTelegramAllowFromForAccount(params: {
  cfg: MarvConfig;
  prompter: WizardPrompter;
  accountId?: string;
}): Promise<MarvConfig> {
  const accountId =
    params.accountId && normalizeAccountId(params.accountId)
      ? (normalizeAccountId(params.accountId) ?? DEFAULT_ACCOUNT_ID)
      : resolveDefaultTelegramAccountId(params.cfg);
  return promptTelegramAllowFrom({
    cfg: params.cfg,
    prompter: params.prompter,
    accountId,
  });
}

const dmPolicy: ChannelOnboardingDmPolicy = {
  label: "Telegram",
  channel,
  policyKey: "channels.telegram.dmPolicy",
  allowFromKey: "channels.telegram.allowFrom",
  getCurrent: (cfg) => cfg.channels?.telegram?.dmPolicy ?? "pairing",
  setPolicy: (cfg, policy) => setTelegramDmPolicy(cfg, policy),
  promptAllowFrom: promptTelegramAllowFromForAccount,
};

export const telegramOnboardingAdapter: ChannelOnboardingAdapter = {
  channel,
  getStatus: async ({ cfg }) => {
    const configured = listTelegramAccountIds(cfg).some((accountId) =>
      Boolean(resolveTelegramAccount({ cfg, accountId }).token),
    );
    return {
      channel,
      configured,
      statusLines: [`Telegram: ${configured ? "configured" : "needs token"}`],
      selectionHint: configured ? "recommended · configured" : "recommended · newcomer-friendly",
      quickstartScore: configured ? 1 : 10,
    };
  },
  configure: async ({
    cfg,
    prompter,
    accountOverrides,
    shouldPromptAccountIds,
    forceAllowFrom,
  }) => {
    const telegramOverride = accountOverrides.telegram?.trim();
    const defaultTelegramAccountId = resolveDefaultTelegramAccountId(cfg);
    let telegramAccountId = telegramOverride
      ? normalizeAccountId(telegramOverride)
      : defaultTelegramAccountId;
    if (shouldPromptAccountIds && !telegramOverride) {
      telegramAccountId = await promptAccountId({
        cfg,
        prompter,
        label: "Telegram",
        currentId: telegramAccountId,
        listAccountIds: listTelegramAccountIds,
        defaultAccountId: defaultTelegramAccountId,
      });
    }

    let next = cfg;
    const resolvedAccount = resolveTelegramAccount({
      cfg: next,
      accountId: telegramAccountId,
    });
    const accountConfigured = Boolean(resolvedAccount.token);
    const allowEnv = telegramAccountId === DEFAULT_ACCOUNT_ID;
    const hasConfigToken = Boolean(
      resolvedAccount.config.botToken || resolvedAccount.config.tokenFile,
    );
    const envToken = allowEnv ? process.env.TELEGRAM_BOT_TOKEN?.trim() : "";
    const canUseEnv = Boolean(envToken) && !hasConfigToken;

    let token: string | null = null;
    if (!accountConfigured) {
      await noteTelegramTokenHelp(prompter);
    }
    if (canUseEnv) {
      const keepEnv = await prompter.confirm({
        message: "TELEGRAM_BOT_TOKEN detected. Use env var?",
        initialValue: true,
      });
      if (keepEnv) {
        const validation = await validateTelegramBotToken({
          token: envToken ?? "",
          prompter,
          sourceLabel: "TELEGRAM_BOT_TOKEN",
        });
        if (validation.ok) {
          next = setTelegramEnvTokenSource(next);
          if (validation.username) {
            await prompter.note(
              `Validated TELEGRAM_BOT_TOKEN for @${validation.username}.`,
              "Telegram",
            );
          } else {
            await prompter.note("Validated TELEGRAM_BOT_TOKEN.", "Telegram");
          }
        } else {
          token = await promptValidatedTelegramBotToken({
            prompter,
            invalidReason: `TELEGRAM_BOT_TOKEN failed validation: ${validation.error}`,
          });
        }
      } else {
        token = await promptValidatedTelegramBotToken({ prompter });
      }
    } else if (hasConfigToken) {
      const keep = await prompter.confirm({
        message: "Telegram token already configured. Keep it?",
        initialValue: true,
      });
      if (keep) {
        const sourceLabel =
          resolvedAccount.tokenSource === "tokenFile"
            ? "configured Telegram token file"
            : "configured Telegram token";
        const validation = await validateTelegramBotToken({
          token: resolvedAccount.token,
          prompter,
          sourceLabel,
        });
        if (!validation.ok) {
          token = await promptValidatedTelegramBotToken({
            prompter,
            invalidReason: `${sourceLabel} failed validation: ${validation.error}`,
          });
        }
      } else {
        token = await promptValidatedTelegramBotToken({ prompter });
      }
    } else {
      token = await promptValidatedTelegramBotToken({ prompter });
    }

    if (token) {
      next = setTelegramManualToken({
        cfg: next,
        accountId: telegramAccountId,
        token,
      });
    }

    if (!forceAllowFrom) {
      await prompter.note(
        [
          "Telegram token is valid.",
          "Direct chats still default to pairing until you open or allowlist DM access.",
          `Check later: ${formatCliCommand("marv channels status --probe")}`,
        ].join("\n"),
        "Telegram",
      );
    }

    if (forceAllowFrom) {
      next = await promptTelegramAllowFrom({
        cfg: next,
        prompter,
        accountId: telegramAccountId,
      });
    }

    return { cfg: next, accountId: telegramAccountId };
  },
  dmPolicy,
  disable: (cfg) => ({
    ...cfg,
    channels: {
      ...cfg.channels,
      telegram: { ...cfg.channels?.telegram, enabled: false },
    },
  }),
};
