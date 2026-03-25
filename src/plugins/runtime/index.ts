import { createRequire } from "node:module";
import {
  resolveEffectiveMessagesConfig,
  resolveHumanDelayConfig,
} from "../../agents/prompt/identity.js";
import {
  createMemoryGetTool,
  createMemorySearchTool,
  createMemoryWriteTool,
} from "../../agents/tools/memory-tool.js";
import { handleSlackAction } from "../../agents/tools/slack-actions.js";
import {
  hasControlCommand,
  isControlCommandMessage,
  shouldComputeCommandAuthorized,
} from "../../auto-reply/commands/detection.js";
import { shouldHandleTextCommands } from "../../auto-reply/commands/registry.js";
import { createReplyDispatcherWithTyping } from "../../auto-reply/delivery/dispatcher.js";
import { dispatchReplyWithBufferedBlockDispatcher } from "../../auto-reply/delivery/provider-dispatcher.js";
import { dispatchReplyFromConfig } from "../../auto-reply/dispatch-from-config.js";
import { finalizeInboundContext } from "../../auto-reply/inbound/context.js";
import {
  createInboundDebouncer,
  resolveInboundDebounceMs,
} from "../../auto-reply/inbound/debounce.js";
import {
  formatAgentEnvelope,
  formatInboundEnvelope,
  resolveEnvelopeFormatOptions,
} from "../../auto-reply/inbound/envelope.js";
import {
  chunkByNewline,
  chunkMarkdownText,
  chunkMarkdownTextWithMode,
  chunkText,
  chunkTextWithMode,
  resolveChunkMode,
  resolveTextChunkLimit,
} from "../../auto-reply/support/chunk.js";
import {
  buildMentionRegexes,
  matchesMentionPatterns,
  matchesMentionWithExplicit,
} from "../../auto-reply/support/mentions.js";
import { removeAckReactionAfterReply, shouldAckReaction } from "../../channels/ack-reactions.js";
import { resolveCommandAuthorizedFromAuthorizers } from "../../channels/command-gating.js";
import { auditDiscordChannelPermissions } from "../../channels/discord/audit.js";
import {
  listDiscordDirectoryGroupsLive,
  listDiscordDirectoryPeersLive,
} from "../../channels/discord/directory-live.js";
import { monitorDiscordProvider } from "../../channels/discord/monitor.js";
import { probeDiscord } from "../../channels/discord/probe.js";
import { resolveDiscordChannelAllowlist } from "../../channels/discord/resolve-channels.js";
import { resolveDiscordUserAllowlist } from "../../channels/discord/resolve-users.js";
import { sendMessageDiscord, sendPollDiscord } from "../../channels/discord/send.js";
import { monitorIMessageProvider } from "../../channels/imessage/monitor.js";
import { probeIMessage } from "../../channels/imessage/probe.js";
import { sendMessageIMessage } from "../../channels/imessage/send.js";
import {
  listLineAccountIds,
  normalizeAccountId as normalizeLineAccountId,
  resolveDefaultLineAccountId,
  resolveLineAccount,
} from "../../channels/line/accounts.js";
import { monitorLineProvider } from "../../channels/line/monitor.js";
import { probeLineBot } from "../../channels/line/probe.js";
import {
  createQuickReplyItems,
  pushMessageLine,
  pushMessagesLine,
  pushFlexMessage,
  pushTemplateMessage,
  pushLocationMessage,
  pushTextMessageWithQuickReplies,
  sendMessageLine,
} from "../../channels/line/send.js";
import { buildTemplateMessageFromPayload } from "../../channels/line/template-messages.js";
import { discordMessageActions } from "../../channels/plugins/actions/discord.js";
import { signalMessageActions } from "../../channels/plugins/actions/signal.js";
import { telegramMessageActions } from "../../channels/plugins/actions/telegram.js";
import { createWhatsAppLoginTool } from "../../channels/plugins/agent-tools/whatsapp-login.js";
import { recordInboundSession } from "../../channels/session.js";
import { monitorSignalProvider } from "../../channels/signal/index.js";
import { probeSignal } from "../../channels/signal/probe.js";
import { sendMessageSignal } from "../../channels/signal/send.js";
import {
  listSlackDirectoryGroupsLive,
  listSlackDirectoryPeersLive,
} from "../../channels/slack/directory-live.js";
import { monitorSlackProvider } from "../../channels/slack/index.js";
import { probeSlack } from "../../channels/slack/probe.js";
import { resolveSlackChannelAllowlist } from "../../channels/slack/resolve-channels.js";
import { resolveSlackUserAllowlist } from "../../channels/slack/resolve-users.js";
import { sendMessageSlack } from "../../channels/slack/send.js";
import {
  auditTelegramGroupMembership,
  collectTelegramUnmentionedGroupIds,
} from "../../channels/telegram/audit.js";
import { monitorTelegramProvider } from "../../channels/telegram/monitor.js";
import { probeTelegram } from "../../channels/telegram/probe.js";
import { sendMessageTelegram, sendPollTelegram } from "../../channels/telegram/send.js";
import { resolveTelegramToken } from "../../channels/telegram/token.js";
import { getActiveWebListener } from "../../channels/web/active-listener.js";
import {
  getWebAuthAgeMs,
  logoutWeb,
  logWebSelfId,
  readWebSelfId,
  webAuthExists,
} from "../../channels/web/auth-store.js";
import { loadWebMedia } from "../../channels/web/media.js";
import { registerMemoryCli } from "../../cli/memory-cli.js";
import { loadConfig, writeConfigFile } from "../../core/config/config.js";
import {
  resolveChannelGroupPolicy,
  resolveChannelGroupRequireMention,
} from "../../core/config/group-policy.js";
import { resolveMarkdownTableMode } from "../../core/config/markdown-tables.js";
import { resolveStateDir } from "../../core/config/paths.js";
import {
  readSessionUpdatedAt,
  recordSessionMetaFromInbound,
  resolveStorePath,
  updateLastRoute,
} from "../../core/config/sessions.js";
import { shouldLogVerbose } from "../../globals.js";
import { getChannelActivity, recordChannelActivity } from "../../infra/channel-activity.js";
import { enqueueSystemEvent } from "../../infra/system-events.js";
import { getChildLogger } from "../../logging.js";
import { normalizeLogLevel } from "../../logging/levels.js";
import { convertMarkdownTables } from "../../markdown/tables.js";
import { isVoiceCompatibleAudio } from "../../media/audio.js";
import { mediaKindFromMime } from "../../media/constants.js";
import { fetchRemoteMedia } from "../../media/fetch.js";
import { getImageMetadata, resizeToJpeg } from "../../media/image-ops.js";
import { detectMime } from "../../media/mime.js";
import { saveMediaBuffer } from "../../media/store.js";
import { buildPairingReply } from "../../pairing/pairing-messages.js";
import {
  readChannelAllowFromStore,
  upsertChannelPairingRequest,
} from "../../pairing/pairing-store.js";
import { runCommandWithTimeout } from "../../process/exec.js";
import { resolveAgentRoute } from "../../routing/resolve-route.js";
import { textToSpeechTelephony } from "../../tts/tts.js";
import { formatNativeDependencyHint } from "./native-deps.js";
import type { PluginRuntime } from "./types.js";

let cachedVersion: string | null = null;

function resolveVersion(): string {
  if (cachedVersion) {
    return cachedVersion;
  }
  try {
    const require = createRequire(import.meta.url);
    const pkg = require("../../../package.json") as { version?: string };
    cachedVersion = pkg.version ?? "unknown";
    return cachedVersion;
  } catch {
    cachedVersion = "unknown";
    return cachedVersion;
  }
}

const sendMessageWhatsAppLazy: PluginRuntime["channel"]["whatsapp"]["sendMessageWhatsApp"] = async (
  ...args
) => {
  const { sendMessageWhatsApp } = await loadWebOutbound();
  return sendMessageWhatsApp(...args);
};

const sendPollWhatsAppLazy: PluginRuntime["channel"]["whatsapp"]["sendPollWhatsApp"] = async (
  ...args
) => {
  const { sendPollWhatsApp } = await loadWebOutbound();
  return sendPollWhatsApp(...args);
};

const loginWebLazy: PluginRuntime["channel"]["whatsapp"]["loginWeb"] = async (...args) => {
  const { loginWeb } = await loadWebLogin();
  return loginWeb(...args);
};

const startWebLoginWithQrLazy: PluginRuntime["channel"]["whatsapp"]["startWebLoginWithQr"] = async (
  ...args
) => {
  const { startWebLoginWithQr } = await loadWebLoginQr();
  return startWebLoginWithQr(...args);
};

const waitForWebLoginLazy: PluginRuntime["channel"]["whatsapp"]["waitForWebLogin"] = async (
  ...args
) => {
  const { waitForWebLogin } = await loadWebLoginQr();
  return waitForWebLogin(...args);
};

const monitorWebChannelLazy: PluginRuntime["channel"]["whatsapp"]["monitorWebChannel"] = async (
  ...args
) => {
  const { monitorWebChannel } = await loadWebChannel();
  return monitorWebChannel(...args);
};

const handleWhatsAppActionLazy: PluginRuntime["channel"]["whatsapp"]["handleWhatsAppAction"] =
  async (...args) => {
    const { handleWhatsAppAction } = await loadWhatsAppActions();
    return handleWhatsAppAction(...args);
  };

let webOutboundPromise: Promise<typeof import("../../channels/web/outbound.js")> | null = null;
let webLoginPromise: Promise<typeof import("../../channels/web/login.js")> | null = null;
let webLoginQrPromise: Promise<typeof import("../../channels/web/login-qr.js")> | null = null;
let webChannelPromise: Promise<typeof import("../../channels/web/index.js")> | null = null;
let whatsappActionsPromise: Promise<
  typeof import("../../agents/tools/whatsapp-actions.js")
> | null = null;

function loadWebOutbound() {
  webOutboundPromise ??= import("../../channels/web/outbound.js");
  return webOutboundPromise;
}

function loadWebLogin() {
  webLoginPromise ??= import("../../channels/web/login.js");
  return webLoginPromise;
}

function loadWebLoginQr() {
  webLoginQrPromise ??= import("../../channels/web/login-qr.js");
  return webLoginQrPromise;
}

function loadWebChannel() {
  webChannelPromise ??= import("../../channels/web/index.js");
  return webChannelPromise;
}

function loadWhatsAppActions() {
  whatsappActionsPromise ??= import("../../agents/tools/whatsapp-actions.js");
  return whatsappActionsPromise;
}

export function createPluginRuntime(): PluginRuntime {
  return {
    version: resolveVersion(),
    config: {
      loadConfig,
      writeConfigFile,
    },
    system: {
      enqueueSystemEvent,
      runCommandWithTimeout,
      formatNativeDependencyHint,
    },
    media: {
      loadWebMedia,
      detectMime,
      mediaKindFromMime,
      isVoiceCompatibleAudio,
      getImageMetadata,
      resizeToJpeg,
    },
    tts: {
      textToSpeechTelephony,
    },
    tools: {
      createMemoryGetTool,
      createMemorySearchTool,
      createMemoryWriteTool,
      registerMemoryCli,
    },
    channel: {
      text: {
        chunkByNewline,
        chunkMarkdownText,
        chunkMarkdownTextWithMode,
        chunkText,
        chunkTextWithMode,
        resolveChunkMode,
        resolveTextChunkLimit,
        hasControlCommand,
        resolveMarkdownTableMode,
        convertMarkdownTables,
      },
      reply: {
        dispatchReplyWithBufferedBlockDispatcher,
        createReplyDispatcherWithTyping,
        resolveEffectiveMessagesConfig,
        resolveHumanDelayConfig,
        dispatchReplyFromConfig,
        finalizeInboundContext,
        formatAgentEnvelope,
        /** @deprecated Prefer `BodyForAgent` + structured user-context blocks (do not build plaintext envelopes for prompts). */
        formatInboundEnvelope,
        resolveEnvelopeFormatOptions,
      },
      routing: {
        resolveAgentRoute,
      },
      pairing: {
        buildPairingReply,
        readAllowFromStore: readChannelAllowFromStore,
        upsertPairingRequest: upsertChannelPairingRequest,
      },
      media: {
        fetchRemoteMedia,
        saveMediaBuffer,
      },
      activity: {
        record: recordChannelActivity,
        get: getChannelActivity,
      },
      session: {
        resolveStorePath,
        readSessionUpdatedAt,
        recordSessionMetaFromInbound,
        recordInboundSession,
        updateLastRoute,
      },
      mentions: {
        buildMentionRegexes,
        matchesMentionPatterns,
        matchesMentionWithExplicit,
      },
      reactions: {
        shouldAckReaction,
        removeAckReactionAfterReply,
      },
      groups: {
        resolveGroupPolicy: resolveChannelGroupPolicy,
        resolveRequireMention: resolveChannelGroupRequireMention,
      },
      debounce: {
        createInboundDebouncer,
        resolveInboundDebounceMs,
      },
      commands: {
        resolveCommandAuthorizedFromAuthorizers,
        isControlCommandMessage,
        shouldComputeCommandAuthorized,
        shouldHandleTextCommands,
      },
      discord: {
        messageActions: discordMessageActions,
        auditChannelPermissions: auditDiscordChannelPermissions,
        listDirectoryGroupsLive: listDiscordDirectoryGroupsLive,
        listDirectoryPeersLive: listDiscordDirectoryPeersLive,
        probeDiscord,
        resolveChannelAllowlist: resolveDiscordChannelAllowlist,
        resolveUserAllowlist: resolveDiscordUserAllowlist,
        sendMessageDiscord,
        sendPollDiscord,
        monitorDiscordProvider,
      },
      slack: {
        listDirectoryGroupsLive: listSlackDirectoryGroupsLive,
        listDirectoryPeersLive: listSlackDirectoryPeersLive,
        probeSlack,
        resolveChannelAllowlist: resolveSlackChannelAllowlist,
        resolveUserAllowlist: resolveSlackUserAllowlist,
        sendMessageSlack,
        monitorSlackProvider,
        handleSlackAction,
      },
      telegram: {
        auditGroupMembership: auditTelegramGroupMembership,
        collectUnmentionedGroupIds: collectTelegramUnmentionedGroupIds,
        probeTelegram,
        resolveTelegramToken,
        sendMessageTelegram,
        sendPollTelegram,
        monitorTelegramProvider,
        messageActions: telegramMessageActions,
      },
      signal: {
        probeSignal,
        sendMessageSignal,
        monitorSignalProvider,
        messageActions: signalMessageActions,
      },
      imessage: {
        monitorIMessageProvider,
        probeIMessage,
        sendMessageIMessage,
      },
      whatsapp: {
        getActiveWebListener,
        getWebAuthAgeMs,
        logoutWeb,
        logWebSelfId,
        readWebSelfId,
        webAuthExists,
        sendMessageWhatsApp: sendMessageWhatsAppLazy,
        sendPollWhatsApp: sendPollWhatsAppLazy,
        loginWeb: loginWebLazy,
        startWebLoginWithQr: startWebLoginWithQrLazy,
        waitForWebLogin: waitForWebLoginLazy,
        monitorWebChannel: monitorWebChannelLazy,
        handleWhatsAppAction: handleWhatsAppActionLazy,
        createLoginTool: createWhatsAppLoginTool,
      },
      line: {
        listLineAccountIds,
        resolveDefaultLineAccountId,
        resolveLineAccount,
        normalizeAccountId: normalizeLineAccountId,
        probeLineBot,
        sendMessageLine,
        pushMessageLine,
        pushMessagesLine,
        pushFlexMessage,
        pushTemplateMessage,
        pushLocationMessage,
        pushTextMessageWithQuickReplies,
        createQuickReplyItems,
        buildTemplateMessageFromPayload,
        monitorLineProvider,
      },
    },
    logging: {
      shouldLogVerbose,
      getChildLogger: (bindings, opts) => {
        const logger = getChildLogger(bindings, {
          level: opts?.level ? normalizeLogLevel(opts.level) : undefined,
        });
        return {
          debug: (message) => logger.debug?.(message),
          info: (message) => logger.info(message),
          warn: (message) => logger.warn(message),
          error: (message) => logger.error(message),
        };
      },
    },
    state: {
      resolveStateDir,
    },
  };
}

export type { PluginRuntime } from "./types.js";
