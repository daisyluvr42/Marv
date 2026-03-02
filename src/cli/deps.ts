import type { sendMessageDiscord } from "../channels/discord/send.js";
import type { sendMessageIMessage } from "../channels/imessage/send.js";
import type { sendMessageSignal } from "../channels/signal/send.js";
import type { sendMessageSlack } from "../channels/slack/send.js";
import type { sendMessageTelegram } from "../channels/telegram/send.js";
import type { sendMessageWhatsApp } from "../channels/web/index.js";
import type { OutboundSendDeps } from "../infra/outbound/deliver.js";

export type CliDeps = {
  sendMessageWhatsApp: typeof sendMessageWhatsApp;
  sendMessageTelegram: typeof sendMessageTelegram;
  sendMessageDiscord: typeof sendMessageDiscord;
  sendMessageSlack: typeof sendMessageSlack;
  sendMessageSignal: typeof sendMessageSignal;
  sendMessageIMessage: typeof sendMessageIMessage;
};

export function createDefaultDeps(): CliDeps {
  return {
    sendMessageWhatsApp: async (...args) => {
      const { sendMessageWhatsApp } = await import("../channels/web/index.js");
      return await sendMessageWhatsApp(...args);
    },
    sendMessageTelegram: async (...args) => {
      const { sendMessageTelegram } = await import("../channels/telegram/send.js");
      return await sendMessageTelegram(...args);
    },
    sendMessageDiscord: async (...args) => {
      const { sendMessageDiscord } = await import("../channels/discord/send.js");
      return await sendMessageDiscord(...args);
    },
    sendMessageSlack: async (...args) => {
      const { sendMessageSlack } = await import("../channels/slack/send.js");
      return await sendMessageSlack(...args);
    },
    sendMessageSignal: async (...args) => {
      const { sendMessageSignal } = await import("../channels/signal/send.js");
      return await sendMessageSignal(...args);
    },
    sendMessageIMessage: async (...args) => {
      const { sendMessageIMessage } = await import("../channels/imessage/send.js");
      return await sendMessageIMessage(...args);
    },
  };
}

// Provider docking: extend this mapping when adding new outbound send deps.
export function createOutboundSendDeps(deps: CliDeps): OutboundSendDeps {
  return {
    sendWhatsApp: deps.sendMessageWhatsApp,
    sendTelegram: deps.sendMessageTelegram,
    sendDiscord: deps.sendMessageDiscord,
    sendSlack: deps.sendMessageSlack,
    sendSignal: deps.sendMessageSignal,
    sendIMessage: deps.sendMessageIMessage,
  };
}

export { logWebSelfId } from "../channels/web/auth-store.js";
