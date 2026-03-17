import type { MarvConfig, RuntimeEnv } from "agentmarv/plugin-sdk";
import type { ResolvedWeChatAccount } from "./types.js";

export type WeChatMonitorParams = {
  cfg: MarvConfig;
  account: ResolvedWeChatAccount;
  runtime: RuntimeEnv;
  abortSignal?: AbortSignal;
  statusSink: (patch: Record<string, unknown>) => void;
};

/**
 * Start monitoring a WeChat account via Wechaty.
 *
 * This initializes the Wechaty bot, listens for incoming messages,
 * and dispatches them to the Marv reply pipeline.
 */
export async function monitorWeChatProvider(params: WeChatMonitorParams): Promise<void> {
  const { account, statusSink, abortSignal } = params;

  if (!account.puppet) {
    statusSink({ lastError: "No puppet configured for WeChat account" });
    return;
  }

  // Dynamic import — wechaty is an optional peer dependency.
  const { WechatyBuilder } = await import("wechaty" as string);

  const bot = WechatyBuilder.build({
    name: `marv-wechat-${account.accountId}`,
    puppet: account.puppet,
    puppetOptions: account.puppetToken ? { token: account.puppetToken } : undefined,
  });

  bot.on("scan", (qrcode: string, status: number) => {
    statusSink({ qrcode, scanStatus: status });
  });

  bot.on("login", (user: { name: () => string }) => {
    statusSink({
      running: true,
      lastStartAt: new Date().toISOString(),
      userName: user.name(),
      lastError: null,
    });
  });

  bot.on("logout", () => {
    statusSink({
      running: false,
      lastStopAt: new Date().toISOString(),
    });
  });

  bot.on(
    "message",
    async (message: {
      self: () => boolean;
      text: () => string;
      talker: () => { id: string; name: () => string };
      room: () => { id: string; topic: () => Promise<string> } | null;
      type: () => number;
    }) => {
      // Skip self messages.
      if (message.self()) return;

      // Only handle text messages (type 7 in Wechaty).
      if (message.type() !== 7) return;

      const talker = message.talker();
      const room = message.room();
      const isGroup = room !== null;

      statusSink({ lastInboundAt: new Date().toISOString() });

      // Build message context for the Marv reply pipeline.
      const _msgContext = {
        Body: message.text(),
        From: isGroup ? `wechat:group:${room!.id}` : `wechat:${talker.id}`,
        To: `wechat:bot`,
        SenderId: talker.id,
        SenderName: talker.name(),
        ChatType: isGroup ? "group" : "direct",
        GroupSubject: isGroup ? await room!.topic() : undefined,
        Provider: "wechat",
        WasMentioned: false,
      };

      // TODO: Dispatch to Marv reply pipeline via runtime.dispatchInboundMessage().
      // This requires the runtime dispatch adapter to be wired up.
      void _msgContext;
    },
  );

  bot.on("error", (error: Error) => {
    statusSink({ lastError: error.message });
  });

  await bot.start();

  // Handle abort signal for graceful shutdown.
  if (abortSignal) {
    abortSignal.addEventListener("abort", () => {
      bot.stop().catch(() => {});
    });
  }
}
