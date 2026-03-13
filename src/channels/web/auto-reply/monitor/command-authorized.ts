import type { loadConfig } from "../../../../core/config/config.js";
import { readChannelAllowFromStore } from "../../../../pairing/pairing-store.js";
import { normalizeE164 } from "../../../../utils.js";
import { resolveCommandAuthorizedFromAuthorizers } from "../../../command-gating.js";
import type { WebInboundMsg } from "../types.js";

export function normalizeAllowFromE164(values: Array<string | number> | undefined): string[] {
  const list = Array.isArray(values) ? values : [];
  return list
    .map((entry) => String(entry).trim())
    .filter((entry) => entry && entry !== "*")
    .map((entry) => normalizeE164(entry))
    .filter((entry): entry is string => Boolean(entry));
}

export async function resolveWhatsAppCommandAuthorized(params: {
  cfg: ReturnType<typeof loadConfig>;
  msg: WebInboundMsg;
}): Promise<boolean> {
  const useAccessGroups = params.cfg.commands?.useAccessGroups !== false;
  if (!useAccessGroups) {
    return true;
  }

  const isGroup = params.msg.chatType === "group";
  const senderE164 = normalizeE164(
    isGroup ? (params.msg.senderE164 ?? "") : (params.msg.senderE164 ?? params.msg.from ?? ""),
  );
  if (!senderE164) {
    return false;
  }

  const configuredAllowFrom = params.cfg.channels?.whatsapp?.allowFrom ?? [];
  const configuredGroupAllowFrom =
    params.cfg.channels?.whatsapp?.groupAllowFrom ??
    (configuredAllowFrom.length > 0 ? configuredAllowFrom : undefined);

  if (isGroup) {
    const configured = Boolean(configuredGroupAllowFrom && configuredGroupAllowFrom.length > 0);
    const allowed =
      configured &&
      (configuredGroupAllowFrom?.some((v) => String(v).trim() === "*") ||
        normalizeAllowFromE164(configuredGroupAllowFrom).includes(senderE164));
    return resolveCommandAuthorizedFromAuthorizers({
      useAccessGroups,
      authorizers: [{ configured, allowed }],
    });
  }

  const storeAllowFrom = await readChannelAllowFromStore(
    "whatsapp",
    process.env,
    params.msg.accountId,
  ).catch(() => []);
  const combinedAllowFrom = Array.from(
    new Set([...(configuredAllowFrom ?? []), ...storeAllowFrom]),
  );
  const allowFrom =
    combinedAllowFrom.length > 0
      ? combinedAllowFrom
      : params.msg.selfE164
        ? [params.msg.selfE164]
        : [];
  const configured = allowFrom.length > 0;
  const allowed =
    configured &&
    (allowFrom.some((v) => String(v).trim() === "*") ||
      normalizeAllowFromE164(allowFrom).includes(senderE164));
  return resolveCommandAuthorizedFromAuthorizers({
    useAccessGroups,
    authorizers: [{ configured, allowed }],
  });
}
