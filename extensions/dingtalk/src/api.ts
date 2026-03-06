import { parseDingTalkTarget } from "./targets.js";
import type {
  ResolvedDingTalkAccount,
  DingTalkProbeResult,
  DingTalkReplyContext,
} from "./types.js";

const DINGTALK_API = "https://api.dingtalk.com";
const tokenCache = new Map<string, { token: string; expiresAt: number }>();

async function postJson<T>(
  url: string,
  init: { body: unknown; headers?: Record<string, string> },
): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(init.headers ?? {}),
    },
    body: JSON.stringify(init.body),
  });
  const text = await res.text();
  const parsed = text ? (JSON.parse(text) as T) : ({} as T);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return parsed;
}

export async function getDingTalkAccessToken(account: ResolvedDingTalkAccount): Promise<string> {
  const clientId = account.clientId?.trim() ?? "";
  const clientSecret = account.clientSecret?.trim() ?? "";
  if (!clientId || !clientSecret) {
    throw new Error("DingTalk clientId and clientSecret are required");
  }
  const cacheKey = `${clientId}:${clientSecret}`;
  const cached = tokenCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now() + 60_000) {
    return cached.token;
  }
  const payload = await postJson<{ accessToken?: string; expireIn?: number }>(
    `${DINGTALK_API}/v1.0/oauth2/accessToken`,
    {
      body: {
        appKey: clientId,
        appSecret: clientSecret,
      },
    },
  );
  const token = payload.accessToken?.trim() ?? "";
  if (!token) {
    throw new Error("DingTalk access token missing from response");
  }
  tokenCache.set(cacheKey, {
    token,
    expiresAt: Date.now() + (payload.expireIn ?? 7200) * 1000,
  });
  return token;
}

export async function probeDingTalk(
  account: ResolvedDingTalkAccount,
): Promise<DingTalkProbeResult> {
  if (!account.clientId || !account.clientSecret) {
    return {
      ok: false,
      accountId: account.accountId,
      clientId: account.clientId,
      robotCode: account.robotCode,
      stage: "credentials",
      error: "missing credentials (clientId, clientSecret)",
    };
  }
  try {
    await getDingTalkAccessToken(account);
    return {
      ok: true,
      accountId: account.accountId,
      clientId: account.clientId,
      robotCode: account.robotCode,
      stage: "token",
    };
  } catch (error) {
    return {
      ok: false,
      accountId: account.accountId,
      clientId: account.clientId,
      robotCode: account.robotCode,
      stage: "token",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function buildTextBody(text: string) {
  return {
    msgtype: "text",
    text: {
      content: text,
    },
  };
}

export async function sendDingTalkReply(params: {
  account: ResolvedDingTalkAccount;
  context: DingTalkReplyContext;
  text: string;
}) {
  const token = await getDingTalkAccessToken(params.account);
  await postJson(params.context.sessionWebhook, {
    headers: {
      "x-acs-dingtalk-access-token": token,
    },
    body: buildTextBody(params.text),
  });
}

export async function sendDingTalkText(params: {
  account: ResolvedDingTalkAccount;
  target: string;
  text: string;
}) {
  const account = params.account;
  const target = parseDingTalkTarget(params.target);
  if (!target) {
    throw new Error('invalid DingTalk target (use "user:<id>" or "group:<conversationId>")');
  }
  if (!account.robotCode) {
    throw new Error("DingTalk robotCode is required for outbound sends");
  }
  const token = await getDingTalkAccessToken(account);
  if (target.kind === "user") {
    await postJson(`${DINGTALK_API}/v1.0/robot/oToMessages/batchSend`, {
      headers: {
        "x-acs-dingtalk-access-token": token,
      },
      body: {
        robotCode: account.robotCode,
        userIds: [target.value],
        msgKey: "sampleText",
        msgParam: JSON.stringify({ content: params.text }),
      },
    });
    return;
  }
  await postJson(`${DINGTALK_API}/v1.0/robot/groupMessages/send`, {
    headers: {
      "x-acs-dingtalk-access-token": token,
    },
    body: {
      robotCode: account.robotCode,
      openConversationId: target.value,
      msgKey: "sampleText",
      msgParam: JSON.stringify({ content: params.text }),
    },
  });
}
