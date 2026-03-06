import type { DingTalkTarget } from "./types.js";

export function normalizeDingTalkTarget(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = trimmed.replace(/^dingtalk:/i, "");
  if (/^(user|group|conversation):[^:\s]+$/i.test(normalized)) {
    const [kind, value] = normalized.split(":", 2);
    return `${kind.toLowerCase()}:${value.trim()}`;
  }
  return null;
}

export function parseDingTalkTarget(raw: string): DingTalkTarget | null {
  const normalized = normalizeDingTalkTarget(raw);
  if (!normalized) {
    return null;
  }
  const [kind, value] = normalized.split(":", 2);
  if (kind === "user") {
    return { kind: "user", value };
  }
  if (kind === "group" || kind === "conversation") {
    return { kind: "group", value };
  }
  return null;
}

export function looksLikeDingTalkId(raw: string): boolean {
  return normalizeDingTalkTarget(raw) !== null;
}
