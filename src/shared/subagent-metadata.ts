export const SUBAGENT_ANNOUNCE_MODES = ["child", "aggregate"] as const;

export type SubagentAnnounceMode = (typeof SUBAGENT_ANNOUNCE_MODES)[number];

export type SubagentSessionMetadata = {
  subagentRole?: string;
  subagentPreset?: string;
  subagentTaskGroup?: string;
  subagentDispatchId?: string;
  subagentAnnounceMode?: SubagentAnnounceMode;
};

export function normalizeSubagentMetadataValue(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

export function isSubagentAnnounceMode(value: unknown): value is SubagentAnnounceMode {
  return (
    typeof value === "string" && SUBAGENT_ANNOUNCE_MODES.includes(value as SubagentAnnounceMode)
  );
}

export function normalizeSubagentAnnounceMode(value: unknown): SubagentAnnounceMode | undefined {
  const normalized = normalizeSubagentMetadataValue(value)?.toLowerCase();
  return isSubagentAnnounceMode(normalized) ? normalized : undefined;
}
