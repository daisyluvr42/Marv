import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export type SkillUsageRecord = {
  skillId: string;
  installedAt: number;
  firstUsedAt?: number;
  lastUsedAt?: number;
  successCount: number;
  failureCount: number;
  lastOutcome?: "success" | "failure";
  lastValidatedAt?: number;
  ok: boolean;
  failureReason?: string;
  quarantined: boolean;
};

type SkillUsageRecordMap = Record<string, SkillUsageRecord>;

export function resolveSkillUsageRecordsPath(): string {
  return path.join(os.homedir(), ".marv", "skills", ".usage-records.json");
}

export async function readSkillUsageRecords(
  recordsPath = resolveSkillUsageRecordsPath(),
): Promise<SkillUsageRecordMap> {
  try {
    const raw = await fs.readFile(recordsPath, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const records: SkillUsageRecordMap = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const record = value as Partial<SkillUsageRecord>;
      const skillId =
        typeof record.skillId === "string" && record.skillId.trim() ? record.skillId : key;
      const installedAt =
        typeof record.installedAt === "number" && Number.isFinite(record.installedAt)
          ? record.installedAt
          : 0;
      records[key] = {
        skillId,
        installedAt,
        successCount:
          typeof record.successCount === "number" && Number.isFinite(record.successCount)
            ? Math.max(0, Math.trunc(record.successCount))
            : 0,
        failureCount:
          typeof record.failureCount === "number" && Number.isFinite(record.failureCount)
            ? Math.max(0, Math.trunc(record.failureCount))
            : 0,
        ok: record.ok !== false,
        ...(typeof record.firstUsedAt === "number" && Number.isFinite(record.firstUsedAt)
          ? { firstUsedAt: record.firstUsedAt }
          : {}),
        ...(typeof record.lastUsedAt === "number" && Number.isFinite(record.lastUsedAt)
          ? { lastUsedAt: record.lastUsedAt }
          : {}),
        ...(record.lastOutcome === "success" || record.lastOutcome === "failure"
          ? { lastOutcome: record.lastOutcome }
          : {}),
        ...(typeof record.lastValidatedAt === "number" && Number.isFinite(record.lastValidatedAt)
          ? { lastValidatedAt: record.lastValidatedAt }
          : {}),
        ...(typeof record.failureReason === "string" && record.failureReason.trim()
          ? { failureReason: record.failureReason.trim() }
          : {}),
        quarantined: record.quarantined === true,
      };
    }
    return records;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException | undefined)?.code;
    if (code === "ENOENT") {
      return {};
    }
    return {};
  }
}

async function writeSkillUsageRecords(
  records: SkillUsageRecordMap,
  recordsPath = resolveSkillUsageRecordsPath(),
): Promise<void> {
  await fs.mkdir(path.dirname(recordsPath), { recursive: true });
  await fs.writeFile(recordsPath, `${JSON.stringify(records, null, 2)}\n`, "utf8");
}

export async function markInstalledSkillUsageRecord(params: {
  skillId: string;
  installedAt?: number;
  recordsPath?: string;
}): Promise<void> {
  const records = await readSkillUsageRecords(params.recordsPath);
  records[params.skillId] = {
    skillId: params.skillId,
    installedAt: params.installedAt ?? Date.now(),
    firstUsedAt: records[params.skillId]?.firstUsedAt,
    lastUsedAt: records[params.skillId]?.lastUsedAt,
    successCount: records[params.skillId]?.successCount ?? 0,
    failureCount: records[params.skillId]?.failureCount ?? 0,
    lastOutcome: records[params.skillId]?.lastOutcome,
    lastValidatedAt: records[params.skillId]?.lastValidatedAt,
    ok: records[params.skillId]?.ok ?? true,
    failureReason: records[params.skillId]?.failureReason,
    quarantined: records[params.skillId]?.quarantined ?? false,
  };
  await writeSkillUsageRecords(records, params.recordsPath);
}

export async function markSkillUsed(params: {
  skillId: string;
  usedAt?: number;
  recordsPath?: string;
}): Promise<void> {
  const records = await readSkillUsageRecords(params.recordsPath);
  const usedAt = params.usedAt ?? Date.now();
  const existing = records[params.skillId];
  records[params.skillId] = {
    skillId: params.skillId,
    installedAt: existing?.installedAt ?? usedAt,
    firstUsedAt: existing?.firstUsedAt ?? usedAt,
    lastUsedAt: usedAt,
    successCount: existing?.successCount ?? 0,
    failureCount: existing?.failureCount ?? 0,
    lastOutcome: existing?.lastOutcome,
    lastValidatedAt: existing?.lastValidatedAt,
    ok: existing?.ok ?? true,
    failureReason: existing?.failureReason,
    quarantined: existing?.quarantined ?? false,
  };
  await writeSkillUsageRecords(records, params.recordsPath);
}

export async function markSkillOutcome(params: {
  skillId: string;
  outcome: "success" | "failure";
  validatedAt?: number;
  failureReason?: string;
  recordsPath?: string;
}): Promise<void> {
  const records = await readSkillUsageRecords(params.recordsPath);
  const now = params.validatedAt ?? Date.now();
  const existing = records[params.skillId];
  const successCount = existing?.successCount ?? 0;
  const failureCount = existing?.failureCount ?? 0;
  records[params.skillId] = {
    skillId: params.skillId,
    installedAt: existing?.installedAt ?? now,
    firstUsedAt: existing?.firstUsedAt ?? now,
    lastUsedAt: now,
    successCount: params.outcome === "success" ? successCount + 1 : successCount,
    failureCount: params.outcome === "failure" ? failureCount + 1 : failureCount,
    lastOutcome: params.outcome,
    lastValidatedAt: now,
    ok: params.outcome === "success",
    failureReason:
      params.outcome === "failure"
        ? params.failureReason?.trim() || existing?.failureReason
        : undefined,
    quarantined: existing?.quarantined ?? false,
  };
  await writeSkillUsageRecords(records, params.recordsPath);
}

export async function removeSkillUsageRecord(params: {
  skillId: string;
  recordsPath?: string;
}): Promise<void> {
  const records = await readSkillUsageRecords(params.recordsPath);
  delete records[params.skillId];
  await writeSkillUsageRecords(records, params.recordsPath);
}

export function isSkillQuarantined(
  skillId: string,
  records: SkillUsageRecordMap | undefined,
): boolean {
  return records?.[skillId]?.quarantined === true;
}
