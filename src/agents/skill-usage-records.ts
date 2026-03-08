import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export type SkillUsageRecord = {
  skillId: string;
  installedAt: number;
  firstUsedAt?: number;
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
        ok: record.ok !== false,
        ...(typeof record.firstUsedAt === "number" && Number.isFinite(record.firstUsedAt)
          ? { firstUsedAt: record.firstUsedAt }
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
    ok: records[params.skillId]?.ok ?? true,
    failureReason: records[params.skillId]?.failureReason,
    quarantined: records[params.skillId]?.quarantined ?? false,
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
