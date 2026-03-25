import type { SkillScanFinding } from "../security/skill-scanner.js";

export type SkillInstallResult = {
  ok: boolean;
  message: string;
  stdout: string;
  stderr: string;
  code: number | null;
  warnings?: string[];
  scan?: SkillInstallSafetyReport;
};

export type SkillInstallSafetyLevel = "clean" | "warn" | "critical";

export type SkillInstallSafetyReport = {
  level: SkillInstallSafetyLevel;
  warnings: string[];
  findings: SkillScanFinding[];
  blocked: boolean;
};
