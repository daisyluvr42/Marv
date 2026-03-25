export type SecurityAuditSeverity = "info" | "warn" | "critical";

export type SecurityAuditFinding = {
  checkId: string;
  severity: SecurityAuditSeverity;
  title: string;
  detail: string;
  remediation?: string;
};
