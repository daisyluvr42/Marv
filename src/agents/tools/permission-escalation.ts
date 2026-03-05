/**
 * Permission Escalation Manager — Task-scoped privilege management.
 *
 * Allows the agent to request elevated permissions for specific operations
 * during a task. All granted permissions are automatically revoked when the
 * task ends (taskScoped = true, enforced).
 *
 * Integrates with the existing exec-approvals system for user notification.
 */

export type EscalationLevel = "read" | "write" | "execute" | "admin";

export type EscalationRequest = {
  /** Unique ID for this escalation request. */
  requestId: string;
  /** Agent requesting the escalation. */
  agentId: string;
  /** Task to which this escalation is bound. */
  taskId: string;
  /** Current permission level. */
  currentLevel: EscalationLevel;
  /** Requested permission level. */
  requestedLevel: EscalationLevel;
  /** Human-readable reason for the escalation. */
  reason: string;
  /** Tool that needs the elevated permission. */
  toolName?: string;
  /** Scope of the escalation (directory, service, etc.). */
  scope?: string;
  /** When this request was created. */
  createdAt: number;
};

export type EscalationDecision = {
  requestId: string;
  decision: "approve" | "deny" | "approve-once";
  decidedBy: "user" | "auto";
  decidedAt: number;
  /** Task this approval is scoped to. Cleared when the task ends. */
  taskId: string;
};

type GrantedEscalation = {
  decision: EscalationDecision;
  request: EscalationRequest;
};

const LEVEL_RANK: Record<EscalationLevel, number> = {
  read: 0,
  write: 1,
  execute: 2,
  admin: 3,
};

export class PermissionEscalationManager {
  /** Granted permissions indexed by taskId. */
  private granted: Map<string, GrantedEscalation[]> = new Map();
  /** Pending requests awaiting user decision. */
  private pending: Map<string, EscalationRequest> = new Map();

  /**
   * Check whether the given level is already granted for the task/scope.
   */
  checkPermission(taskId: string, requiredLevel: EscalationLevel, scope?: string): boolean {
    const grants = this.granted.get(taskId);
    if (!grants || grants.length === 0) {
      // Base level is "read" — always available.
      return LEVEL_RANK[requiredLevel] <= LEVEL_RANK.read;
    }
    return grants.some((grant) => {
      if (grant.decision.decision === "deny") {
        return false;
      }
      if (scope && grant.request.scope && grant.request.scope !== scope) {
        return false;
      }
      return LEVEL_RANK[grant.request.requestedLevel] >= LEVEL_RANK[requiredLevel];
    });
  }

  /**
   * Build an escalation request.
   * Returns the request for the caller to present to the user.
   */
  createRequest(params: {
    agentId: string;
    taskId: string;
    currentLevel: EscalationLevel;
    requestedLevel: EscalationLevel;
    reason: string;
    toolName?: string;
    scope?: string;
  }): EscalationRequest {
    const requestId = `esc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const request: EscalationRequest = {
      requestId,
      agentId: params.agentId,
      taskId: params.taskId,
      currentLevel: params.currentLevel,
      requestedLevel: params.requestedLevel,
      reason: params.reason,
      toolName: params.toolName,
      scope: params.scope,
      createdAt: Date.now(),
    };
    this.pending.set(requestId, request);
    return request;
  }

  /**
   * Record a user's decision on a pending request.
   */
  recordDecision(
    requestId: string,
    decision: EscalationDecision["decision"],
  ): EscalationDecision | null {
    const request = this.pending.get(requestId);
    if (!request) {
      return null;
    }

    const record: EscalationDecision = {
      requestId,
      decision,
      decidedBy: "user",
      decidedAt: Date.now(),
      taskId: request.taskId,
    };

    if (decision === "approve" || decision === "approve-once") {
      const existing = this.granted.get(request.taskId) ?? [];
      existing.push({ decision: record, request });
      this.granted.set(request.taskId, existing);
    }

    this.pending.delete(requestId);
    return record;
  }

  /**
   * Revoke all escalated permissions for a task.
   * Called automatically when a task completes or is archived.
   */
  revokeTaskPermissions(taskId: string): number {
    const decisions = this.granted.get(taskId);
    const count = decisions?.length ?? 0;
    this.granted.delete(taskId);
    // Also clean up any pending requests for this task.
    for (const [id, req] of this.pending) {
      if (req.taskId === taskId) {
        this.pending.delete(id);
      }
    }
    return count;
  }

  /**
   * Get all pending escalation requests.
   */
  getPendingRequests(taskId?: string): EscalationRequest[] {
    const all = Array.from(this.pending.values());
    return taskId ? all.filter((r) => r.taskId === taskId) : all;
  }

  /**
   * Get all granted permissions for a task.
   */
  getGrantedPermissions(taskId: string): EscalationDecision[] {
    return (this.granted.get(taskId) ?? []).map((item) => item.decision);
  }
}

// ── Singleton ──
let _instance: PermissionEscalationManager | null = null;

export function getEscalationManager(): PermissionEscalationManager {
  if (!_instance) {
    _instance = new PermissionEscalationManager();
  }
  return _instance;
}

/** Reset for testing. */
export function resetEscalationManager(): void {
  _instance = null;
}
