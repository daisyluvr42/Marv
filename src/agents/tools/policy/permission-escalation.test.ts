import { afterEach, describe, expect, it } from "vitest";
import {
  PermissionEscalationManager,
  getEscalationManager,
  resetEscalationManager,
} from "./permission-escalation.js";

describe("PermissionEscalationManager", () => {
  afterEach(() => {
    resetEscalationManager();
  });

  it("grants read by default without any escalation", () => {
    const mgr = new PermissionEscalationManager();
    expect(mgr.checkPermission("task-1", "read")).toBe(true);
    expect(mgr.checkPermission("task-1", "write")).toBe(false);
    expect(mgr.checkPermission("task-1", "execute")).toBe(false);
    expect(mgr.checkPermission("task-1", "admin")).toBe(false);
  });

  it("creates an escalation request", () => {
    const mgr = new PermissionEscalationManager();
    const req = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "execute",
      reason: "Need to run sudo command",
      toolName: "exec",
    });
    expect(req.requestId).toMatch(/^esc-/);
    expect(req.taskId).toBe("task-1");
    expect(req.requestedLevel).toBe("execute");
    expect(mgr.getPendingRequests()).toHaveLength(1);
  });

  it("records approval and grants permission", () => {
    const mgr = new PermissionEscalationManager();
    const req = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "write",
      reason: "Need to write system files",
    });

    const decision = mgr.recordDecision(req.requestId, "approve");
    expect(decision).not.toBeNull();
    expect(decision!.decision).toBe("approve");
    expect(decision!.taskId).toBe("task-1");

    expect(mgr.getPendingRequests()).toHaveLength(0);
    expect(mgr.getGrantedPermissions("task-1")).toHaveLength(1);
    expect(mgr.checkPermission("task-1", "write")).toBe(true);
    expect(mgr.checkPermission("task-1", "execute")).toBe(false);
  });

  it("records denial and does not grant permission", () => {
    const mgr = new PermissionEscalationManager();
    const req = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "admin",
      reason: "Need root access",
    });

    mgr.recordDecision(req.requestId, "deny");

    expect(mgr.getPendingRequests()).toHaveLength(0);
    expect(mgr.getGrantedPermissions("task-1")).toHaveLength(0);
  });

  it("revokes all task permissions on task end", () => {
    const mgr = new PermissionEscalationManager();
    const req1 = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "write",
      reason: "Step 1",
    });
    const req2 = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "write",
      requestedLevel: "execute",
      reason: "Step 2",
    });

    mgr.recordDecision(req1.requestId, "approve");
    mgr.recordDecision(req2.requestId, "approve");
    expect(mgr.getGrantedPermissions("task-1")).toHaveLength(2);

    const revoked = mgr.revokeTaskPermissions("task-1");
    expect(revoked).toBe(2);
    expect(mgr.getGrantedPermissions("task-1")).toHaveLength(0);
  });

  it("does not leak permissions between tasks", () => {
    const mgr = new PermissionEscalationManager();
    const req = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "admin",
      reason: "Admin access",
    });
    mgr.recordDecision(req.requestId, "approve");

    // task-2 should have no permissions
    expect(mgr.getGrantedPermissions("task-2")).toHaveLength(0);
    expect(mgr.checkPermission("task-2", "admin")).toBe(false);
  });

  it("enforces escalation scope when provided", () => {
    const mgr = new PermissionEscalationManager();
    const req = mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "execute",
      reason: "Need elevated shell command",
      scope: "/tmp/project-a",
    });
    mgr.recordDecision(req.requestId, "approve");

    expect(mgr.checkPermission("task-1", "execute", "/tmp/project-a")).toBe(true);
    expect(mgr.checkPermission("task-1", "execute", "/tmp/project-b")).toBe(false);
  });

  it("cleans up pending requests on task revocation", () => {
    const mgr = new PermissionEscalationManager();
    mgr.createRequest({
      agentId: "agent-1",
      taskId: "task-1",
      currentLevel: "read",
      requestedLevel: "write",
      reason: "Pending request",
    });
    expect(mgr.getPendingRequests("task-1")).toHaveLength(1);

    mgr.revokeTaskPermissions("task-1");
    expect(mgr.getPendingRequests("task-1")).toHaveLength(0);
  });

  it("returns null for unknown request id", () => {
    const mgr = new PermissionEscalationManager();
    expect(mgr.recordDecision("nonexistent", "approve")).toBeNull();
  });

  it("singleton works", () => {
    const mgr1 = getEscalationManager();
    const mgr2 = getEscalationManager();
    expect(mgr1).toBe(mgr2);
  });
});
