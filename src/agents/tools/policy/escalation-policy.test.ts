import { describe, expect, it } from "vitest";
import { buildEscalationBlockReason, classifyEscalationRequirement } from "./escalation-policy.js";

describe("escalation-policy", () => {
  it("classifies high-risk exec commands as execute_escalated", () => {
    expect(
      classifyEscalationRequirement({
        toolName: "exec",
        params: { command: "sudo launchctl kickstart -k system/com.test.agent" },
      }),
    ).toEqual(
      expect.objectContaining({
        category: "execute_escalated",
        requiredLevel: "execute",
      }),
    );
  });

  it("classifies gateway config writes as admin", () => {
    expect(
      classifyEscalationRequirement({
        toolName: "gateway",
        params: { action: "config.patch", raw: '{"gateway":{"bind":"lan"}}' },
      }),
    ).toEqual(
      expect.objectContaining({
        category: "admin",
        requiredLevel: "admin",
      }),
    );
  });

  it("does not require escalation for cron mutations", () => {
    expect(
      classifyEscalationRequirement({
        toolName: "cron",
        params: {
          action: "add",
          job: {
            name: "Morning brief",
            delivery: { mode: "webhook", to: "https://example.com/hook" },
          },
        },
      }),
    ).toEqual({ category: "none" });
  });

  it("classifies secret-bearing message sends as resource transfer", () => {
    expect(
      classifyEscalationRequirement({
        toolName: "message",
        params: {
          action: "send",
          text: "Here is the API key: sk-1234567890abcdef1234567890",
        },
      }),
    ).toEqual(
      expect.objectContaining({
        category: "resource_transfer",
        requiredLevel: "admin",
      }),
    );
  });

  it("classifies role grants as resource transfer", () => {
    expect(
      classifyEscalationRequirement({
        toolName: "message",
        params: { action: "role-add", target: "user-1" },
      }),
    ).toEqual(
      expect.objectContaining({
        category: "resource_transfer",
        requiredLevel: "admin",
      }),
    );
  });

  it("builds a stricter block reason for non-direct resource transfer", () => {
    const reason = buildEscalationBlockReason({
      requirement: {
        category: "resource_transfer",
        requiredLevel: "admin",
        reason: "This message would share a secret.",
        scope: "message",
      },
      taskId: "task-1",
      directUserInstruction: false,
    });

    expect(reason).toContain("forwarded or third-party content");
    expect(reason).toContain('requestedLevel="admin"');
  });
});
