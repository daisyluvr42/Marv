import { describe, expect, it } from "vitest";
import type { MarvConfig } from "./config.js";
import type { GroupToolPolicySender } from "./group-policy.js";
import { resolveGroupMemberRole } from "./group-roles.js";

function cfg(overrides?: Partial<MarvConfig>): MarvConfig {
  return { ...overrides } as MarvConfig;
}

function sender(overrides?: Partial<GroupToolPolicySender>): GroupToolPolicySender {
  return { senderId: "user123", ...overrides };
}

describe("resolveGroupMemberRole", () => {
  it("returns member when sender has no identifiers", () => {
    expect(resolveGroupMemberRole({ cfg: cfg(), sender: {} })).toBe("member");
  });

  it("returns member when no ownerAllowFrom and no channelAllowFrom", () => {
    expect(resolveGroupMemberRole({ cfg: cfg(), sender: sender() })).toBe("member");
  });

  it("returns owner when sender matches ownerAllowFrom", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["user123"] } }),
      sender: sender(),
    });
    expect(result).toBe("owner");
  });

  it("returns member when sender does not match ownerAllowFrom", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["someone_else"] } }),
      sender: sender(),
    });
    expect(result).toBe("member");
  });

  it("returns owner when ownerAllowFrom is wildcard", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["*"] } }),
      sender: sender(),
    });
    expect(result).toBe("owner");
  });

  it("handles channel-prefixed ownerAllowFrom entries", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["telegram:user123"] } }),
      providerId: "telegram",
      sender: sender(),
    });
    expect(result).toBe("owner");
  });

  it("skips channel-prefixed entries for different provider", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["telegram:user123"] } }),
      providerId: "discord",
      sender: sender(),
    });
    expect(result).toBe("member");
  });

  it("falls back to channelAllowFrom when ownerAllowFrom is not configured", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg(),
      channelAllowFrom: ["user123"],
      sender: sender(),
    });
    expect(result).toBe("owner");
  });

  it("returns member when channelAllowFrom is wildcard only", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg(),
      channelAllowFrom: ["*"],
      sender: sender(),
    });
    expect(result).toBe("member");
  });

  it("returns member when sender not in channelAllowFrom", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg(),
      channelAllowFrom: ["other_user"],
      sender: sender(),
    });
    expect(result).toBe("member");
  });

  it("matches case-insensitively", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["User123"] } }),
      sender: sender({ senderId: "user123" }),
    });
    expect(result).toBe("owner");
  });

  it("matches by senderE164", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["+15551234567"] } }),
      sender: sender({ senderId: "other", senderE164: "+15551234567" }),
    });
    expect(result).toBe("owner");
  });

  it("matches by senderUsername", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["jdoe"] } }),
      sender: sender({ senderId: "id999", senderUsername: "jdoe" }),
    });
    expect(result).toBe("owner");
  });

  it("skips empty ownerAllowFrom entries", () => {
    const result = resolveGroupMemberRole({
      cfg: cfg({ commands: { ownerAllowFrom: ["", "  ", "user123"] } }),
      sender: sender(),
    });
    expect(result).toBe("owner");
  });
});
