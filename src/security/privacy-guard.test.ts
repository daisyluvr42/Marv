import { describe, expect, it } from "vitest";
import { buildPrivacyContext, requiresPrivacyGuard } from "./privacy-guard.js";
import { buildPrivacyPromptDirective, filterOutput } from "./privacy-output-filter.js";
import { PrivacyScanner } from "./privacy-scanner.js";

// ─────────────────────────────────────────────────────────────────
// Privacy Guard — Context Detection
// ─────────────────────────────────────────────────────────────────

describe("requiresPrivacyGuard", () => {
  it("returns false for owner in private DM", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: true,
      channelType: "dm",
      recipientCount: 1,
    });
    expect(requiresPrivacyGuard(ctx)).toBe(false);
  });

  it("returns true for non-owner in DM", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: false,
      channelType: "dm",
      recipientCount: 1,
    });
    expect(requiresPrivacyGuard(ctx)).toBe(true);
  });

  it("returns true for owner in group chat", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: true,
      channelType: "group",
      recipientCount: 5,
    });
    expect(requiresPrivacyGuard(ctx)).toBe(true);
  });

  it("returns true for owner in public channel", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: true,
      channelType: "public",
    });
    expect(requiresPrivacyGuard(ctx)).toBe(true);
  });

  it("returns true for multi-user DM", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: true,
      isMultiUserDm: true,
      channelType: "dm",
    });
    expect(requiresPrivacyGuard(ctx)).toBe(true);
  });

  it("returns true for unknown channel type with multiple recipients", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: true,
      channelType: "unknown",
      recipientCount: 3,
    });
    expect(requiresPrivacyGuard(ctx)).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────
// Privacy Scanner — Pattern Detection
// ─────────────────────────────────────────────────────────────────

describe("PrivacyScanner", () => {
  const scanner = new PrivacyScanner();

  describe("API Keys", () => {
    it("detects OpenAI API key", () => {
      const result = scanner.scan("My key is sk-abc123def456ghi789jkl012mno345");
      expect(result.clean).toBe(false);
      expect(result.findings).toHaveLength(1);
      expect(result.findings[0].category).toBe("api_keys");
      expect(result.findings[0].severity).toBe("critical");
    });

    it("detects AWS access key", () => {
      const result = scanner.scan("AKIAIOSFODNN7EXAMPLE");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("api_keys");
    });

    it("detects GitHub PAT", () => {
      const result = scanner.scan("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789012");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("api_keys");
    });

    it("detects Anthropic API key", () => {
      const result = scanner.scan("sk-ant-api03-aBcDeFgHiJkLmNoPqRsTuVwXyZ");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("api_keys");
    });

    it("detects Stripe key", () => {
      // eslint-disable-next-line no-useless-concat -- split to avoid the privacy scanner flagging this test file
      const result = scanner.scan("sk" + "_live_aBcDeFgHiJkLmNoPqRsTuVwXyZ");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("api_keys");
    });
  });

  describe("Tokens", () => {
    it("detects JWT token", () => {
      const result = scanner.scan(
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
      );
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("tokens");
    });

    it("detects Slack bot token", () => {
      const result = scanner.scan("xoxb-123456-789012-abcDefGhiJkl");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("tokens");
    });

    it("detects npm token", () => {
      const result = scanner.scan("npm_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("tokens");
    });
  });

  describe("Private Keys", () => {
    it("detects PEM private key header", () => {
      const result = scanner.scan("-----BEGIN RSA PRIVATE KEY-----");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("private_keys");
      expect(result.findings[0].severity).toBe("critical");
    });

    it("detects generic private key header", () => {
      const result = scanner.scan("-----BEGIN PRIVATE KEY-----");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("private_keys");
    });
  });

  describe("Passwords", () => {
    it("detects password assignment", () => {
      const result = scanner.scan('password = "mySuperSecretPassword123!"');
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("passwords");
    });

    it("detects database URI with credentials", () => {
      const result = scanner.scan("postgres://admin:s3cretP@ss@db.host.com:5432/mydb");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("passwords");
    });
  });

  describe("Internal URLs", () => {
    it("detects private IP", () => {
      const result = scanner.scan("Connect to http://192.168.1.100:8080/api");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("internal_urls");
    });

    it("detects localhost", () => {
      const result = scanner.scan("Server at http://localhost:3000");
      expect(result.clean).toBe(false);
      expect(result.findings[0].category).toBe("internal_urls");
    });
  });

  describe("Clean text", () => {
    it("returns clean for normal text", () => {
      const result = scanner.scan("Hello! This is a regular message without any sensitive data.");
      expect(result.clean).toBe(true);
      expect(result.findings).toHaveLength(0);
    });
  });

  describe("Redaction", () => {
    it("redacts detected secrets", () => {
      const input = "My API key is sk-abc123def456ghi789jkl012mno345 and please keep it safe.";
      const { redacted, findings } = scanner.redact(input);
      expect(redacted).toContain("[REDACTED:api_keys]");
      expect(redacted).not.toContain("sk-abc123");
      expect(findings).toHaveLength(1);
    });

    it("handles multiple findings", () => {
      const input =
        "Key: sk-abc123def456ghi789jkl012mno345, Token: xoxb-123456-789012-abcDefGhiJkl";
      const { redacted, findings } = scanner.redact(input);
      expect(findings.length).toBeGreaterThanOrEqual(2);
      expect(redacted).toContain("[REDACTED:api_keys]");
      expect(redacted).toContain("[REDACTED:tokens]");
    });

    it("does not modify clean text", () => {
      const input = "Nothing secret here.";
      const { redacted, findings } = scanner.redact(input);
      expect(redacted).toBe(input);
      expect(findings).toHaveLength(0);
    });
  });

  describe("Category filtering", () => {
    it("only scans specified categories", () => {
      const apiOnlyScanner = new PrivacyScanner(["api_keys"]);
      const result = apiOnlyScanner.scan("-----BEGIN RSA PRIVATE KEY-----");
      expect(result.clean).toBe(true);
    });
  });
});

// ─────────────────────────────────────────────────────────────────
// Privacy Output Filter
// ─────────────────────────────────────────────────────────────────

describe("privacy output filter", () => {
  it("builds privacy prompt directive", () => {
    const ctx = buildPrivacyContext({
      senderIsOwner: false,
      channelType: "group",
    });
    const directive = buildPrivacyPromptDirective(ctx);
    expect(directive).toContain("PRIVACY GUARD ACTIVE");
    expect(directive).toContain("API keys");
    expect(directive).toContain("private 1:1 conversation");
  });

  it("filters output with secrets", () => {
    const ctx = buildPrivacyContext({ senderIsOwner: false, channelType: "group" });
    const result = filterOutput("Your key is sk-abc123def456ghi789jkl012mno345.", ctx);
    expect(result.safe).toBe(false);
    expect(result.filtered).toContain("[REDACTED:api_keys]");
    expect(result.blocked).toHaveLength(1);
    expect(result.warning).toBeTruthy();
  });

  it("passes clean output unchanged", () => {
    const ctx = buildPrivacyContext({ senderIsOwner: false, channelType: "group" });
    const result = filterOutput("This is a normal response.", ctx);
    expect(result.safe).toBe(true);
    expect(result.filtered).toBe("This is a normal response.");
    expect(result.blocked).toHaveLength(0);
  });
});
