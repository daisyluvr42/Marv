/**
 * Privacy Scanner — Detects sensitive information in text.
 *
 * Scans agent output for API keys, passwords, tokens, private keys, and
 * other sensitive data patterns. Used by the PrivacyOutputFilter to redact
 * secrets before they reach non-owner recipients.
 */

import type { PrivacyCategory } from "../core/config/types.autonomy.js";
import { ALL_PRIVACY_CATEGORIES } from "../core/config/types.autonomy.js";

export type ScanFinding = {
  /** Which category of sensitive data was matched. */
  category: PrivacyCategory;
  /** Human-readable pattern name for logging. */
  patternName: string;
  /** Start and end character indices of the match. */
  position: { start: number; end: number };
  /** Redacted preview for safe logging (e.g. "sk-****...****"). */
  preview: string;
  /** How critical is this finding. */
  severity: "critical" | "warning";
};

export type ScanResult = {
  /** True if no sensitive data was found. */
  clean: boolean;
  /** All findings. */
  findings: ScanFinding[];
};

type PatternDef = {
  name: string;
  category: PrivacyCategory;
  regex: RegExp;
  severity: "critical" | "warning";
};

/**
 * Built-in pattern library for detecting sensitive data.
 * Each entry is [name, category, pattern, severity].
 */
const PATTERN_DEFS: PatternDef[] = [
  // ── API Keys ──
  {
    name: "OpenAI API Key",
    category: "api_keys",
    regex: /sk-[a-zA-Z0-9]{20,}/g,
    severity: "critical",
  },
  {
    name: "OpenAI Project Key",
    category: "api_keys",
    regex: /sk-proj-[a-zA-Z0-9_-]{20,}/g,
    severity: "critical",
  },
  {
    name: "AWS Access Key",
    category: "api_keys",
    regex: /AKIA[0-9A-Z]{16}/g,
    severity: "critical",
  },
  { name: "GitHub PAT", category: "api_keys", regex: /ghp_[a-zA-Z0-9]{36}/g, severity: "critical" },
  {
    name: "GitHub OAuth",
    category: "api_keys",
    regex: /gho_[a-zA-Z0-9]{36}/g,
    severity: "critical",
  },
  {
    name: "GitHub Fine-Grained PAT",
    category: "api_keys",
    regex: /github_pat_[a-zA-Z0-9_]{22,}/g,
    severity: "critical",
  },
  {
    name: "Google API Key",
    category: "api_keys",
    regex: /AIza[0-9A-Za-z_-]{35}/g,
    severity: "critical",
  },
  {
    name: "Anthropic API Key",
    category: "api_keys",
    regex: /sk-ant-[a-zA-Z0-9_-]{20,}/g,
    severity: "critical",
  },
  {
    name: "Stripe Key",
    category: "api_keys",
    regex: /(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{20,}/g,
    severity: "critical",
  },
  { name: "Supabase Key", category: "api_keys", regex: /sbp_[a-f0-9]{40}/g, severity: "critical" },
  {
    name: "SendGrid API Key",
    category: "api_keys",
    regex: /SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}/g,
    severity: "critical",
  },

  // ── Tokens ──
  {
    name: "JWT Token",
    category: "tokens",
    regex: /eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}/g,
    severity: "critical",
  },
  {
    name: "Slack Bot Token",
    category: "tokens",
    regex: /xoxb-[0-9]+-[a-zA-Z0-9]+/g,
    severity: "critical",
  },
  {
    name: "Slack User Token",
    category: "tokens",
    regex: /xoxp-[0-9]+-[a-zA-Z0-9]+/g,
    severity: "critical",
  },
  {
    name: "Slack App Token",
    category: "tokens",
    regex: /xapp-[0-9]+-[a-zA-Z0-9]+-[0-9]+-[a-zA-Z0-9]+/g,
    severity: "critical",
  },
  {
    name: "Discord Bot Token",
    category: "tokens",
    regex: /[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}/g,
    severity: "critical",
  },
  {
    name: "Telegram Bot Token",
    category: "tokens",
    regex: /\d{9,10}:[A-Za-z0-9_-]{35}/g,
    severity: "critical",
  },
  { name: "npm Token", category: "tokens", regex: /npm_[a-zA-Z0-9]{36}/g, severity: "critical" },
  {
    name: "Bearer Token",
    category: "tokens",
    regex: /Bearer\s+[a-zA-Z0-9_\-.~+/]{20,}/g,
    severity: "warning",
  },

  // ── Private Keys ──
  {
    name: "PEM Private Key",
    category: "private_keys",
    regex: /-----BEGIN\s(?:RSA\s|EC\s|DSA\s|OPENSSH\s)?PRIVATE\sKEY-----/g,
    severity: "critical",
  },
  {
    name: "PGP Private Key",
    category: "private_keys",
    regex: /-----BEGIN\sPGP\sPRIVATE\sKEY\sBLOCK-----/g,
    severity: "critical",
  },

  // ── Passwords ──
  {
    name: "Password Assignment",
    category: "passwords",
    regex: /(?:password|passwd|pwd)\s*[:=]\s*["']?[^\s"']{8,}/gi,
    severity: "critical",
  },
  {
    name: "Database URI with Password",
    category: "passwords",
    regex: /(?:postgres|mysql|mongodb|redis):\/\/[^:]+:[^@]+@/gi,
    severity: "critical",
  },

  // ── Environment Variable Secrets ──
  {
    name: "Env Secret Assignment",
    category: "env_secrets",
    regex:
      /(?:export\s+)?(?:\w*(?:SECRET|KEY|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH)\w*)\s*=\s*["']?[^\s"']{8,}/gi,
    severity: "warning",
  },

  // ── Personal Info ──
  {
    name: "Email Address",
    category: "personal_info",
    regex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    severity: "warning",
  },

  // ── Internal URLs ──
  {
    name: "Private IPv4",
    category: "internal_urls",
    regex:
      /(?:https?:\/\/)?(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?::\d+)?/g,
    severity: "warning",
  },
  {
    name: "Localhost URL",
    category: "internal_urls",
    regex: /https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?(?:\/\S*)?/g,
    severity: "warning",
  },
];

/** Redact a matched string to a safe preview (e.g. "sk-****...****"). */
function redactMatch(match: string): string {
  if (match.length <= 8) {
    return "*".repeat(match.length);
  }
  const prefix = match.slice(0, 4);
  return `${prefix}${"*".repeat(Math.min(8, match.length - 4))}`;
}

export class PrivacyScanner {
  private patterns: PatternDef[];

  constructor(categories?: PrivacyCategory[]) {
    const enabledCategories = new Set(categories ?? ALL_PRIVACY_CATEGORIES);
    this.patterns = PATTERN_DEFS.filter((def) => enabledCategories.has(def.category));
  }

  /** Scan text for sensitive information. */
  scan(text: string): ScanResult {
    const findings: ScanFinding[] = [];

    for (const def of this.patterns) {
      // Reset regex lastIndex for global patterns
      def.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = def.regex.exec(text)) !== null) {
        findings.push({
          category: def.category,
          patternName: def.name,
          position: { start: match.index, end: match.index + match[0].length },
          preview: redactMatch(match[0]),
          severity: def.severity,
        });
      }
    }

    // Sort by position
    findings.sort((a, b) => a.position.start - b.position.start);

    return {
      clean: findings.length === 0,
      findings,
    };
  }

  /**
   * Redact all detected sensitive patterns in the text.
   * Returns the redacted text and list of findings.
   */
  redact(text: string): { redacted: string; findings: ScanFinding[] } {
    const result = this.scan(text);
    if (result.clean) {
      return { redacted: text, findings: [] };
    }

    // Build replacement ranges, handling overlaps
    const ranges = result.findings
      .map((f) => ({ ...f.position, replacement: `[REDACTED:${f.category}]` }))
      .toSorted((a, b) => a.start - b.start);

    let out = "";
    let cursor = 0;
    for (const range of ranges) {
      if (range.start < cursor) {
        continue; // Skip overlapping
      }
      out += text.slice(cursor, range.start);
      out += range.replacement;
      cursor = range.end;
    }
    out += text.slice(cursor);

    return { redacted: out, findings: result.findings };
  }
}
