import fs from "node:fs";
import path from "node:path";

// Candidate startup files in priority order — the first that exists on disk is required.
// This makes the audit project-aware rather than assuming a specific repo layout.
const STARTUP_FILE_CANDIDATES = ["WORKFLOW_AUTO.md", "AGENTS.md", "CLAUDE.md"];

/**
 * Build the required-reads list for a workspace by probing which startup files
 * actually exist on disk.  Falls back to an empty list when none are found.
 */
export function resolveRequiredReads(workspaceDir: string): Array<string | RegExp> {
  const required: Array<string | RegExp> = [];
  for (const candidate of STARTUP_FILE_CANDIDATES) {
    try {
      if (fs.existsSync(path.resolve(workspaceDir, candidate))) {
        required.push(candidate);
        break; // only require the first matching startup file
      }
    } catch {
      // Ignore permission / fs errors — best-effort
    }
  }
  // Daily memory file is still expected when the memory directory exists
  try {
    const memoryDir = path.resolve(workspaceDir, "memory");
    if (fs.existsSync(memoryDir) && fs.statSync(memoryDir).isDirectory()) {
      required.push(/memory\/\d{4}-\d{2}-\d{2}\.md/);
    }
  } catch {
    // best-effort
  }
  return required;
}

/**
 * Audit whether agent read required startup files after compaction.
 * Returns list of missing file patterns.
 */
export function auditPostCompactionReads(
  readFilePaths: string[],
  workspaceDir: string,
  requiredReads?: Array<string | RegExp>,
): { passed: boolean; missingPatterns: string[] } {
  if (!requiredReads) {
    requiredReads = resolveRequiredReads(workspaceDir);
  }
  if (requiredReads.length === 0) {
    return { passed: true, missingPatterns: [] };
  }
  const normalizedReads = readFilePaths.map((p) => path.resolve(workspaceDir, p));
  const missingPatterns: string[] = [];

  for (const required of requiredReads) {
    if (typeof required === "string") {
      const requiredResolved = path.resolve(workspaceDir, required);
      const found = normalizedReads.some((r) => r === requiredResolved);
      if (!found) {
        missingPatterns.push(required);
      }
    } else {
      // RegExp — match against relative paths from workspace
      const found = readFilePaths.some((p) => {
        const rel = path.relative(workspaceDir, path.resolve(workspaceDir, p));
        // Normalize to forward slashes for cross-platform RegExp matching
        const normalizedRel = rel.split(path.sep).join("/");
        return required.test(normalizedRel);
      });
      if (!found) {
        missingPatterns.push(required.source);
      }
    }
  }

  return { passed: missingPatterns.length === 0, missingPatterns };
}

/**
 * Read messages from a session JSONL file.
 * Returns messages from the last N lines (default 100).
 */
export function readSessionMessages(
  sessionFile: string,
  maxLines = 100,
): Array<{ role?: string; content?: unknown }> {
  if (!fs.existsSync(sessionFile)) {
    return [];
  }

  try {
    const content = fs.readFileSync(sessionFile, "utf-8");
    const lines = content.trim().split("\n");
    const recentLines = lines.slice(-maxLines);

    const messages: Array<{ role?: string; content?: unknown }> = [];
    for (const line of recentLines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === "message" && entry.message) {
          messages.push(entry.message);
        }
      } catch {
        // Skip malformed lines
      }
    }
    return messages;
  } catch {
    return [];
  }
}

/**
 * Extract file paths from Read tool calls in agent messages.
 * Supports both transcript shapes:
 *   - `tool_use`  blocks with `input.file_path` / `input.path`  (SessionManager JSONL)
 *   - `toolCall`  blocks with `arguments.file_path` / `arguments.path`  (pi-agent-core)
 */
export function extractReadPaths(messages: Array<{ role?: string; content?: unknown }>): string[] {
  const paths: string[] = [];
  for (const msg of messages) {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) {
      continue;
    }
    for (const block of msg.content) {
      if (block.type === "tool_use" && block.name === "read") {
        const filePath = block.input?.file_path ?? block.input?.path;
        if (typeof filePath === "string") {
          paths.push(filePath);
        }
      } else if (block.type === "toolCall" && block.name === "read") {
        const filePath = block.arguments?.file_path ?? block.arguments?.path;
        if (typeof filePath === "string") {
          paths.push(filePath);
        }
      }
    }
  }
  return paths;
}

/** Format the audit warning message */
export function formatAuditWarning(missingPatterns: string[]): string {
  const fileList = missingPatterns.map((p) => `  - ${p}`).join("\n");
  return (
    "⚠️ Post-Compaction Audit: The following required startup files were not read after context reset:\n" +
    fileList +
    "\n\nPlease read them now using the Read tool before continuing. " +
    "This ensures your operating protocols are restored after memory compaction."
  );
}
