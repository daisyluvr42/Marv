import type { CliBackendConfig } from "../../../core/config/types.js";
import type {
  ExternalCliAdapterId,
  ExternalCliOverrideConfig,
} from "../../../core/config/types.tools.js";
import { runExec } from "../../../process/exec.js";
import { CLAUDE_MODEL_ALIASES } from "../../cli-backends.js";
import { normalizeCliModel, parseCliJson, parseCliJsonl } from "../../cli-runner/helpers.js";

export type ParsedExternalCliOutput = {
  text: string;
  raw: string;
};

export type ExternalCliInvocation = {
  command: string;
  args: string[];
  input?: string;
};

export type ExternalCliAdapter = {
  id: ExternalCliAdapterId;
  command: string;
  buildInvocation(params: {
    task: string;
    model?: string;
    override?: ExternalCliOverrideConfig;
  }): ExternalCliInvocation;
  parseOutput(stdout: string, stderr: string, exitCode: number | null): ParsedExternalCliOutput;
  detect(command?: string): Promise<boolean>;
};

const CLAUDE_BACKEND: CliBackendConfig = {
  command: "claude",
  args: [],
  output: "json",
  input: "arg",
  modelArg: "--model",
  modelAliases: CLAUDE_MODEL_ALIASES,
  sessionIdFields: ["session_id", "sessionId", "conversation_id", "conversationId"],
};

const CODEX_BACKEND: CliBackendConfig = {
  command: "codex",
  args: [],
  output: "jsonl",
  input: "arg",
  modelArg: "--model",
  sessionIdFields: ["thread_id"],
};

const GEMINI_BACKEND: CliBackendConfig = {
  command: "gemini",
  args: [],
  output: "json",
  input: "arg",
  modelArg: "--model",
  sessionIdFields: ["session_id"],
};

const detectCache = new Map<string, Promise<boolean>>();

function uniqueArgs(parts: string[]): string[] {
  const out: string[] = [];
  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) {
      continue;
    }
    out.push(trimmed);
  }
  return out;
}

async function detectBinary(command: string): Promise<boolean> {
  const trimmed = command.trim();
  if (!trimmed) {
    return false;
  }
  const cached = detectCache.get(trimmed);
  if (cached) {
    return await cached;
  }
  const pending = (async () => {
    try {
      const { stdout } = await runExec("which", [trimmed], { timeoutMs: 3_000, maxBuffer: 16_384 });
      return stdout.trim().length > 0;
    } catch {
      return false;
    }
  })();
  detectCache.set(trimmed, pending);
  return await pending;
}

function parseTextFallback(stdout: string, stderr: string): ParsedExternalCliOutput {
  const raw = [stdout, stderr].filter(Boolean).join("\n").trim();
  return {
    text: stdout.trim() || stderr.trim(),
    raw,
  };
}

const adapters: Record<ExternalCliAdapterId, ExternalCliAdapter> = {
  codex: {
    id: "codex",
    command: "codex",
    buildInvocation: ({ task, model, override }) => {
      const command = override?.command?.trim() || "codex";
      const args = uniqueArgs([
        "exec",
        "--json",
        "--color",
        "never",
        "--full-auto",
        "--skip-git-repo-check",
        ...(model ? ["--model", model] : []),
        ...(override?.args ?? []),
        task,
      ]);
      return { command, args };
    },
    parseOutput: (stdout, stderr) => {
      const parsed = parseCliJsonl(stdout, CODEX_BACKEND);
      if (!parsed) {
        return parseTextFallback(stdout, stderr);
      }
      return {
        text: parsed.text,
        raw: stdout.trim(),
      };
    },
    detect: async (command) => await detectBinary(command ?? "codex"),
  },
  claude: {
    id: "claude",
    command: "claude",
    buildInvocation: ({ task, model, override }) => {
      const command = override?.command?.trim() || "claude";
      const normalizedModel =
        model && model.trim() ? normalizeCliModel(model, CLAUDE_BACKEND) : undefined;
      const args = uniqueArgs([
        "-p",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        ...(normalizedModel ? ["--model", normalizedModel] : []),
        ...(override?.args ?? []),
        task,
      ]);
      return { command, args };
    },
    parseOutput: (stdout, stderr) => {
      const parsed = parseCliJson(stdout, CLAUDE_BACKEND);
      if (!parsed) {
        return parseTextFallback(stdout, stderr);
      }
      return {
        text: parsed.text,
        raw: stdout.trim(),
      };
    },
    detect: async (command) => await detectBinary(command ?? "claude"),
  },
  aider: {
    id: "aider",
    command: "aider",
    buildInvocation: ({ task, model, override }) => {
      const command = override?.command?.trim() || "aider";
      const args = uniqueArgs([
        "--message",
        task,
        "--yes",
        ...(model ? ["--model", model] : []),
        ...(override?.args ?? []),
      ]);
      return { command, args };
    },
    parseOutput: (stdout, stderr, exitCode) => {
      const text =
        stdout.trim() || stderr.trim() || `aider exited with code ${exitCode ?? "unknown"}`;
      return {
        text,
        raw: [stdout, stderr].filter(Boolean).join("\n").trim(),
      };
    },
    detect: async (command) => await detectBinary(command ?? "aider"),
  },
  gemini: {
    id: "gemini",
    command: "gemini",
    buildInvocation: ({ task, model, override }) => {
      const command = override?.command?.trim() || "gemini";
      const args = uniqueArgs([
        "-y",
        "--output-format",
        "json",
        ...(model ? ["--model", model] : []),
        ...(override?.args ?? []),
        task,
      ]);
      return { command, args };
    },
    parseOutput: (stdout, stderr) => {
      const parsed = parseCliJson(stdout, GEMINI_BACKEND);
      if (!parsed) {
        return parseTextFallback(stdout, stderr);
      }
      return {
        text: parsed.text,
        raw: stdout.trim(),
      };
    },
    detect: async (command) => await detectBinary(command ?? "gemini"),
  },
};

export function normalizeExternalCliId(value: string | undefined): ExternalCliAdapterId | null {
  const normalized = value?.trim().toLowerCase();
  if (
    normalized === "codex" ||
    normalized === "claude" ||
    normalized === "aider" ||
    normalized === "gemini"
  ) {
    return normalized;
  }
  return null;
}

export function listExternalCliAdapterIds(): ExternalCliAdapterId[] {
  return Object.keys(adapters) as ExternalCliAdapterId[];
}

export function getExternalCliAdapter(id: ExternalCliAdapterId): ExternalCliAdapter {
  return adapters[id];
}
