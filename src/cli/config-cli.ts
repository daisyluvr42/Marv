import type { Command } from "commander";
import JSON5 from "json5";
import { readConfigFileSnapshot, writeConfigFile } from "../core/config/config.js";
import { danger, info, success } from "../globals.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { theme } from "../terminal/theme.js";
import { shortenHomePath } from "../utils.js";
import { formatCliCommand } from "./command-format.js";
import { defineCommandPolicies } from "./command-policy.js";

export const CONFIG_CLI_COMMAND_POLICIES = defineCommandPolicies("config", [
  {
    path: "get",
    cliBootstrap: "skip",
    sideEffect: "none",
    configValidity: "allow-invalid",
  },
  {
    path: "unset",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
  {
    path: "validate",
    cliBootstrap: "skip",
    sideEffect: "none",
    configValidity: "allow-invalid",
  },
]);

type PathSegment = string;

function isIndexSegment(raw: string): boolean {
  return /^[0-9]+$/.test(raw);
}

function parsePath(raw: string): PathSegment[] {
  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }
  const parts: string[] = [];
  let current = "";
  let i = 0;
  while (i < trimmed.length) {
    const ch = trimmed[i];
    if (ch === "\\") {
      const next = trimmed[i + 1];
      if (next) {
        current += next;
      }
      i += 2;
      continue;
    }
    if (ch === ".") {
      if (current) {
        parts.push(current);
      }
      current = "";
      i += 1;
      continue;
    }
    if (ch === "[") {
      if (current) {
        parts.push(current);
      }
      current = "";
      const close = trimmed.indexOf("]", i);
      if (close === -1) {
        throw new Error(`Invalid path (missing "]"): ${raw}`);
      }
      const inside = trimmed.slice(i + 1, close).trim();
      if (!inside) {
        throw new Error(`Invalid path (empty "[]"): ${raw}`);
      }
      parts.push(inside);
      i = close + 1;
      continue;
    }
    current += ch;
    i += 1;
  }
  if (current) {
    parts.push(current);
  }
  return parts.map((part) => part.trim()).filter(Boolean);
}

function parseValue(raw: string, opts: { json?: boolean }): unknown {
  const trimmed = raw.trim();
  if (opts.json) {
    try {
      return JSON5.parse(trimmed);
    } catch (err) {
      throw new Error(`Failed to parse JSON5 value: ${String(err)}`, { cause: err });
    }
  }

  try {
    return JSON5.parse(trimmed);
  } catch {
    return raw;
  }
}

function getAtPath(root: unknown, path: PathSegment[]): { found: boolean; value?: unknown } {
  let current: unknown = root;
  for (const segment of path) {
    if (!current || typeof current !== "object") {
      return { found: false };
    }
    if (Array.isArray(current)) {
      if (!isIndexSegment(segment)) {
        return { found: false };
      }
      const index = Number.parseInt(segment, 10);
      if (!Number.isFinite(index) || index < 0 || index >= current.length) {
        return { found: false };
      }
      current = current[index];
      continue;
    }
    const record = current as Record<string, unknown>;
    if (!(segment in record)) {
      return { found: false };
    }
    current = record[segment];
  }
  return { found: true, value: current };
}

function setAtPath(root: Record<string, unknown>, path: PathSegment[], value: unknown): void {
  let current: unknown = root;
  for (let i = 0; i < path.length - 1; i += 1) {
    const segment = path[i];
    const next = path[i + 1];
    const nextIsIndex = Boolean(next && isIndexSegment(next));
    if (Array.isArray(current)) {
      if (!isIndexSegment(segment)) {
        throw new Error(`Expected numeric index for array segment "${segment}"`);
      }
      const index = Number.parseInt(segment, 10);
      const existing = current[index];
      if (!existing || typeof existing !== "object") {
        current[index] = nextIsIndex ? [] : {};
      }
      current = current[index];
      continue;
    }
    if (!current || typeof current !== "object") {
      throw new Error(`Cannot traverse into "${segment}" (not an object)`);
    }
    const record = current as Record<string, unknown>;
    const existing = record[segment];
    if (!existing || typeof existing !== "object") {
      record[segment] = nextIsIndex ? [] : {};
    }
    current = record[segment];
  }

  const last = path[path.length - 1];
  if (Array.isArray(current)) {
    if (!isIndexSegment(last)) {
      throw new Error(`Expected numeric index for array segment "${last}"`);
    }
    const index = Number.parseInt(last, 10);
    current[index] = value;
    return;
  }
  if (!current || typeof current !== "object") {
    throw new Error(`Cannot set "${last}" (parent is not an object)`);
  }
  (current as Record<string, unknown>)[last] = value;
}

function unsetAtPath(root: Record<string, unknown>, path: PathSegment[]): boolean {
  let current: unknown = root;
  for (let i = 0; i < path.length - 1; i += 1) {
    const segment = path[i];
    if (!current || typeof current !== "object") {
      return false;
    }
    if (Array.isArray(current)) {
      if (!isIndexSegment(segment)) {
        return false;
      }
      const index = Number.parseInt(segment, 10);
      if (!Number.isFinite(index) || index < 0 || index >= current.length) {
        return false;
      }
      current = current[index];
      continue;
    }
    const record = current as Record<string, unknown>;
    if (!(segment in record)) {
      return false;
    }
    current = record[segment];
  }

  const last = path[path.length - 1];
  if (Array.isArray(current)) {
    if (!isIndexSegment(last)) {
      return false;
    }
    const index = Number.parseInt(last, 10);
    if (!Number.isFinite(index) || index < 0 || index >= current.length) {
      return false;
    }
    current.splice(index, 1);
    return true;
  }
  if (!current || typeof current !== "object") {
    return false;
  }
  const record = current as Record<string, unknown>;
  if (!(last in record)) {
    return false;
  }
  delete record[last];
  return true;
}

async function loadConfigSnapshot(
  runtime: RuntimeEnv = defaultRuntime,
  options?: { allowInvalid?: boolean },
) {
  const snapshot = await readConfigFileSnapshot();
  if (snapshot.valid || options?.allowInvalid) {
    return snapshot;
  }
  runtime.error(`Config invalid at ${shortenHomePath(snapshot.path)}.`);
  for (const issue of snapshot.issues) {
    runtime.error(`- ${issue.path || "<root>"}: ${issue.message}`);
  }
  runtime.error(`Run \`${formatCliCommand("marv doctor")}\` to repair, then retry.`);
  runtime.exit(1);
  return snapshot;
}

function parseRequiredPath(path: string): PathSegment[] {
  const parsedPath = parsePath(path);
  if (parsedPath.length === 0) {
    throw new Error("Path is empty.");
  }
  return parsedPath;
}

export async function runConfigGet(opts: { path: string; json?: boolean; runtime?: RuntimeEnv }) {
  const runtime = opts.runtime ?? defaultRuntime;
  try {
    const parsedPath = parseRequiredPath(opts.path);
    const snapshot = await loadConfigSnapshot(runtime, { allowInvalid: true });
    const source =
      snapshot.valid && snapshot.config
        ? snapshot.config
        : typeof snapshot.resolved === "object" && snapshot.resolved
          ? snapshot.resolved
          : snapshot.parsed;
    const res = getAtPath(source, parsedPath);
    if (!res.found) {
      runtime.error(danger(`Config path not found: ${opts.path}`));
      runtime.exit(1);
      return;
    }
    if (!snapshot.valid) {
      runtime.error(
        info(
          `Config is invalid at ${shortenHomePath(snapshot.path)}. Showing the stored value anyway.`,
        ),
      );
    }
    if (opts.json) {
      runtime.log(JSON.stringify(res.value ?? null, null, 2));
      return;
    }
    if (
      typeof res.value === "string" ||
      typeof res.value === "number" ||
      typeof res.value === "boolean"
    ) {
      runtime.log(String(res.value));
      return;
    }
    runtime.log(JSON.stringify(res.value ?? null, null, 2));
  } catch (err) {
    runtime.error(danger(String(err)));
    runtime.exit(1);
  }
}

export async function runConfigUnset(opts: { path: string; runtime?: RuntimeEnv }) {
  const runtime = opts.runtime ?? defaultRuntime;
  try {
    const parsedPath = parseRequiredPath(opts.path);
    const snapshot = await loadConfigSnapshot(runtime);
    // Use snapshot.resolved (config after $include and ${ENV} resolution, but BEFORE runtime defaults)
    // instead of snapshot.config (runtime-merged with defaults).
    // This prevents runtime defaults from leaking into the written config file (issue #6070)
    const next = structuredClone(snapshot.resolved) as Record<string, unknown>;
    const removed = unsetAtPath(next, parsedPath);
    if (!removed) {
      runtime.error(danger(`Config path not found: ${opts.path}`));
      runtime.exit(1);
      return;
    }
    await writeConfigFile(next);
    runtime.log(info(`Removed ${opts.path}. Restart the gateway to apply.`));
  } catch (err) {
    runtime.error(danger(String(err)));
    runtime.exit(1);
  }
}

export async function runConfigValidate(opts: { json?: boolean; runtime?: RuntimeEnv } = {}) {
  const runtime = opts.runtime ?? defaultRuntime;
  try {
    const snapshot = await loadConfigSnapshot(runtime, { allowInvalid: true });

    if (!snapshot.exists) {
      if (opts.json) {
        runtime.log(
          JSON.stringify(
            {
              valid: false,
              path: snapshot.path,
              issues: [{ path: "", message: "config file not found" }],
            },
            null,
            2,
          ),
        );
      } else {
        runtime.error(danger(`Config file not found: ${shortenHomePath(snapshot.path)}`));
      }
      runtime.exit(1);
      return;
    }

    if (!snapshot.valid) {
      if (opts.json) {
        runtime.log(
          JSON.stringify(
            {
              valid: false,
              path: snapshot.path,
              issues: snapshot.issues,
              warnings: snapshot.warnings,
            },
            null,
            2,
          ),
        );
      } else {
        runtime.error(danger(`Config invalid at ${shortenHomePath(snapshot.path)}.`));
        for (const issue of snapshot.issues) {
          runtime.error(`- ${issue.path || "<root>"}: ${issue.message}`);
        }
      }
      runtime.exit(1);
      return;
    }

    if (opts.json) {
      runtime.log(
        JSON.stringify(
          {
            valid: true,
            path: snapshot.path,
            warnings: snapshot.warnings,
          },
          null,
          2,
        ),
      );
      return;
    }

    runtime.log(success(`Config valid: ${shortenHomePath(snapshot.path)}`));
    for (const warning of snapshot.warnings) {
      runtime.log(info(`warning: ${warning.path || "<root>"}: ${warning.message}`));
    }
  } catch (err) {
    if (err instanceof Error && err.message.startsWith("__exit__:")) {
      throw err;
    }
    if (opts.json) {
      runtime.log(
        JSON.stringify(
          {
            valid: false,
            issues: [{ path: "", message: String(err) }],
          },
          null,
          2,
        ),
      );
    } else {
      runtime.error(danger(`Config validation failed: ${String(err)}`));
    }
    runtime.exit(1);
  }
}

export function registerConfigCli(program: Command) {
  const cmd = program
    .command("config")
    .description(
      "Non-interactive config helpers (get/set/unset/validate). Run without subcommand for the setup wizard.",
    )
    .addHelpText(
      "after",
      () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/config", "docs: /cli/config")}\n`,
    )
    .option(
      "--section <section>",
      "Configure wizard sections (repeatable). Use with no subcommand.",
      (value: string, previous: string[]) => [...previous, value],
      [] as string[],
    )
    .action(async (opts) => {
      const { configureCommandFromSectionsArg } = await import("../commands/configure.js");
      await configureCommandFromSectionsArg(opts.section, defaultRuntime);
    });

  cmd
    .command("get")
    .description("Get a config value by dot path")
    .argument("<path>", "Config path (dot or bracket notation)")
    .option("--json", "Output JSON", false)
    .action(async (path: string, opts) => {
      await runConfigGet({ path, json: Boolean(opts.json) });
    });

  cmd
    .command("set")
    .description("Set a config value by dot path")
    .argument("<path>", "Config path (dot or bracket notation)")
    .argument("<value>", "Value (JSON5 or raw string)")
    .option("--json", "Parse value as JSON5 (required)", false)
    .action(async (path: string, value: string, opts) => {
      try {
        const parsedPath = parsePath(path);
        if (parsedPath.length === 0) {
          throw new Error("Path is empty.");
        }
        const parsedValue = parseValue(value, opts);
        const snapshot = await loadConfigSnapshot();
        // Use snapshot.resolved (config after $include and ${ENV} resolution, but BEFORE runtime defaults)
        // instead of snapshot.config (runtime-merged with defaults).
        // This prevents runtime defaults from leaking into the written config file (issue #6070)
        const next = structuredClone(snapshot.resolved) as Record<string, unknown>;
        setAtPath(next, parsedPath, parsedValue);
        await writeConfigFile(next);
        defaultRuntime.log(info(`Updated ${path}. Restart the gateway to apply.`));
      } catch (err) {
        defaultRuntime.error(danger(String(err)));
        defaultRuntime.exit(1);
      }
    });

  cmd
    .command("unset")
    .description("Remove a config value by dot path")
    .argument("<path>", "Config path (dot or bracket notation)")
    .action(async (path: string) => {
      await runConfigUnset({ path });
    });

  cmd
    .command("validate")
    .description("Validate current config without starting the gateway")
    .option("--json", "Output validation result as JSON", false)
    .action(async (opts) => {
      await runConfigValidate({ json: Boolean(opts.json) });
    });
}
