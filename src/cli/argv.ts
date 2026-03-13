const HELP_FLAGS = new Set(["-h", "--help"]);
const VERSION_FLAGS = new Set(["-v", "-V", "--version"]);
const FLAG_TERMINATOR = "--";
const COMMAND_CLI_BOOTSTRAP = new Map<string, CommandCliBootstrap>([
  ["status", "skip"],
  ["health", "skip"],
  ["sessions", "skip"],
  ["logs", "skip"],
  ["config:get", "skip"],
  ["config:unset", "skip"],
  ["config:validate", "skip"],
  ["gateway:status", "skip"],
  ["gateway:probe", "skip"],
  ["gateway:health", "skip"],
  ["gateway:discover", "skip"],
  ["models:list", "skip"],
  ["models:status", "skip"],
  ["memory:status", "skip"],
  ["browser:status", "skip"],
  ["browser:console", "skip"],
  ["browser:errors", "skip"],
  ["browser:requests", "skip"],
  ["daemon:status", "skip"],
  ["update:status", "skip"],
  ["system:presence", "skip"],
  ["system:heartbeat:last", "skip"],
]);
const COMMAND_SIDE_EFFECTS = new Map<string, CommandSideEffect>([
  ["logs", "none"],
  ["status", "none"],
  ["health", "none"],
  ["sessions", "none"],
  ["config:get", "none"],
  ["config:validate", "none"],
  ["config:unset", "none"],
  ["gateway:status", "none"],
  ["gateway:probe", "none"],
  ["gateway:health", "none"],
  ["gateway:discover", "none"],
  ["models:list", "none"],
  ["models:status", "none"],
  ["memory:status", "none"],
  ["browser:status", "none"],
  ["browser:console", "none"],
  ["browser:errors", "none"],
  ["browser:requests", "none"],
  ["daemon:status", "none"],
  ["update:status", "none"],
  ["system:presence", "none"],
  ["system:heartbeat:last", "none"],
  ["agent", "none"],
]);
const COMMAND_CONFIG_VALIDITY = new Map<string, CommandConfigValidity>([
  ["doctor", "allow-invalid"],
  ["logs", "allow-invalid"],
  ["health", "allow-invalid"],
  ["help", "allow-invalid"],
  ["status", "allow-invalid"],
  ["config:get", "allow-invalid"],
  ["config:validate", "allow-invalid"],
  ["gateway:status", "allow-invalid"],
  ["gateway:probe", "allow-invalid"],
  ["gateway:health", "allow-invalid"],
  ["gateway:discover", "allow-invalid"],
  ["gateway:call", "allow-invalid"],
  ["gateway:install", "allow-invalid"],
  ["gateway:uninstall", "allow-invalid"],
  ["gateway:start", "allow-invalid"],
  ["gateway:stop", "allow-invalid"],
  ["gateway:restart", "allow-invalid"],
]);

export type CommandCliBootstrap = "skip" | "require";
export type CommandSideEffect = "none" | "state-migrate";
export type CommandConfigValidity = "require-valid" | "allow-invalid";

export function hasHelpOrVersion(argv: string[]): boolean {
  return argv.some((arg) => HELP_FLAGS.has(arg) || VERSION_FLAGS.has(arg));
}

function isValueToken(arg: string | undefined): boolean {
  if (!arg) {
    return false;
  }
  if (arg === FLAG_TERMINATOR) {
    return false;
  }
  if (!arg.startsWith("-")) {
    return true;
  }
  return /^-\d+(?:\.\d+)?$/.test(arg);
}

function parsePositiveInt(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed) || parsed <= 0) {
    return undefined;
  }
  return parsed;
}

export function hasFlag(argv: string[], name: string): boolean {
  const args = argv.slice(2);
  for (const arg of args) {
    if (arg === FLAG_TERMINATOR) {
      break;
    }
    if (arg === name) {
      return true;
    }
  }
  return false;
}

export function getFlagValue(argv: string[], name: string): string | null | undefined {
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === FLAG_TERMINATOR) {
      break;
    }
    if (arg === name) {
      const next = args[i + 1];
      return isValueToken(next) ? next : null;
    }
    if (arg.startsWith(`${name}=`)) {
      const value = arg.slice(name.length + 1);
      return value ? value : null;
    }
  }
  return undefined;
}

export function getVerboseFlag(argv: string[], options?: { includeDebug?: boolean }): boolean {
  if (hasFlag(argv, "--verbose")) {
    return true;
  }
  if (options?.includeDebug && hasFlag(argv, "--debug")) {
    return true;
  }
  return false;
}

export function getPositiveIntFlagValue(argv: string[], name: string): number | null | undefined {
  const raw = getFlagValue(argv, name);
  if (raw === null || raw === undefined) {
    return raw;
  }
  return parsePositiveInt(raw);
}

export function getCommandPath(argv: string[], depth = 2): string[] {
  const args = argv.slice(2);
  const path: string[] = [];
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (!arg) {
      continue;
    }
    if (arg === "--") {
      break;
    }
    if (arg.startsWith("-")) {
      continue;
    }
    path.push(arg);
    if (path.length >= depth) {
      break;
    }
  }
  return path;
}

export function getPrimaryCommand(argv: string[]): string | null {
  const [primary] = getCommandPath(argv, 1);
  return primary ?? null;
}

export function buildParseArgv(params: {
  programName?: string;
  rawArgs?: string[];
  fallbackArgv?: string[];
}): string[] {
  const baseArgv =
    params.rawArgs && params.rawArgs.length > 0
      ? params.rawArgs
      : params.fallbackArgv && params.fallbackArgv.length > 0
        ? params.fallbackArgv
        : process.argv;
  const programName = params.programName ?? "";
  const normalizedArgv =
    programName && baseArgv[0] === programName
      ? baseArgv.slice(1)
      : baseArgv[0]?.endsWith("marv")
        ? baseArgv.slice(1)
        : baseArgv;
  const executable = (normalizedArgv[0]?.split(/[/\\]/).pop() ?? "").toLowerCase();
  const looksLikeNode =
    normalizedArgv.length >= 2 && (isNodeExecutable(executable) || isBunExecutable(executable));
  if (looksLikeNode) {
    return normalizedArgv;
  }
  return ["node", programName || "marv", ...normalizedArgv];
}

const nodeExecutablePattern = /^node-\d+(?:\.\d+)*(?:\.exe)?$/;

function isNodeExecutable(executable: string): boolean {
  return (
    executable === "node" ||
    executable === "node.exe" ||
    executable === "nodejs" ||
    executable === "nodejs.exe" ||
    nodeExecutablePattern.test(executable)
  );
}

function isBunExecutable(executable: string): boolean {
  return executable === "bun" || executable === "bun.exe";
}

export function shouldMigrateStateFromPath(path: string[]): boolean {
  return resolveCommandSideEffectFromPath(path) === "state-migrate";
}

export function shouldMigrateState(argv: string[]): boolean {
  return shouldMigrateStateFromPath(getCommandPath(argv, 3));
}

export function resolveCommandSideEffectFromPath(path: string[]): CommandSideEffect {
  if (path.length === 0) {
    return "state-migrate";
  }
  for (const key of getCommandPolicyKeys(path)) {
    const policy = COMMAND_SIDE_EFFECTS.get(key);
    if (policy) {
      return policy;
    }
  }
  return "state-migrate";
}

export function resolveCommandConfigValidityFromPath(path: string[]): CommandConfigValidity {
  if (path.length === 0) {
    return "require-valid";
  }
  for (const key of getCommandPolicyKeys(path)) {
    const policy = COMMAND_CONFIG_VALIDITY.get(key);
    if (policy) {
      return policy;
    }
  }
  return "require-valid";
}

export function resolveCommandCliBootstrapFromPath(path: string[]): CommandCliBootstrap {
  if (path.length === 0) {
    return "require";
  }
  for (const key of getCommandPolicyKeys(path)) {
    const policy = COMMAND_CLI_BOOTSTRAP.get(key);
    if (policy) {
      return policy;
    }
  }
  return "require";
}

function getCommandPolicyKeys(path: string[]): string[] {
  const trimmed = path.map((segment) => segment.trim()).filter(Boolean);
  const keys: string[] = [];
  for (let length = trimmed.length; length >= 1; length -= 1) {
    keys.push(trimmed.slice(0, length).join(":"));
  }
  return keys;
}
