import path from "node:path";

export const DEFAULT_CLI_NAME = "marv";

function getKnownCliNames(): ReadonlySet<string> {
  return new Set(["marv"]);
}

function getDefaultCliName(): string {
  return "marv";
}

function getCliPrefixPattern(): RegExp {
  return /^(?:((?:pnpm|npm|bunx|npx)\s+))?(?:marv|marv)\b/;
}

export function resolveCliName(argv: string[] = process.argv): string {
  const argv1 = argv[1];
  if (!argv1) {
    return getDefaultCliName();
  }
  const base = path.basename(argv1).trim();
  if (getKnownCliNames().has(base)) {
    return base;
  }
  return getDefaultCliName();
}

export function replaceCliName(command: string, cliName = resolveCliName()): string {
  if (!command.trim()) {
    return command;
  }
  const cliPrefixPattern = getCliPrefixPattern();
  if (!cliPrefixPattern.test(command)) {
    return command;
  }
  return command.replace(cliPrefixPattern, (_match, runner: string | undefined) => {
    return `${runner ?? ""}${cliName}`;
  });
}
