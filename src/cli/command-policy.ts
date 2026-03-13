export type CommandCliBootstrap = "skip" | "require";
export type CommandSideEffect = "none" | "state-migrate";
export type CommandConfigValidity = "require-valid" | "allow-invalid";

export type CommandPolicyDefinition = {
  path: string;
  cliBootstrap?: CommandCliBootstrap;
  sideEffect?: CommandSideEffect;
  configValidity?: CommandConfigValidity;
};

export type DefinedCommandPolicy = CommandPolicyDefinition & {
  key: string;
};

export function defineCommandPolicies(
  namespace: string,
  definitions: CommandPolicyDefinition[],
): DefinedCommandPolicy[] {
  return definitions.map((definition) => ({
    ...definition,
    key: [namespace, definition.path].filter(Boolean).join(":"),
  }));
}

export function selectCommandPolicyEntries<T extends keyof CommandPolicyDefinition>(
  definitions: DefinedCommandPolicy[],
  field: T,
): Array<[string, NonNullable<CommandPolicyDefinition[T]>]> {
  const entries: Array<[string, NonNullable<CommandPolicyDefinition[T]>]> = [];
  for (const definition of definitions) {
    const value = definition[field];
    if (value !== undefined) {
      entries.push([definition.key, value as NonNullable<CommandPolicyDefinition[T]>]);
    }
  }
  return entries;
}
