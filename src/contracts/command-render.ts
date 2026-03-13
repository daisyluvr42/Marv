import type { CommandArgValues } from "../auto-reply/commands-registry.types.js";

export type CommandRenderContext = {
  commandName: string;
  values: CommandArgValues;
};

export type CommandArgRenderer = (context: CommandRenderContext) => string | undefined;

export function renderCommandArgs(params: {
  commandName: string;
  values: CommandArgValues;
  renderers: Record<string, CommandArgRenderer>;
}): string | undefined {
  const renderer = params.renderers[params.commandName];
  if (!renderer) {
    return undefined;
  }
  return renderer({
    commandName: params.commandName,
    values: params.values,
  });
}
