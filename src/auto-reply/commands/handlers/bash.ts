import { logVerbose } from "../../../globals.js";
import { handleBashChatCommand } from "../../delivery/bash-command.js";
import type { CommandRegistry } from "../command-registry.js";
import type { CommandHandler } from "../handler-types.js";

export const handleBashCommand: CommandHandler = async (params, allowTextCommands) => {
  if (!allowTextCommands) {
    return null;
  }
  const { command } = params;
  const bashSlashRequested =
    command.commandBodyNormalized === "/bash" || command.commandBodyNormalized.startsWith("/bash ");
  const bashBangRequested = command.commandBodyNormalized.startsWith("!");
  if (!bashSlashRequested && !(bashBangRequested && command.isAuthorizedSender)) {
    return null;
  }
  if (!command.isAuthorizedSender) {
    logVerbose(`Ignoring /bash from unauthorized sender: ${command.senderId || "<unknown>"}`);
    return { shouldContinue: false };
  }
  const reply = await handleBashChatCommand({
    ctx: params.ctx,
    cfg: params.cfg,
    agentId: params.agentId,
    sessionKey: params.sessionKey,
    isGroup: params.isGroup,
    elevated: params.elevated,
  });
  return { shouldContinue: false, reply };
};

export function registerBashCommands(registry: CommandRegistry): void {
  registry.register("bash", handleBashCommand);
}
