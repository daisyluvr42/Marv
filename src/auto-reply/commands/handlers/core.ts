import fs from "node:fs/promises";
import { resolveSendPolicy } from "../../../core/session/send-policy.js";
import { logVerbose } from "../../../globals.js";
import { getGlobalHookRunner } from "../../../plugins/hook-runner-global.js";
import { CommandRegistry } from "../command-registry.js";
import type { CommandHandlerResult, HandleCommandsParams } from "../handler-types.js";
import { shouldHandleTextCommands } from "../registry.js";
import { registerAllowlistCommands } from "./allowlist.js";
import { registerApproveCommands } from "./approve.js";
import { registerBashCommands } from "./bash.js";
import { registerCompactCommands } from "./compact.js";
import { registerConfigCommands } from "./config.js";
import { registerInfoCommands } from "./info.js";
import { registerModelsCommands } from "./models.js";
import { registerPluginCommands } from "./plugin.js";
import { registerRevertCommands } from "./revert.js";
import { registerSessionCommands } from "./session.js";
import { registerSubagentsCommands } from "./subagents.js";
import { registerTtsCommands } from "./tts.js";

let registry: CommandRegistry | null = null;

function getCommandRegistry(): CommandRegistry {
  if (registry) {
    return registry;
  }
  registry = new CommandRegistry();
  // Registration order determines dispatch priority.
  // Plugin commands are processed first, before built-in commands.
  registerPluginCommands(registry);
  registerBashCommands(registry);
  registerSessionCommands(registry);
  registerRevertCommands(registry);
  registerTtsCommands(registry);
  registerInfoCommands(registry);
  registerAllowlistCommands(registry);
  registerApproveCommands(registry);
  registerSubagentsCommands(registry);
  registerConfigCommands(registry);
  registerModelsCommands(registry);
  registerCompactCommands(registry);
  return registry;
}

export { CommandRegistry };

export async function handleCommands(params: HandleCommandsParams): Promise<CommandHandlerResult> {
  const reg = getCommandRegistry();

  const resetMatch = params.command.commandBodyNormalized.match(/^\/(new|reset)(?:\s|$)/);
  const resetRequested = Boolean(resetMatch);
  if (resetRequested && !params.command.isAuthorizedSender) {
    logVerbose(
      `Ignoring /reset from unauthorized sender: ${params.command.senderId || "<unknown>"}`,
    );
    return { shouldContinue: false };
  }

  if (resetRequested && params.command.isAuthorizedSender) {
    const commandAction = resetMatch?.[1] ?? "new";
    // Fire before_reset plugin hook — extract memories before session history is lost
    const hookRunner = getGlobalHookRunner();
    if (hookRunner?.hasHooks("before_reset")) {
      const prevEntry = params.previousSessionEntry;
      const sessionFile = prevEntry?.sessionFile;
      // Fire-and-forget: read old session messages and run hook
      void (async () => {
        try {
          const messages: unknown[] = [];
          if (sessionFile) {
            const content = await fs.readFile(sessionFile, "utf-8");
            for (const line of content.split("\n")) {
              if (!line.trim()) {
                continue;
              }
              try {
                const entry = JSON.parse(line);
                if (entry.type === "message" && entry.message) {
                  messages.push(entry.message);
                }
              } catch {
                // skip malformed lines
              }
            }
          } else {
            logVerbose("before_reset: no session file available, firing hook with empty messages");
          }
          await hookRunner.runBeforeReset(
            { sessionFile, messages, reason: commandAction },
            {
              agentId: params.sessionKey?.split(":")[0] ?? "main",
              sessionKey: params.sessionKey,
              sessionId: prevEntry?.sessionId,
              workspaceDir: params.workspaceDir,
            },
          );
        } catch (err: unknown) {
          logVerbose(`before_reset hook failed: ${String(err)}`);
        }
      })();
    }
  }

  const allowTextCommands = shouldHandleTextCommands({
    cfg: params.cfg,
    surface: params.command.surface,
    commandSource: params.ctx.CommandSource,
  });

  const result = await reg.dispatch(params, allowTextCommands);
  if (result) {
    return result;
  }

  const sendPolicy = resolveSendPolicy({
    cfg: params.cfg,
    entry: params.sessionEntry,
    sessionKey: params.sessionKey,
    channel: params.sessionEntry?.channel ?? params.command.channel,
    chatType: params.sessionEntry?.chatType,
  });
  if (sendPolicy === "deny") {
    logVerbose(`Send blocked by policy for session ${params.sessionKey ?? "unknown"}`);
    return { shouldContinue: false };
  }

  return { shouldContinue: true };
}
