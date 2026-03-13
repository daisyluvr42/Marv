import { BROWSER_CLI_COMMAND_POLICIES } from "./browser-cli.js";
import { CONFIG_CLI_COMMAND_POLICIES } from "./config-cli.js";
import { DAEMON_CLI_COMMAND_POLICIES } from "./daemon-cli/register.js";
import { GATEWAY_CLI_COMMAND_POLICIES } from "./gateway-cli/register.js";
import { LOGS_CLI_COMMAND_POLICIES } from "./logs-cli.js";
import { MEMORY_CLI_COMMAND_POLICIES } from "./memory-cli.js";
import { MODELS_CLI_COMMAND_POLICIES } from "./models-cli.js";
import { AGENT_CLI_COMMAND_POLICIES } from "./program/register.agent.js";
import { STATUS_HEALTH_SESSIONS_COMMAND_POLICIES } from "./program/register.status-health-sessions.js";
import { SYSTEM_CLI_COMMAND_POLICIES } from "./system-cli.js";
import { UPDATE_CLI_COMMAND_POLICIES } from "./update-cli.js";

export const CLI_DEFINITION_COMMAND_POLICIES = [
  ...STATUS_HEALTH_SESSIONS_COMMAND_POLICIES,
  ...BROWSER_CLI_COMMAND_POLICIES,
  ...CONFIG_CLI_COMMAND_POLICIES,
  ...LOGS_CLI_COMMAND_POLICIES,
  ...MODELS_CLI_COMMAND_POLICIES,
  ...MEMORY_CLI_COMMAND_POLICIES,
  ...GATEWAY_CLI_COMMAND_POLICIES,
  ...DAEMON_CLI_COMMAND_POLICIES,
  ...UPDATE_CLI_COMMAND_POLICIES,
  ...SYSTEM_CLI_COMMAND_POLICIES,
  ...AGENT_CLI_COMMAND_POLICIES,
];
