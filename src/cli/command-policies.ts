import { defineCommandPolicies } from "./command-policy.js";
import { CONFIG_CLI_COMMAND_POLICIES } from "./config-cli.js";
import { defineGatewayServiceCommandPolicies } from "./daemon-cli/register-service-commands.js";
import { LOGS_CLI_COMMAND_POLICIES } from "./logs-cli.js";
import { MEMORY_CLI_COMMAND_POLICIES } from "./memory-cli.js";
import { MODELS_CLI_COMMAND_POLICIES } from "./models-cli.js";
import { SYSTEM_CLI_COMMAND_POLICIES } from "./system-cli.js";
import { UPDATE_CLI_COMMAND_POLICIES } from "./update-cli.js";

const STATUS_HEALTH_SESSIONS_COMMAND_POLICIES = defineCommandPolicies("", [
  {
    path: "status",
    cliBootstrap: "skip",
    sideEffect: "none",
    configValidity: "allow-invalid",
  },
  {
    path: "health",
    cliBootstrap: "skip",
    sideEffect: "none",
    configValidity: "allow-invalid",
  },
  {
    path: "sessions",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
]);

const BROWSER_MANAGE_COMMAND_POLICIES = defineCommandPolicies("browser", [
  {
    path: "status",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
]);

const BROWSER_OBSERVE_COMMAND_POLICIES = defineCommandPolicies("browser", [
  {
    path: "console",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
]);

const BROWSER_DEBUG_COMMAND_POLICIES = defineCommandPolicies("browser", [
  {
    path: "errors",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
  {
    path: "requests",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
]);

const BROWSER_CLI_COMMAND_POLICIES = [
  ...BROWSER_MANAGE_COMMAND_POLICIES,
  ...BROWSER_OBSERVE_COMMAND_POLICIES,
  ...BROWSER_DEBUG_COMMAND_POLICIES,
];

const GATEWAY_CLI_COMMAND_POLICIES = [
  ...defineGatewayServiceCommandPolicies("gateway"),
  ...defineCommandPolicies("gateway", [
    {
      path: "probe",
      cliBootstrap: "skip",
      sideEffect: "none",
      configValidity: "allow-invalid",
    },
    {
      path: "health",
      cliBootstrap: "skip",
      sideEffect: "none",
      configValidity: "allow-invalid",
    },
    {
      path: "discover",
      cliBootstrap: "skip",
      sideEffect: "none",
      configValidity: "allow-invalid",
    },
    {
      path: "call",
      configValidity: "allow-invalid",
    },
  ]),
];

const DAEMON_CLI_COMMAND_POLICIES = defineGatewayServiceCommandPolicies("daemon");

const AGENT_CLI_COMMAND_POLICIES = defineCommandPolicies("", [
  {
    path: "agent",
    sideEffect: "none",
  },
]);

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
