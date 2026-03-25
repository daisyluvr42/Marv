export { createSessionSettingsTool } from "./self-settings-session.js";
export { createMemorySettingsTool } from "./self-settings-memory.js";
export { createHeartbeatSettingsTool } from "./self-settings-heartbeat.js";
export { createConfigManageTool } from "./self-settings-config.js";
export type { SelfSettingsToolOpts } from "./self-settings-normalize.js";

// Backward-compatible alias: maps to the session settings tool for existing callers.
export { createSessionSettingsTool as createSelfSettingsTool } from "./self-settings-session.js";
