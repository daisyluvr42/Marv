import type { LegacyConfigRule } from "./legacy.shared.js";

export const LEGACY_CONFIG_RULES: LegacyConfigRule[] = [
  {
    path: ["whatsapp"],
    message: "whatsapp config moved to channels.whatsapp (auto-migrated on load).",
  },
  {
    path: ["telegram"],
    message: "telegram config moved to channels.telegram (auto-migrated on load).",
  },
  {
    path: ["discord"],
    message: "discord config moved to channels.discord (auto-migrated on load).",
  },
  {
    path: ["slack"],
    message: "slack config moved to channels.slack (auto-migrated on load).",
  },
  {
    path: ["signal"],
    message: "signal config moved to channels.signal (auto-migrated on load).",
  },
  {
    path: ["imessage"],
    message: "imessage config moved to channels.imessage (auto-migrated on load).",
  },
  {
    path: ["msteams"],
    message: "msteams config moved to channels.msteams (auto-migrated on load).",
  },
  {
    path: ["routing", "allowFrom"],
    message:
      "routing.allowFrom was removed; use channels.whatsapp.allowFrom instead (auto-migrated on load).",
  },
  {
    path: ["routing", "bindings"],
    message: 'routing.bindings was removed. Routing now targets only "main".',
  },
  {
    path: ["routing", "agents"],
    message:
      'routing.agents was removed. Configure only agents.defaults for "main" and use enhanced subagents for delegation.',
  },
  {
    path: ["routing", "defaultAgentId"],
    message: 'routing.defaultAgentId was removed. The only durable top-level agent id is "main".',
  },
  {
    path: ["routing", "agentToAgent"],
    message:
      "routing.agentToAgent was moved; use tools.agentToAgent instead (auto-migrated on load).",
  },
  {
    path: ["routing", "groupChat", "requireMention"],
    message:
      'routing.groupChat.requireMention was removed; use channels.whatsapp/telegram/imessage groups defaults (e.g. channels.whatsapp.groups."*".requireMention) instead (auto-migrated on load).',
  },
  {
    path: ["routing", "groupChat", "mentionPatterns"],
    message:
      "routing.groupChat.mentionPatterns was removed; use agents.defaults.groupChat.mentionPatterns or messages.groupChat.mentionPatterns instead.",
  },
  {
    path: ["routing", "queue"],
    message: "routing.queue was moved; use messages.queue instead (auto-migrated on load).",
  },
  {
    path: ["routing", "transcribeAudio"],
    message:
      "routing.transcribeAudio was moved; use tools.media.audio.models instead (auto-migrated on load).",
  },
  {
    path: ["telegram", "requireMention"],
    message:
      'telegram.requireMention was removed; use channels.telegram.groups."*".requireMention instead (auto-migrated on load).',
  },
  {
    path: ["identity"],
    message: "identity was moved; use agents.defaults.identity instead.",
  },
  {
    path: ["agent"],
    message:
      "agent.* was moved; use agents.defaults (and tools.* for tool/elevated/exec settings) instead (auto-migrated on load).",
  },
  {
    path: ["memorySearch"],
    message:
      "top-level memorySearch was moved; use agents.defaults.memorySearch instead (auto-migrated on load).",
  },
  {
    path: ["tools", "bash"],
    message: "tools.bash was removed; use tools.exec instead (auto-migrated on load).",
  },
  {
    path: ["agent", "model"],
    message:
      "agent.model string was replaced by agent model pools and models.catalog metadata (legacy import only).",
    match: (value) => typeof value === "string",
  },
  {
    path: ["agent", "imageModel"],
    message:
      "agent.imageModel string was replaced by agents.defaults.imageModel.primary/fallbacks (auto-migrated on load).",
    match: (value) => typeof value === "string",
  },
  {
    path: ["agent", "allowedModels"],
    message: "agent.allowedModels was replaced by agents.defaults.models (auto-migrated on load).",
  },
  {
    path: ["agent", "modelAliases"],
    message:
      "agent.modelAliases was replaced by agents.defaults.models.*.alias (auto-migrated on load).",
  },
  {
    path: ["agent", "modelFallbacks"],
    message:
      "agent.modelFallbacks was removed; use model pool ordering instead (legacy import only).",
  },
  {
    path: ["agent", "imageModelFallbacks"],
    message:
      "agent.imageModelFallbacks was replaced by agents.defaults.imageModel.fallbacks (auto-migrated on load).",
  },
  {
    path: ["messages", "tts", "enabled"],
    message: "messages.tts.enabled was replaced by messages.tts.auto (auto-migrated on load).",
  },
  {
    path: ["gateway", "token"],
    message: "gateway.token is ignored; use gateway.auth.token instead (auto-migrated on load).",
  },
];
