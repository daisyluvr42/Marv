import {
  ensureRecord,
  getRecord,
  type LegacyConfigMigration,
  mapLegacyAudioTranscription,
} from "./legacy.shared.js";

function applyLegacyAudioTranscriptionModel(params: {
  raw: Record<string, unknown>;
  source: unknown;
  changes: string[];
  movedMessage: string;
  alreadySetMessage: string;
  invalidMessage: string;
}) {
  const mapped = mapLegacyAudioTranscription(params.source);
  if (!mapped) {
    params.changes.push(params.invalidMessage);
    return;
  }
  const tools = ensureRecord(params.raw, "tools");
  const media = ensureRecord(tools, "media");
  const mediaAudio = ensureRecord(media, "audio");
  const models = Array.isArray(mediaAudio.models) ? (mediaAudio.models as unknown[]) : [];
  if (models.length === 0) {
    mediaAudio.enabled = true;
    mediaAudio.models = [mapped];
    params.changes.push(params.movedMessage);
    return;
  }
  params.changes.push(params.alreadySetMessage);
}

export const LEGACY_CONFIG_MIGRATIONS_PART_2: LegacyConfigMigration[] = [
  {
    id: "agent.model-config-v2",
    describe:
      "Migrate legacy agent.model/allowedModels/modelAliases/modelFallbacks/imageModelFallbacks to agent.models + model lists",
    apply: (raw, changes) => {
      const agentRoot = getRecord(raw.agent);
      const defaults = getRecord(getRecord(raw.agents)?.defaults);
      const agent = agentRoot ?? defaults;
      if (!agent) {
        return;
      }
      const label = agentRoot ? "agent" : "agents.defaults";

      const legacyModel = typeof agent.model === "string" ? String(agent.model) : undefined;
      const legacyImageModel =
        typeof agent.imageModel === "string" ? String(agent.imageModel) : undefined;
      const legacyModelFallbacks = Array.isArray(agent.modelFallbacks)
        ? (agent.modelFallbacks as unknown[]).map(String)
        : [];
      const legacyImageModelFallbacks = Array.isArray(agent.imageModelFallbacks)
        ? (agent.imageModelFallbacks as unknown[]).map(String)
        : [];
      const legacyAliases =
        agent.modelAliases && typeof agent.modelAliases === "object"
          ? (agent.modelAliases as Record<string, unknown>)
          : {};

      const hasLegacy =
        legacyModel ||
        legacyImageModel ||
        legacyModelFallbacks.length > 0 ||
        legacyImageModelFallbacks.length > 0 ||
        Object.keys(legacyAliases).length > 0;
      if (!hasLegacy) {
        return;
      }

      const rootModels = ensureRecord(raw, "models");
      const metadata =
        rootModels.metadata && typeof rootModels.metadata === "object"
          ? (rootModels.metadata as Record<string, unknown>)
          : {};
      rootModels.metadata = metadata;

      const ensureModel = (rawKey?: string) => {
        if (typeof rawKey !== "string") {
          return;
        }
        const key = rawKey.trim();
        if (!key) {
          return;
        }
        if (!metadata[key]) {
          metadata[key] = {};
        }
      };

      ensureModel(legacyModel);
      ensureModel(legacyImageModel);
      for (const key of legacyModelFallbacks) {
        ensureModel(key);
      }
      for (const key of legacyImageModelFallbacks) {
        ensureModel(key);
      }
      for (const target of Object.values(legacyAliases)) {
        if (typeof target !== "string") {
          continue;
        }
        ensureModel(target);
      }

      for (const [alias, targetRaw] of Object.entries(legacyAliases)) {
        if (typeof targetRaw !== "string") {
          continue;
        }
        const target = targetRaw.trim();
        if (!target) {
          continue;
        }
        const entry =
          metadata[target] && typeof metadata[target] === "object"
            ? (metadata[target] as Record<string, unknown>)
            : {};
        if (!("alias" in entry)) {
          entry.alias = alias;
          metadata[target] = entry;
        }
      }

      const currentModel =
        agent.model && typeof agent.model === "object"
          ? (agent.model as Record<string, unknown>)
          : null;
      if (currentModel) {
        if (!currentModel.primary && legacyModel) {
          currentModel.primary = legacyModel;
        }
        if (
          legacyModelFallbacks.length > 0 &&
          (!Array.isArray(currentModel.fallbacks) || currentModel.fallbacks.length === 0)
        ) {
          currentModel.fallbacks = legacyModelFallbacks;
        }
        agent.model = currentModel;
      } else if (legacyModel || legacyModelFallbacks.length > 0) {
        agent.model = {
          primary: legacyModel,
          fallbacks: legacyModelFallbacks.length ? legacyModelFallbacks : [],
        };
      }

      const currentImageModel =
        agent.imageModel && typeof agent.imageModel === "object"
          ? (agent.imageModel as Record<string, unknown>)
          : null;
      if (currentImageModel) {
        if (!currentImageModel.primary && legacyImageModel) {
          currentImageModel.primary = legacyImageModel;
        }
        if (
          legacyImageModelFallbacks.length > 0 &&
          (!Array.isArray(currentImageModel.fallbacks) || currentImageModel.fallbacks.length === 0)
        ) {
          currentImageModel.fallbacks = legacyImageModelFallbacks;
        }
        agent.imageModel = currentImageModel;
      } else if (legacyImageModel || legacyImageModelFallbacks.length > 0) {
        agent.imageModel = {
          primary: legacyImageModel,
          fallbacks: legacyImageModelFallbacks.length ? legacyImageModelFallbacks : [],
        };
      }

      if (legacyModel !== undefined) {
        changes.push(`Migrated ${label}.model string → ${label}.model.primary.`);
      }
      if (legacyModelFallbacks.length > 0) {
        changes.push(`Migrated ${label}.modelFallbacks → ${label}.model.fallbacks.`);
      }
      if (legacyImageModel !== undefined) {
        changes.push(`Migrated ${label}.imageModel string → ${label}.imageModel.primary.`);
      }
      if (legacyImageModelFallbacks.length > 0) {
        changes.push(`Migrated ${label}.imageModelFallbacks → ${label}.imageModel.fallbacks.`);
      }
      if (Object.keys(legacyAliases).length > 0) {
        changes.push("Migrated agent.modelAliases → models.metadata.*.alias.");
      }

      delete agent.allowedModels;
      delete agent.modelAliases;
      delete agent.modelFallbacks;
      delete agent.imageModelFallbacks;
    },
  },
  {
    id: "routing.config-v2",
    describe: "Move routing groupChat/queue/agentToAgent/transcribeAudio",
    apply: (raw, changes) => {
      const routing = getRecord(raw.routing);
      if (!routing) {
        return;
      }

      if (routing.agentToAgent !== undefined) {
        const tools = ensureRecord(raw, "tools");
        if (tools.agentToAgent === undefined) {
          tools.agentToAgent = routing.agentToAgent;
          changes.push("Moved routing.agentToAgent → tools.agentToAgent.");
        } else {
          changes.push("Removed routing.agentToAgent (tools.agentToAgent already set).");
        }
        delete routing.agentToAgent;
      }

      if (routing.queue !== undefined) {
        const messages = ensureRecord(raw, "messages");
        if (messages.queue === undefined) {
          messages.queue = routing.queue;
          changes.push("Moved routing.queue → messages.queue.");
        } else {
          changes.push("Removed routing.queue (messages.queue already set).");
        }
        delete routing.queue;
      }

      const groupChat = getRecord(routing.groupChat);
      if (groupChat) {
        const historyLimit = groupChat.historyLimit;
        if (historyLimit !== undefined) {
          const messages = ensureRecord(raw, "messages");
          const messagesGroup = ensureRecord(messages, "groupChat");
          if (messagesGroup.historyLimit === undefined) {
            messagesGroup.historyLimit = historyLimit;
            changes.push("Moved routing.groupChat.historyLimit → messages.groupChat.historyLimit.");
          } else {
            changes.push(
              "Removed routing.groupChat.historyLimit (messages.groupChat.historyLimit already set).",
            );
          }
          delete groupChat.historyLimit;
        }

        const mentionPatterns = groupChat.mentionPatterns;
        if (mentionPatterns !== undefined) {
          const messages = ensureRecord(raw, "messages");
          const messagesGroup = ensureRecord(messages, "groupChat");
          if (messagesGroup.mentionPatterns === undefined) {
            messagesGroup.mentionPatterns = mentionPatterns;
            changes.push(
              "Moved routing.groupChat.mentionPatterns → messages.groupChat.mentionPatterns.",
            );
          } else {
            changes.push(
              "Removed routing.groupChat.mentionPatterns (messages.groupChat.mentionPatterns already set).",
            );
          }
          delete groupChat.mentionPatterns;
        }

        if (Object.keys(groupChat).length === 0) {
          delete routing.groupChat;
        } else {
          routing.groupChat = groupChat;
        }
      }

      if (routing.transcribeAudio !== undefined) {
        applyLegacyAudioTranscriptionModel({
          raw,
          source: routing.transcribeAudio,
          changes,
          movedMessage: "Moved routing.transcribeAudio → tools.media.audio.models.",
          alreadySetMessage:
            "Removed routing.transcribeAudio (tools.media.audio.models already set).",
          invalidMessage: "Removed routing.transcribeAudio (invalid or empty command).",
        });
        delete routing.transcribeAudio;
      }

      if (Object.keys(routing).length === 0) {
        delete raw.routing;
      }
    },
  },
  {
    id: "audio.transcription-v2",
    describe: "Move audio.transcription to tools.media.audio.models",
    apply: (raw, changes) => {
      const audio = getRecord(raw.audio);
      if (audio?.transcription === undefined) {
        return;
      }

      applyLegacyAudioTranscriptionModel({
        raw,
        source: audio.transcription,
        changes,
        movedMessage: "Moved audio.transcription → tools.media.audio.models.",
        alreadySetMessage: "Removed audio.transcription (tools.media.audio.models already set).",
        invalidMessage: "Removed audio.transcription (invalid or empty command).",
      });
      delete audio.transcription;
      if (Object.keys(audio).length === 0) {
        delete raw.audio;
      } else {
        raw.audio = audio;
      }
    },
  },
];
