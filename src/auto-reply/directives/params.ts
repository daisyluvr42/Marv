import type { RuntimeConfiguredModel } from "../../agents/model/model-pool.js";
import type { ModelAliasIndex } from "../../agents/model/model-resolve.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { SessionEntry } from "../../core/config/sessions.js";
import type { TurnContext } from "../support/templating.js";
import type { ElevatedLevel, ReasoningLevel, ThinkLevel, VerboseLevel } from "./directives.js";
import type { InlineDirectives } from "./parse.js";

export type HandleDirectiveOnlyCoreParams = {
  cfg: MarvConfig;
  directives: InlineDirectives;
  sessionEntry: SessionEntry;
  sessionStore: Record<string, SessionEntry>;
  sessionKey: string;
  storePath?: string;
  elevatedEnabled: boolean;
  elevatedAllowed: boolean;
  elevatedFailures?: Array<{ gate: string; key: string }>;
  messageProviderKey?: string;
  defaultProvider: string;
  defaultModel: string;
  aliasIndex: ModelAliasIndex;
  allowedModelKeys: Set<string>;
  allowedModelCatalog: Awaited<
    ReturnType<typeof import("../../agents/model/model-catalog.js").loadModelCatalog>
  >;
  poolName: string;
  candidates: RuntimeConfiguredModel[];
  resetModelOverride: boolean;
  provider: string;
  model: string;
  initialModelLabel: string;
  formatModelSwitchEvent: (label: string, alias?: string) => string;
};

export type HandleDirectiveOnlyParams = HandleDirectiveOnlyCoreParams & {
  currentThinkLevel?: ThinkLevel;
  currentVerboseLevel?: VerboseLevel;
  currentReasoningLevel?: ReasoningLevel;
  currentElevatedLevel?: ElevatedLevel;
  surface?: string;
};

export type ApplyInlineDirectivesFastLaneParams = HandleDirectiveOnlyCoreParams & {
  commandAuthorized: boolean;
  ctx: TurnContext;
  agentId?: string;
  isGroup: boolean;
  agentCfg?: NonNullable<MarvConfig["agents"]>["defaults"];
  modelState: {
    poolName: string;
    candidates: RuntimeConfiguredModel[];
    resolveDefaultThinkingLevel: () => Promise<ThinkLevel | undefined>;
    allowedModelKeys: Set<string>;
    allowedModelCatalog: Awaited<
      ReturnType<typeof import("../../agents/model/model-catalog.js").loadModelCatalog>
    >;
    resetModelOverride: boolean;
  };
};
