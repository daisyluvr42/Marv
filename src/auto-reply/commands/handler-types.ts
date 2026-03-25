import type { SkillCommandSpec } from "../../agents/skills.js";
import type { ChannelId } from "../../channels/plugins/types.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { SessionEntry, SessionScope } from "../../core/config/sessions.js";
import type { GroupMemberRole } from "../../core/config/types.messages.js";
import type { InlineDirectives } from "../directives/parse.js";
import type { TurnContext } from "../support/templating.js";
import type {
  ElevatedLevel,
  ReasoningLevel,
  ThinkLevel,
  VerboseLevel,
} from "../support/thinking.js";
import type { ReplyPayload } from "../support/types.js";

export type CommandContext = {
  surface: string;
  channel: string;
  channelId?: ChannelId;
  ownerList: string[];
  senderIsOwner: boolean;
  isAuthorizedSender: boolean;
  /** Group member role — "owner" for bot owner, "member" for everyone else. */
  senderRole: GroupMemberRole;
  senderId?: string;
  abortKey?: string;
  rawBodyNormalized: string;
  commandBodyNormalized: string;
  from?: string;
  to?: string;
};

export type HandleCommandsParams = {
  ctx: TurnContext;
  cfg: MarvConfig;
  command: CommandContext;
  agentId?: string;
  directives: InlineDirectives;
  elevated: {
    enabled: boolean;
    allowed: boolean;
    failures: Array<{ gate: string; key: string }>;
  };
  sessionEntry?: SessionEntry;
  previousSessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey: string;
  storePath?: string;
  sessionScope?: SessionScope;
  workspaceDir: string;
  defaultGroupActivation: () => "always" | "mention" | "smart";
  resolvedThinkLevel?: ThinkLevel;
  resolvedVerboseLevel: VerboseLevel;
  resolvedReasoningLevel: ReasoningLevel;
  resolvedElevatedLevel?: ElevatedLevel;
  resolveDefaultThinkingLevel: () => Promise<ThinkLevel | undefined>;
  provider: string;
  model: string;
  contextTokens: number;
  isGroup: boolean;
  skillCommands?: SkillCommandSpec[];
};

export type CommandHandlerResult = {
  reply?: ReplyPayload;
  shouldContinue: boolean;
};

export type CommandHandler = (
  params: HandleCommandsParams,
  allowTextCommands: boolean,
) => Promise<CommandHandlerResult | null>;
