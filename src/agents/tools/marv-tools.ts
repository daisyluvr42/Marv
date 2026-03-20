import type { MarvConfig } from "../../core/config/config.js";
import { resolvePluginTools } from "../../plugins/tools.js";
import type { GatewayMessageChannel } from "../../utils/message-channel.js";
import { resolveSessionAgentId } from "../agent-scope.js";
import type { SandboxFsBridge } from "../sandbox/fs-bridge.js";
import { resolveWorkspaceRoot } from "../workspace-dir.js";
import { createAgentsListTool } from "./agents-list-tool.js";
import { createBrowserTool } from "./browser-tool.js";
import { createCanvasTool } from "./canvas-tool.js";
import { createCliInvokeTool } from "./cli-invoke-tool.js";
import { createCliProfilesTool } from "./cli-profiles-tool.js";
import { createCliSynthesizeTool } from "./cli-synthesize-tool.js";
// cli_verify no longer registered as an agent tool (cli_synthesize auto-verifies).
import type { AnyAgentTool } from "./common.js";
import { createCronTool } from "./cron-tool.js";
import { createExternalCliTool } from "./external-cli-tool.js";
import { createGatewayTool } from "./gateway-tool.js";
import { createImageTool } from "./image-tool.js";
import { createIosDeployTool } from "./ios-deploy-tool.js";
import { createMessageTool } from "./message-tool.js";
import { createNodesTool } from "./nodes-tool.js";
import { createParallelSpawnTool } from "./parallel-spawn-tool.js";
import { createProactiveBufferTool } from "./proactive-buffer-tool.js";
import { createDeliverableTool } from "./proactive-deliverable-tool.js";
import { createInfoSourcesTool } from "./proactive-sources-tool.js";
import { createProactiveTasksTool } from "./proactive-tasks-tool.js";
import { createRequestEscalationTool } from "./request-escalation-tool.js";
import { createRequestMissingToolsTool } from "./request-missing-tools-tool.js";
import { createSelfInspectingTool } from "./self-inspecting-tool.js";
import { createSelfSettingsTool } from "./self-settings-tool.js";
import { createSessionStatusTool } from "./session-status-tool.js";
import { createSessionsHistoryTool } from "./sessions-history-tool.js";
import { createSessionsListTool } from "./sessions-list-tool.js";
import { createSessionsSendTool } from "./sessions-send-tool.js";
import { createSessionsSpawnTool } from "./sessions-spawn-tool.js";
import { createSubagentsTool } from "./subagents-tool.js";
import { createTaskDispatchTool } from "./task-dispatch-tool.js";
import { createTtsTool } from "./tts-tool.js";
import { createWebFetchTool, createWebSearchTool } from "./web-tools.js";

export type CreateMarvToolsOptions = {
  sandboxBrowserBridgeUrl?: string;
  allowHostBrowserControl?: boolean;
  agentSessionKey?: string;
  agentChannel?: GatewayMessageChannel;
  agentAccountId?: string;
  /** Delivery target (e.g. telegram:group:123:topic:456) for topic/thread routing. */
  agentTo?: string;
  /** Thread/topic identifier for routing replies to the originating thread. */
  agentThreadId?: string | number;
  /** Group id for channel-level tool policy inheritance. */
  agentGroupId?: string | null;
  /** Group channel label for channel-level tool policy inheritance. */
  agentGroupChannel?: string | null;
  /** Group space label for channel-level tool policy inheritance. */
  agentGroupSpace?: string | null;
  agentDir?: string;
  sandboxRoot?: string;
  sandboxFsBridge?: SandboxFsBridge;
  workspaceDir?: string;
  sandboxed?: boolean;
  config?: MarvConfig;
  pluginToolAllowlist?: string[];
  /** Current channel ID for auto-threading (Slack). */
  currentChannelId?: string;
  /** Current thread timestamp for auto-threading (Slack). */
  currentThreadTs?: string;
  /** Reply-to mode for Slack auto-threading. */
  replyToMode?: "off" | "first" | "all";
  /** Mutable ref to track if a reply was sent (for "first" mode). */
  hasRepliedRef?: { value: boolean };
  /** If true, the model has native vision capability */
  modelHasVision?: boolean;
  /** Explicit agent ID override for cron/hook sessions. */
  requesterAgentIdOverride?: string;
  /** Require explicit message targets (no implicit last-route sends). */
  requireExplicitMessageTarget?: boolean;
  /** If true, omit the message tool from the tool list. */
  disableMessageTool?: boolean;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
  /** True when the current request is not a forwarded or quoted third-party instruction. */
  directUserInstruction?: boolean;
  /** Enable proactive buffer tooling for managed proactive runs only. */
  enableProactiveBuffer?: boolean;
  /** Enable proactive tasks/goals tooling (continuous loop mode). */
  enableProactiveTasks?: boolean;
};

export function createMarvTools(options?: CreateMarvToolsOptions): AnyAgentTool[] {
  const workspaceDir = resolveWorkspaceRoot(options?.workspaceDir);
  const imageTool = options?.agentDir?.trim()
    ? createImageTool({
        config: options?.config,
        agentDir: options.agentDir,
        workspaceDir,
        sandbox:
          options?.sandboxRoot && options?.sandboxFsBridge
            ? { root: options.sandboxRoot, bridge: options.sandboxFsBridge }
            : undefined,
        modelHasVision: options?.modelHasVision,
      })
    : null;
  const webSearchTool = createWebSearchTool({
    config: options?.config,
    sandboxed: options?.sandboxed,
  });
  const webFetchTool = createWebFetchTool({
    config: options?.config,
    sandboxed: options?.sandboxed,
  });
  const externalCliTool = createExternalCliTool({
    config: options?.config,
    workspaceDir,
    sandboxed: options?.sandboxed,
  });
  const cliProfilesTool = createCliProfilesTool({
    config: options?.config,
    sandboxed: options?.sandboxed,
  });
  const cliInvokeTool = createCliInvokeTool({
    config: options?.config,
    workspaceDir,
    sandboxed: options?.sandboxed,
  });
  const cliSynthesizeTool = createCliSynthesizeTool({
    config: options?.config,
    workspaceDir,
    sandboxed: options?.sandboxed,
  });
  // cli_verify removed: cli_synthesize auto-verifies on registration.
  // Kept available as createCliVerifyTool() for programmatic re-verification if needed.
  const messageTool = options?.disableMessageTool
    ? null
    : createMessageTool({
        agentAccountId: options?.agentAccountId,
        agentSessionKey: options?.agentSessionKey,
        config: options?.config,
        currentChannelId: options?.currentChannelId,
        currentChannelProvider: options?.agentChannel,
        currentThreadTs: options?.currentThreadTs,
        replyToMode: options?.replyToMode,
        hasRepliedRef: options?.hasRepliedRef,
        sandboxRoot: options?.sandboxRoot,
        requireExplicitTarget: options?.requireExplicitMessageTarget,
      });
  const proactiveBufferTool = createProactiveBufferTool({
    config: options?.config,
    agentSessionKey: options?.agentSessionKey,
    enabled: options?.enableProactiveBuffer,
  });
  const proactiveTasksTool = createProactiveTasksTool({
    config: options?.config,
    agentSessionKey: options?.agentSessionKey,
    enabled: options?.enableProactiveTasks,
  });
  const infoSourcesTool = createInfoSourcesTool({
    config: options?.config,
    agentSessionKey: options?.agentSessionKey,
    enabled: options?.enableProactiveTasks,
  });
  const deliverableTool = createDeliverableTool({
    config: options?.config,
    agentSessionKey: options?.agentSessionKey,
    enabled: options?.enableProactiveTasks,
  });
  const tools: AnyAgentTool[] = [
    createBrowserTool({
      sandboxBridgeUrl: options?.sandboxBrowserBridgeUrl,
      allowHostControl: options?.allowHostBrowserControl,
    }),
    createCanvasTool({ config: options?.config }),
    createNodesTool({
      agentSessionKey: options?.agentSessionKey,
      config: options?.config,
    }),
    createCronTool({
      agentSessionKey: options?.agentSessionKey,
    }),
    ...(externalCliTool ? [externalCliTool] : []),
    ...(cliProfilesTool ? [cliProfilesTool] : []),
    ...(cliInvokeTool ? [cliInvokeTool] : []),
    ...(cliSynthesizeTool ? [cliSynthesizeTool] : []),
    // cli_verify no longer exposed as agent tool (merged into cli_synthesize).
    ...(proactiveBufferTool ? [proactiveBufferTool] : []),
    ...(proactiveTasksTool ? [proactiveTasksTool] : []),
    ...(infoSourcesTool ? [infoSourcesTool] : []),
    ...(deliverableTool ? [deliverableTool] : []),
    ...(messageTool ? [messageTool] : []),
    createTtsTool({
      agentChannel: options?.agentChannel,
      config: options?.config,
    }),
    createIosDeployTool(),
    createGatewayTool({
      agentSessionKey: options?.agentSessionKey,
      config: options?.config,
    }),
    createAgentsListTool({
      agentSessionKey: options?.agentSessionKey,
      requesterAgentIdOverride: options?.requesterAgentIdOverride,
    }),
    createSessionsListTool({
      agentSessionKey: options?.agentSessionKey,
      sandboxed: options?.sandboxed,
    }),
    createSessionsHistoryTool({
      agentSessionKey: options?.agentSessionKey,
      sandboxed: options?.sandboxed,
    }),
    createSessionsSendTool({
      agentSessionKey: options?.agentSessionKey,
      agentChannel: options?.agentChannel,
      sandboxed: options?.sandboxed,
    }),
    createSessionsSpawnTool({
      agentSessionKey: options?.agentSessionKey,
      agentChannel: options?.agentChannel,
      agentAccountId: options?.agentAccountId,
      agentTo: options?.agentTo,
      agentThreadId: options?.agentThreadId,
      agentGroupId: options?.agentGroupId,
      agentGroupChannel: options?.agentGroupChannel,
      agentGroupSpace: options?.agentGroupSpace,
      sandboxed: options?.sandboxed,
      requesterAgentIdOverride: options?.requesterAgentIdOverride,
    }),
    createTaskDispatchTool({
      agentSessionKey: options?.agentSessionKey,
      agentChannel: options?.agentChannel,
      agentAccountId: options?.agentAccountId,
      agentTo: options?.agentTo,
      agentThreadId: options?.agentThreadId,
      agentGroupId: options?.agentGroupId,
      agentGroupChannel: options?.agentGroupChannel,
      agentGroupSpace: options?.agentGroupSpace,
      requesterAgentIdOverride: options?.requesterAgentIdOverride,
    }),
    createParallelSpawnTool({
      agentSessionKey: options?.agentSessionKey,
      agentChannel: options?.agentChannel,
      agentAccountId: options?.agentAccountId,
      agentTo: options?.agentTo,
      agentThreadId: options?.agentThreadId,
      agentGroupId: options?.agentGroupId,
      agentGroupChannel: options?.agentGroupChannel,
      agentGroupSpace: options?.agentGroupSpace,
      sandboxed: options?.sandboxed,
      requesterAgentIdOverride: options?.requesterAgentIdOverride,
    }),
    createSubagentsTool({
      agentSessionKey: options?.agentSessionKey,
    }),
    createSessionStatusTool({
      agentSessionKey: options?.agentSessionKey,
      config: options?.config,
    }),
    createSelfSettingsTool({
      agentSessionKey: options?.agentSessionKey,
      config: options?.config,
      agentChannel: options?.agentChannel,
      agentAccountId: options?.agentAccountId,
      agentTo: options?.agentTo,
      senderId: options?.senderId,
      senderName: options?.senderName,
      senderUsername: options?.senderUsername,
      senderE164: options?.senderE164,
      directUserInstruction: options?.directUserInstruction,
    }),
    createRequestEscalationTool({
      agentSessionKey: options?.agentSessionKey,
      config: options?.config,
    }),
    createRequestMissingToolsTool({
      workspaceDir,
      config: options?.config,
      agentSessionKey: options?.agentSessionKey,
    }),
    ...(webSearchTool ? [webSearchTool] : []),
    ...(webFetchTool ? [webFetchTool] : []),
    ...(imageTool ? [imageTool] : []),
  ];

  const pluginTools = resolvePluginTools({
    context: {
      config: options?.config,
      workspaceDir,
      agentDir: options?.agentDir,
      agentId: resolveSessionAgentId({
        sessionKey: options?.agentSessionKey,
        config: options?.config,
      }),
      sessionKey: options?.agentSessionKey,
      messageChannel: options?.agentChannel,
      agentAccountId: options?.agentAccountId,
      sandboxed: options?.sandboxed,
    },
    existingToolNames: new Set(tools.map((tool) => tool.name)),
    toolAllowlist: options?.pluginToolAllowlist,
  });

  const availableToolNames = [
    ...tools.map((tool) => tool.name),
    ...pluginTools.map((tool) => tool.name),
    "self_inspecting",
  ];
  const selfInspectingTool = createSelfInspectingTool({
    agentSessionKey: options?.agentSessionKey,
    config: options?.config,
    availableToolNames,
    directUserInstruction: options?.directUserInstruction,
  });

  return [...tools, selfInspectingTool, ...pluginTools];
}
