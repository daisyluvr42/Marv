import { formatCliCommand } from "../cli/command-format.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { shortenHomePath } from "../utils.js";
import { requireValidConfig } from "./agents.command-shared.js";
import type { AgentSummary } from "./agents.config.js";
import { buildAgentSummaries } from "./agents.config.js";
import { buildProviderStatusIndex, listProvidersForAgent } from "./agents.providers.js";

type AgentsListOptions = {
  json?: boolean;
};

function formatSummary(summary: AgentSummary) {
  const defaultTag = summary.isDefault ? " (default)" : "";
  const header =
    summary.name && summary.name !== summary.id
      ? `${summary.id}${defaultTag} (${summary.name})`
      : `${summary.id}${defaultTag}`;

  const identityParts = [];
  if (summary.identityEmoji) {
    identityParts.push(summary.identityEmoji);
  }
  if (summary.identityName) {
    identityParts.push(summary.identityName);
  }
  const identityLine = identityParts.length > 0 ? identityParts.join(" ") : null;
  const identitySource =
    summary.identitySource === "identity"
      ? "IDENTITY.md"
      : summary.identitySource === "config"
        ? "config"
        : null;

  const lines = [`- ${header}`];
  if (identityLine) {
    lines.push(`  Identity: ${identityLine}${identitySource ? ` (${identitySource})` : ""}`);
  }
  lines.push(`  Workspace: ${shortenHomePath(summary.workspace)}`);
  lines.push(`  Agent dir: ${shortenHomePath(summary.agentDir)}`);
  if (summary.model) {
    lines.push(`  Model: ${summary.model}`);
  }
  lines.push("  Top-level routing: main only");
  if (summary.providers?.length) {
    lines.push("  Providers:");
    for (const provider of summary.providers) {
      lines.push(`    - ${provider}`);
    }
  }
  return lines.join("\n");
}

export async function agentsListCommand(
  opts: AgentsListOptions,
  runtime: RuntimeEnv = defaultRuntime,
) {
  const cfg = await requireValidConfig(runtime);
  if (!cfg) {
    return;
  }

  const summaries = buildAgentSummaries(cfg);

  const providerStatus = await buildProviderStatusIndex(cfg);

  for (const summary of summaries) {
    const providerLines = listProvidersForAgent({
      summaryIsDefault: summary.isDefault,
      cfg,
      providerStatus,
    });
    if (providerLines.length > 0) {
      summary.providers = providerLines;
    }
  }

  if (opts.json) {
    runtime.log(JSON.stringify(summaries, null, 2));
    return;
  }

  const lines = ["Agents:", ...summaries.map(formatSummary)];
  lines.push('Top-level multi-agent routing was removed. Use "main" plus enhanced subagents.');
  lines.push(
    `Channel status reflects local config/creds. For live health: ${formatCliCommand("marv channels status --probe")}.`,
  );
  runtime.log(lines.join("\n"));
}
