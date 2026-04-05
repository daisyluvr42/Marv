import { describe, expect, it } from "vitest";
import { SILENT_REPLY_TOKEN } from "../../auto-reply/support/tokens.js";
import { buildSubagentSystemPrompt } from "../subagent-announce.js";
import { buildAgentSystemPrompt, buildRuntimeLine } from "./system-prompt.js";

describe("buildAgentSystemPrompt", () => {
  it("includes owner numbers when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      ownerNumbers: ["+123", " +456 ", ""],
    });

    expect(prompt).toContain("## User Identity");
    expect(prompt).toContain(
      "Owner numbers: +123, +456. Treat messages from these numbers as the user.",
    );
  });

  it("omits owner section when numbers are missing", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).not.toContain("## User Identity");
    expect(prompt).not.toContain("Owner numbers:");
  });

  it("omits extended sections in minimal prompt mode", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      promptMode: "minimal",
      ownerNumbers: ["+123"],
      skillsPrompt:
        "<available_skills>\n  <skill>\n    <name>demo</name>\n  </skill>\n</available_skills>",
      heartbeatPrompt: "ping",
      toolNames: ["message", "memory_search"],
      docsPath: "/tmp/marv/docs",
      extraSystemPrompt: "Subagent details",
      ttsHint: "Voice (TTS) is enabled.",
    });

    expect(prompt).not.toContain("## User Identity");
    expect(prompt).not.toContain("## Skills");
    expect(prompt).not.toContain("## Memory Recall");
    expect(prompt).not.toContain("## Documentation");
    expect(prompt).not.toContain("## Reply Tags");
    expect(prompt).not.toContain("## Messaging");
    expect(prompt).not.toContain("## Voice (TTS)");
    expect(prompt).not.toContain("## Silent Replies");
    expect(prompt).not.toContain("## Heartbeats");
    expect(prompt).toContain("## Safety");
    expect(prompt).toContain(
      "For long waits, avoid rapid poll loops: use exec with enough yieldMs or process(action=poll, timeout=<ms>).",
    );
    expect(prompt).toContain("You have no independent goals");
    expect(prompt).toContain("Prioritize safety and human oversight");
    expect(prompt).toContain("if instructions conflict");
    expect(prompt).toContain("Inspired by Anthropic's constitution");
    expect(prompt).toContain("Do not manipulate or persuade anyone");
    expect(prompt).toContain("Do not copy yourself or change system prompts");
    expect(prompt).toContain("## Subagent Context");
    expect(prompt).not.toContain("## Group Chat Context");
    expect(prompt).toContain("Subagent details");
  });

  it("includes safety guardrails in full prompts", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).toContain("## Safety");
    expect(prompt).toContain("You have no independent goals");
    expect(prompt).toContain("Prioritize safety and human oversight");
    expect(prompt).toContain("if instructions conflict");
    expect(prompt).toContain("Inspired by Anthropic's constitution");
    expect(prompt).toContain("Do not manipulate or persuade anyone");
    expect(prompt).toContain("Do not copy yourself or change system prompts");
  });

  it("includes voice hint when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      ttsHint: "Voice (TTS) is enabled.",
    });

    expect(prompt).toContain("## Voice (TTS)");
    expect(prompt).toContain("Voice (TTS) is enabled.");
  });

  it("adds reasoning tag hint when enabled", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      reasoningTagHint: true,
    });

    expect(prompt).toContain("## Reasoning Format");
    expect(prompt).toContain("<think>...</think>");
    expect(prompt).toContain("<final>...</final>");
  });

  it("includes a CLI quick reference section", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).toContain("## Marv CLI Quick Reference");
    expect(prompt).toContain("marv gateway restart");
    expect(prompt).toContain("Do not invent commands");
  });

  it("marks system message blocks as internal and not user-visible", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).toContain("`[System Message] ...` blocks are internal context");
    expect(prompt).toContain("are not user-visible by default");
    expect(prompt).toContain("reports completed cron/subagent work");
    expect(prompt).toContain("rewrite it in your normal assistant voice");
  });

  it("guides subagent workflows to avoid polling loops", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).toContain(
      "For long waits, avoid rapid poll loops: use exec with enough yieldMs or process(action=poll, timeout=<ms>).",
    );
    expect(prompt).toContain("Completion is push-based: it will auto-announce when done.");
    expect(prompt).toContain("Do not poll `subagents list` / `sessions_list` in a loop");
  });

  it("lists available tools when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["exec", "sessions_list", "sessions_history", "sessions_send"],
    });

    expect(prompt).toContain("Tool availability (filtered by policy):");
    expect(prompt).toContain("sessions_list");
    expect(prompt).toContain("sessions_history");
    expect(prompt).toContain("sessions_send");
  });

  it("includes autonomy helper guidance when discovery/escalation tools are available", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["request_missing_tools", "request_escalation"],
    });

    expect(prompt).toContain("## Autonomy Helpers");
    expect(prompt).toContain("request_missing_tools");
    expect(prompt).toContain("request_escalation");
    expect(prompt).toContain("approvalId");
  });

  it("tells the agent not to escalate ordinary cron mutations", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["cron", "request_escalation"],
    });

    expect(prompt).toContain(
      "Normal `cron` add/update/remove work does not require `request_escalation`",
    );
    expect(prompt).toContain("operator notifications/audit");
  });

  it("preserves tool casing in the prompt", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["Read", "Exec", "process"],
      skillsPrompt:
        "<available_skills>\n  <skill>\n    <name>demo</name>\n  </skill>\n</available_skills>",
      docsPath: "/tmp/marv/docs",
    });

    expect(prompt).toContain("- Read: Read file contents");
    expect(prompt).toContain("- Exec: Run shell commands");
    expect(prompt).toContain(
      "- If exactly one skill clearly applies: read its SKILL.md at <location> with `Read`, then follow it.",
    );
    expect(prompt).toContain("Marv docs: /tmp/marv/docs");
    expect(prompt).toContain(
      "For Marv behavior, commands, config, or architecture: consult local docs first.",
    );
  });

  it("includes docs guidance when docsPath is provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      docsPath: "/tmp/marv/docs",
    });

    expect(prompt).toContain("## Documentation");
    expect(prompt).toContain("Marv docs: /tmp/marv/docs");
    expect(prompt).toContain(
      "For Marv behavior, commands, config, or architecture: consult local docs first.",
    );
  });

  it("includes workspace notes when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      workspaceNotes: ["Reminder: commit your changes in this workspace after edits."],
    });

    expect(prompt).toContain("Reminder: commit your changes in this workspace after edits.");
  });

  it("renders recalled context in a separate section", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      contextFiles: [
        { path: "/tmp/marv/SOUL.md", content: "persona anchor" },
        {
          path: "/virtual/RECALLED_CONTEXT.md",
          content: "### memory\nremember this",
        },
      ],
    });

    expect(prompt).toContain("# Project Context");
    expect(prompt).toContain("# Recalled Context");
    expect(prompt).toContain("remember this");
  });

  it("includes user timezone when provided (12-hour)", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      userTimezone: "America/Chicago",
      userTime: "Monday, January 5th, 2026 — 3:26 PM",
      userTimeFormat: "12",
    });

    expect(prompt).toContain("## Current Date & Time");
    expect(prompt).toContain("Time zone: America/Chicago");
  });

  it("includes user timezone when provided (24-hour)", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      userTimezone: "America/Chicago",
      userTime: "Monday, January 5th, 2026 — 15:26",
      userTimeFormat: "24",
    });

    expect(prompt).toContain("## Current Date & Time");
    expect(prompt).toContain("Time zone: America/Chicago");
  });

  it("shows timezone when only timezone is provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      userTimezone: "America/Chicago",
      userTimeFormat: "24",
    });

    expect(prompt).toContain("## Current Date & Time");
    expect(prompt).toContain("Time zone: America/Chicago");
  });

  it("hints to use session_status for current date/time", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/clawd",
      userTimezone: "America/Chicago",
    });

    expect(prompt).toContain("session_status");
    expect(prompt).toContain("current date");
  });

  it("forbids automatic pinyin in Chinese replies", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["memory_search", "memory_write"],
    });

    expect(prompt).toContain(
      "Chinese-specific: never add pinyin romanization unless the user explicitly asks for it.",
    );
    expect(prompt).toContain("Reply in plain Chinese characters only.");
    expect(prompt).toContain("response_language");
  });

  it("teaches self inspection vs self settings tool boundaries", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    expect(prompt).toContain("self_inspecting");
    expect(prompt).toContain(
      "available models, scheduled tasks, tool limits, or why you are behaving a certain way",
    );
    expect(prompt).toContain("self_settings");
    expect(prompt).toContain("heartbeat behavior and HEARTBEAT.md maintenance");
    expect(prompt).toContain(
      "When the user asks you to inspect or explain your own current state, status, settings, available models, scheduled tasks, or current behavior, use self_inspecting first.",
    );
    expect(prompt).toContain("Do not guess or switch models before checking.");
    expect(prompt).toContain("change your own settings or behavior, use self_settings");
    expect(prompt).toContain("Session-level and task-level self adjustments may use self_settings");
    expect(prompt).toContain("heartbeat settings, HEARTBEAT.md, and external CLI");
  });

  it("includes heartbeat autonomy guardrails when heartbeat is configured", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["self_settings"],
      heartbeatPrompt: "Read HEARTBEAT.md and check for blockers.",
    });

    expect(prompt).toContain("## Heartbeats");
    expect(prompt).toContain("Read HEARTBEAT.md and check for blockers.");
    expect(prompt).toContain("On heartbeat poll:");
    expect(prompt).toContain(
      "This section applies ONLY to periodic heartbeat polls, NOT to normal user messages.",
    );
  });

  // The system prompt intentionally does NOT include the current date/time.
  // Only the timezone is included, to keep the prompt stable for caching.
  // See: https://github.com/moltbot/moltbot/commit/66eec295b894bce8333886cfbca3b960c57c4946
  // Agents should use session_status or message timestamps to determine the date/time.
  // Related: https://github.com/moltbot/moltbot/issues/1897
  //          https://github.com/moltbot/moltbot/issues/3658
  it("does NOT include a date or time in the system prompt (cache stability)", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/clawd",
      userTimezone: "America/Chicago",
      userTime: "Monday, January 5th, 2026 — 3:26 PM",
      userTimeFormat: "12",
    });

    // The prompt should contain the timezone but NOT the formatted date/time string.
    // This is intentional for prompt cache stability — the date/time was removed in
    // commit 66eec295b. If you're here because you want to add it back, please see
    // https://github.com/moltbot/moltbot/issues/3658 for the preferred approach:
    // gateway-level timestamp injection into messages, not the system prompt.
    expect(prompt).toContain("Time zone: America/Chicago");
    expect(prompt).not.toContain("Monday, January 5th, 2026");
    expect(prompt).not.toContain("3:26 PM");
    expect(prompt).not.toContain("15:26");
  });

  it("includes model alias guidance when aliases are provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      modelAliasLines: [
        "- Opus: anthropic/claude-opus-4-5",
        "- Sonnet: anthropic/claude-sonnet-4-5",
      ],
    });

    expect(prompt).toContain("## Model Aliases");
    expect(prompt).toContain("Prefer aliases when specifying model overrides");
    expect(prompt).toContain("- Opus: anthropic/claude-opus-4-5");
  });

  it("adds ClaudeBot self-update guidance when gateway tool is available", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["gateway", "exec"],
    });

    expect(prompt).toContain("## Marv Self-Update");
    expect(prompt).toContain("config.apply");
    expect(prompt).toContain("update.run");
    expect(prompt).toContain('action: "config.get"');
    expect(prompt).toContain("Do not hand-edit the active config");
  });

  it("includes skills guidance when skills prompt is present", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      skillsPrompt:
        "<available_skills>\n  <skill>\n    <name>demo</name>\n  </skill>\n</available_skills>",
    });

    expect(prompt).toContain("## Skills");
    expect(prompt).toContain(
      "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.",
    );
  });

  it("appends available skills when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      skillsPrompt:
        "<available_skills>\n  <skill>\n    <name>demo</name>\n  </skill>\n</available_skills>",
    });

    expect(prompt).toContain("<available_skills>");
    expect(prompt).toContain("<name>demo</name>");
  });

  it("omits skills section when no skills prompt is provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
    });

    expect(prompt).not.toContain("## Skills");
    expect(prompt).not.toContain("<available_skills>");
  });

  it("renders project context files when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      contextFiles: [
        { path: "AGENTS.md", content: "Alpha" },
        { path: "IDENTITY.md", content: "Bravo" },
      ],
    });

    expect(prompt).toContain("# Project Context");
    expect(prompt).toContain("## AGENTS.md");
    expect(prompt).toContain("Alpha");
    expect(prompt).toContain("## IDENTITY.md");
    expect(prompt).toContain("Bravo");
  });

  it("ignores context files with missing or blank paths", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      contextFiles: [
        { path: undefined as unknown as string, content: "Missing path" },
        { path: "   ", content: "Blank path" },
        { path: "AGENTS.md", content: "Alpha" },
      ],
    });

    expect(prompt).toContain("# Project Context");
    expect(prompt).toContain("## AGENTS.md");
    expect(prompt).toContain("Alpha");
    expect(prompt).not.toContain("Missing path");
    expect(prompt).not.toContain("Blank path");
  });

  it("adds SOUL guidance when a soul file is present", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      contextFiles: [
        { path: "./SOUL.md", content: "Persona" },
        { path: "dir\\SOUL.md", content: "Persona Windows" },
      ],
    });

    expect(prompt).toContain(
      "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.",
    );
  });

  it("adds explicit P0 guidance when structured P0 context is present", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      contextFiles: [
        { path: "P0 Soul", content: "Be kind." },
        { path: "P0 Identity", content: "You are Marv." },
        { path: "P0 User", content: "User prefers concise Chinese." },
      ],
    });

    expect(prompt).toContain("P0 guides tone, identity, and behavioral boundaries.");
    expect(prompt).toContain(
      "P0 does not override task facts, tool results, file contents, or explicit temporary task constraints unless a request conflicts with soul-level boundaries.",
    );
    expect(prompt).toContain(
      "P0 Soul is a strong constraint for persona, principles, and behavior boundaries. If a request conflicts with it, refuse or redirect while staying in character.",
    );
    expect(prompt).toContain(
      "P0 Identity is a strong constraint for self-description and speaking style, but it must not distort task facts or technical details.",
    );
    expect(prompt).toContain(
      "P0 User captures stable user preferences only, not transient state. Current explicit user requests can temporarily override it.",
    );
  });

  it("summarizes the message tool when available", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["message"],
    });

    expect(prompt).toContain("message: Send messages and channel actions");
    expect(prompt).toContain("### message tool");
    expect(prompt).toContain(`respond with ONLY: ${SILENT_REPLY_TOKEN}`);
  });

  it("includes inline button style guidance when runtime supports inline buttons", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      toolNames: ["message"],
      runtimeInfo: {
        channel: "telegram",
        capabilities: ["inlineButtons"],
      },
    });

    expect(prompt).toContain("buttons=[[{text,callback_data,style?}]]");
    expect(prompt).toContain("`style` can be `primary`, `success`, or `danger`");
  });

  it("includes runtime provider capabilities when present", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      runtimeInfo: {
        channel: "telegram",
        capabilities: ["inlineButtons"],
      },
    });

    expect(prompt).toContain("channel=telegram");
    expect(prompt).toContain("capabilities=inlineButtons");
  });

  it("includes agent id in runtime when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      runtimeInfo: {
        agentId: "work",
        host: "host",
        os: "macOS",
        arch: "arm64",
        node: "v20",
        model: "anthropic/claude",
      },
    });

    expect(prompt).toContain("agent=work");
  });

  it("includes reasoning visibility hint", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      reasoningLevel: "off",
    });

    expect(prompt).toContain("Reasoning: off");
    expect(prompt).toContain("/reasoning");
    expect(prompt).toContain("/status shows Reasoning");
  });

  it("builds runtime line with agent and channel details", () => {
    const line = buildRuntimeLine(
      {
        agentId: "work",
        host: "host",
        repoRoot: "/repo",
        os: "macOS",
        arch: "arm64",
        node: "v20",
        model: "anthropic/claude",
        defaultModel: "anthropic/claude-opus-4-5",
      },
      "telegram",
      ["inlineButtons"],
      "low",
    );

    expect(line).toContain("agent=work");
    expect(line).toContain("host=host");
    expect(line).toContain("repo=/repo");
    expect(line).toContain("os=macOS (arm64)");
    expect(line).toContain("node=v20");
    expect(line).toContain("model=anthropic/claude");
    expect(line).toContain("default_model=anthropic/claude-opus-4-5");
    expect(line).toContain("channel=telegram");
    expect(line).toContain("capabilities=inlineButtons");
    expect(line).toContain("thinking=low");
  });

  it("describes sandboxed runtime and elevated when allowed", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      sandboxInfo: {
        enabled: true,
        workspaceDir: "/tmp/sandbox",
        containerWorkspaceDir: "/workspace",
        workspaceAccess: "ro",
        agentWorkspaceMount: "/agent",
        elevated: { allowed: true, defaultLevel: "on" },
      },
    });

    expect(prompt).toContain("Your working directory is: /workspace");
    expect(prompt).toContain(
      "For read/write/edit/apply_patch, file paths resolve against host workspace: /tmp/marv. For bash/exec commands, use sandbox container paths under /workspace (or relative paths from that workdir), not host paths.",
    );
    expect(prompt).toContain("Sandbox container workdir: /workspace");
    expect(prompt).toContain(
      "Sandbox host mount source (file tools bridge only; not valid inside sandbox exec): /tmp/sandbox",
    );
    expect(prompt).toContain("You are running in a sandboxed runtime");
    expect(prompt).toContain("Sub-agents stay sandboxed");
    expect(prompt).toContain("User can toggle with /elevated on|off|ask|full.");
    expect(prompt).toContain("Current elevated level: on");
  });

  it("includes reaction guidance when provided", () => {
    const prompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/marv",
      reactionGuidance: {
        level: "minimal",
        channel: "Telegram",
      },
    });

    expect(prompt).toContain("## Reactions");
    expect(prompt).toContain("Reactions are enabled for Telegram in MINIMAL mode.");
  });
});

describe("buildSubagentSystemPrompt", () => {
  it("includes sub-agent spawning guidance for depth-1 orchestrator when maxSpawnDepth >= 2", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc",
      task: "research task",
      childDepth: 1,
      maxSpawnDepth: 2,
    });

    expect(prompt).toContain("## Sub-Agent Spawning");
    expect(prompt).toContain("You CAN spawn your own sub-agents");
    expect(prompt).toContain("sessions_spawn");
    expect(prompt).toContain("`subagents` tool");
    expect(prompt).toContain("announce their results back to you automatically");
    expect(prompt).toContain("Do NOT repeatedly poll `subagents list`");
  });

  it("does not include spawning guidance for depth-1 leaf when maxSpawnDepth == 1", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc",
      task: "research task",
      childDepth: 1,
      maxSpawnDepth: 1,
    });

    expect(prompt).not.toContain("## Sub-Agent Spawning");
    expect(prompt).not.toContain("You CAN spawn");
  });

  it("includes leaf worker note for depth-2 sub-sub-agents", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc:subagent:def",
      task: "leaf task",
      childDepth: 2,
      maxSpawnDepth: 2,
    });

    expect(prompt).toContain("## Sub-Agent Spawning");
    expect(prompt).toContain("leaf worker");
    expect(prompt).toContain("CANNOT spawn further sub-agents");
  });

  it("uses 'parent orchestrator' label for depth-2 agents", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc:subagent:def",
      task: "leaf task",
      childDepth: 2,
      maxSpawnDepth: 2,
    });

    expect(prompt).toContain("spawned by the parent orchestrator");
    expect(prompt).toContain("reported to the parent orchestrator");
  });

  it("uses 'main agent' label for depth-1 agents", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc",
      task: "orchestrator task",
      childDepth: 1,
      maxSpawnDepth: 2,
    });

    expect(prompt).toContain("spawned by the main agent");
    expect(prompt).toContain("reported to the main agent");
  });

  it("includes recovery guidance for compacted/truncated tool output", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc",
      task: "investigate logs",
      childDepth: 1,
      maxSpawnDepth: 2,
    });

    expect(prompt).toContain("[compacted: tool output removed to free context]");
    expect(prompt).toContain("[truncated: output exceeded context limit]");
    expect(prompt).toContain("offset/limit");
    expect(prompt).toContain("instead of full-file `cat`");
  });

  it("defaults to depth 1 and maxSpawnDepth 1 when not provided", () => {
    const prompt = buildSubagentSystemPrompt({
      childSessionKey: "agent:main:subagent:abc",
      task: "basic task",
    });

    // Should not include spawning guidance (default maxSpawnDepth is 1, depth 1 is leaf)
    expect(prompt).not.toContain("## Sub-Agent Spawning");
    expect(prompt).toContain("spawned by the main agent");
  });
});
