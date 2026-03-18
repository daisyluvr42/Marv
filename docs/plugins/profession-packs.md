---
summary: "Build and deploy industry-specific profession packs with custom persona, tools, workflows, and knowledge"
read_when:
  - Creating a new profession pack for a specific industry
  - Customizing Marv for vertical use cases
  - Deploying a pack to end users
title: "Profession Packs"
---

# Profession Packs

A profession pack turns Marv into a domain-specific assistant for a particular industry or role. Each pack bundles four things:

- **Persona** — professional identity, constraints, and communication style
- **Knowledge** — domain reference material (local vector store)
- **Workflows** — multi-step industry processes (e.g. contract review, patient intake)
- **Tools** — domain-specific capabilities (optional)

Packs are standard Marv plugins distributed as independent npm packages.

## Creating a new pack

### 1. Copy the scaffold

```bash
cp -r extensions/pack-example extensions/pack-<industry>
```

### 2. Update package metadata

**`package.json`**

```json
{
  "name": "@marv/pack-<industry>",
  "version": "2026.3.19",
  "type": "module",
  "devDependencies": {
    "agentmarv": "workspace:*"
  },
  "marv": {
    "extensions": ["./index.ts"]
  }
}
```

**`marv.plugin.json`**

```json
{
  "id": "pack-<industry>",
  "name": "Your Industry Pack",
  "description": "...",
  "configSchema": {
    "type": "object",
    "properties": {
      "companyName": { "type": "string" },
      "specialization": { "type": "string" },
      "embeddingApiKey": { "type": "string" }
    }
  }
}
```

### 3. Define the persona

Edit `persona.ts` with your industry identity:

```typescript
import type { PackPersona } from "agentmarv/plugin-sdk";

export const persona: PackPersona = {
  role: "你是XX律师事务所的资深法律顾问助理，专精知识产权领域",
  domain: "legal",
  constraints: [
    "不提供具体法律意见，仅提供信息参考",
    "涉及诉讼策略时建议咨询执业律师",
    "引用法条时标注具体条款号",
  ],
  responseStyle: "严谨专业，使用法律术语但附带通俗解释",
};
```

**Persona fields:**

| Field           | Required | Description                                            |
| --------------- | -------- | ------------------------------------------------------ |
| `role`          | Yes      | System role description injected into agent context    |
| `domain`        | Yes      | Domain identifier (e.g. `legal`, `medical`, `finance`) |
| `constraints`   | No       | Professional rules the agent must follow               |
| `responseStyle` | No       | Tone and formatting guidance                           |

### 4. Add domain knowledge

Place reference files in the `knowledge/` directory:

```
knowledge/
  contract-law-essentials.md
  ip-regulations.md
  common-risk-checklist.md
```

**Supported formats:** `.md`, `.txt`

Files are automatically chunked and indexed into a local LanceDB vector store on first use. The agent can search this knowledge explicitly via the `pack_knowledge_search` tool, and relevant snippets are auto-injected into context before each conversation turn.

To add files after installation:

```bash
marv pack knowledge add pack-legal ~/docs/new-regulation.md
marv pack update pack-legal  # rebuild index
```

**Requirements:** An OpenAI API key is needed for embeddings. Set it via:

- `plugins.pack-<id>.config.embeddingApiKey` in Marv config, or
- `OPENAI_API_KEY` environment variable

### 5. Define workflows (optional)

Create workflow files in `workflows/`:

```typescript
import type { PackWorkflow } from "agentmarv/plugin-sdk";

export const contractReview: PackWorkflow = {
  id: "contract-review",
  name: "合同审查",
  description: "分析合同条款、识别风险、生成审查报告",
  trigger: "审查合同|review contract",
  steps: [
    {
      id: "extract",
      type: "llm-task",
      config: { prompt: "提取合同中所有条款" },
    },
    {
      id: "analyze",
      type: "parallel",
      config: {
        roles: ["risk_analyst", "compliance_officer", "contract_attorney"],
        task: "基于提取的条款，从你的专业角度分析",
      },
    },
    {
      id: "synthesize",
      type: "llm-task",
      config: { prompt: "综合各专家意见生成审查报告" },
    },
    {
      id: "confirm",
      type: "approve",
      config: { prompt: "审查报告已生成，是否发送给客户？" },
    },
  ],
};
```

**Step types:**

| Type       | Maps to               | Use case                                  |
| ---------- | --------------------- | ----------------------------------------- |
| `llm-task` | Structured LLM output | Data extraction, report generation        |
| `parallel` | `parallel_spawn`      | Multiple expert perspectives in parallel  |
| `subagent` | Subagent delegation   | Complex subtask requiring dedicated agent |
| `approve`  | Human approval gate   | Confirmation before sending/publishing    |
| `tool`     | Direct tool call      | Run a specific registered tool            |

### 6. Add domain tools (optional)

Create tool factories in `tools/`:

```typescript
import type { MarvPluginToolContext, AnyAgentTool } from "agentmarv/plugin-sdk";

export function createClauseAnalyzer(_ctx: MarvPluginToolContext): AnyAgentTool | null {
  const { Type } = require("@sinclair/typebox");

  return {
    name: "clause_analyzer",
    label: "Clause Analyzer",
    description: "Analyze contract clauses for risks and issues",
    parameters: Type.Object({
      clause: Type.String({ description: "Contract clause text" }),
    }),
    async execute(_toolCallId, params) {
      const { clause } = params as { clause: string };
      // Your domain logic here
      return {
        content: [{ type: "text", text: "Analysis result..." }],
        details: { clause },
      };
    },
  };
}
```

### 7. Wire everything in index.ts

```typescript
import { join } from "node:path";
import type { MarvPluginApi } from "agentmarv/plugin-sdk";
import { registerProfessionPack } from "agentmarv/plugin-sdk";
import { registerPackKnowledge } from "agentmarv/plugin-sdk";
import { persona } from "./persona.js";
import { createClauseAnalyzer } from "./tools/clause-analyzer.js";
import { contractReview } from "./workflows/contract-review.js";

export default function register(api: MarvPluginApi) {
  registerProfessionPack(api, {
    id: "pack-legal",
    name: "法律顾问",
    persona,
    tools: [createClauseAnalyzer],
    workflows: [contractReview],
  });

  registerPackKnowledge(
    api,
    "pack-legal",
    { dir: "knowledge", autoInject: true },
    join(import.meta.dirname ?? __dirname, "knowledge"),
  );
}
```

## Deploying to users

For users who are not technical, you can deploy remotely:

```bash
# SSH to user machine or run a setup script
npm i -g marv@latest
marv pack install @marv/pack-legal
marv config set plugins.pack-legal.config.companyName "用户公司名"
marv config set plugins.pack-legal.config.specialization "知识产权"
# Knowledge auto-indexes on first agent start
```

## Pack CLI reference

```bash
marv pack install <name>              # Install from npm
marv pack list                        # List installed packs
marv pack update <id>                 # Rebuild knowledge index
marv pack knowledge add <id> <file>   # Add knowledge file
marv pack knowledge list <id>         # List knowledge files
```

## Using Anthropic Skills in packs

Packs can leverage Anthropic managed skills (DOCX, XLSX, PPTX, PDF processing) for document handling:

```typescript
import { useAnthropicSkill } from "agentmarv/plugin-sdk";

const result = await useAnthropicSkill(
  "docx",
  {
    content: base64Content,
    mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    action: "extract_text",
  },
  config,
);
```

Requires an Anthropic API key. Falls back gracefully when unavailable.

## Architecture

```
┌─────────────────────────────────────────┐
│  Profession Pack (@marv/pack-legal)     │
│  persona + tools + workflows + knowledge│
├─────────────────────────────────────────┤
│  Pack SDK (agentmarv/plugin-sdk)        │
│  registerProfessionPack()               │
│  registerPackKnowledge()                │
├─────────────────────────────────────────┤
│  Marv Plugin System                     │
│  hooks → tools → registry               │
├─────────────────────────────────────────┤
│  Core Agent                             │
│  system-prompt → tool-assembly → LLM    │
└─────────────────────────────────────────┘
```

- **Persona** injects via `before_prompt_build` hook (priority 200)
- **Knowledge** auto-injects via the same hook (priority 190)
- **Workflows** register as the `run_workflow` agent tool
- **Multiple packs** can coexist; context is concatenated by priority order
