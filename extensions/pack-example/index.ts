/**
 * Example Profession Pack
 *
 * This is a complete scaffold demonstrating how to build a profession pack.
 * Copy this extension directory and customize for your industry.
 *
 * What to customize:
 * 1. persona.ts — professional identity and constraints
 * 2. tools/ — domain-specific tools
 * 3. workflows/ — industry workflow templates
 * 4. knowledge/ — reference material for the knowledge base
 * 5. marv.plugin.json — configSchema for pack-specific settings
 * 6. package.json — name, description
 */

import { join } from "node:path";
import { registerPackKnowledge } from "../../src/plugin-sdk/pack-knowledge.js";
import { registerProfessionPack } from "../../src/plugin-sdk/pack.js";
import type { MarvPluginApi } from "../../src/plugins/types.js";
import { persona } from "./persona.js";
import { createDomainLookupTool } from "./tools/domain-lookup.js";
import { exampleWorkflow } from "./workflows/example-workflow.js";

export default function register(api: MarvPluginApi) {
  // Register the profession pack (persona + tools + workflows)
  registerProfessionPack(api, {
    id: "pack-example",
    name: "示例行业助手",
    persona,
    tools: [createDomainLookupTool],
    workflows: [exampleWorkflow],
  });

  // Register knowledge base (local vector store)
  registerPackKnowledge(
    api,
    "pack-example",
    { dir: "knowledge", autoInject: true },
    join(import.meta.dirname ?? __dirname, "knowledge"),
  );
}
