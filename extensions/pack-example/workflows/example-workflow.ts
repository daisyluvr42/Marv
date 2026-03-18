import type { PackWorkflow } from "../../../src/plugin-sdk/pack.js";

/**
 * Example workflow definition.
 *
 * Replace with industry-specific workflows:
 * - Legal: contract review, due diligence, case analysis
 * - Medical: patient intake, diagnosis support, treatment planning
 * - Finance: risk assessment, portfolio review, compliance audit
 * - Real estate: property evaluation, market comparison, offer preparation
 */
export const exampleWorkflow: PackWorkflow = {
  id: "example-analysis",
  name: "示例分析流程",
  description: "一个示例多步骤分析工作流，展示如何组合不同类型的步骤",
  trigger: "分析|analyze|review",
  steps: [
    {
      id: "extract",
      type: "llm-task",
      config: {
        prompt: "从输入内容中提取关键信息点，以结构化JSON格式输出",
        schema: {
          type: "object",
          properties: {
            keyPoints: { type: "array", items: { type: "string" } },
            summary: { type: "string" },
          },
        },
      },
    },
    {
      id: "multi-perspective",
      type: "parallel",
      config: {
        roles: ["domain_expert", "risk_analyst", "quality_reviewer"],
        task: "基于提取的关键信息，从你的专业角度进行分析并给出建议",
      },
    },
    {
      id: "synthesize",
      type: "llm-task",
      config: {
        prompt: "综合各专家的分析意见，生成一份完整的分析报告",
      },
    },
    {
      id: "confirm",
      type: "approve",
      config: {
        prompt: "分析报告已生成，是否确认并发送？",
      },
    },
  ],
};
