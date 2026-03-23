// Goal-driven subagent orchestration module.
// Provides evaluation, feedback, and iteration loops for delegated subagents.

export { buildContractFromGoalFrame, buildContractContextBlock } from "./contract-builder.js";
export { evaluateAuditCriterion, evaluateContract } from "./evaluation-gate.js";
export type { EvaluateOptions, ContractEvaluateOptions } from "./evaluation-gate.js";
export { buildFeedback } from "./feedback-builder.js";
export {
  startOrchestration,
  startOrchestrationWithContract,
  handleSubagentCompletion,
  deliverFeedback,
  checkBudget,
  runOrchestrationCycle,
  isTerminalPhase,
} from "./orchestration-loop.js";
export type { OrchestrationDeps } from "./orchestration-loop.js";
export {
  startOrchestrationGroup,
  runGroupOrchestration,
  resolveGroupPhase,
} from "./orchestration-group.js";
export type {
  AuditCriterion,
  AuditEvaluator,
  AuditResult,
  EvaluationResult,
  EvaluationVerdict,
  GoalContract,
  GoalContractBudget,
  OrchestrationConfig,
  OrchestrationEntry,
  OrchestrationGroup,
  OrchestrationPhase,
  StructuredFeedback,
} from "./types.js";
export { ORCHESTRATION_DEFAULTS } from "./types.js";
