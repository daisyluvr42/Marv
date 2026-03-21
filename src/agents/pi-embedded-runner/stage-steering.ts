/**
 * O2: Stage-aware steering for the goal loop.
 *
 * Two layers:
 * 1. Steering annotations (always active) — injected into goal steering context
 * 2. Narrow mutation deny (opt-in, mutation tasks only) — denies write/edit/apply_patch
 *    during early stages, with auto-relaxation after 2 stuck attempts
 */

export type StageSteering = {
  /** Injected into the goal steering context as guidance. */
  annotation: string;
  /** Optional tool names to deny at this stage (narrow: only file-mutation tools). */
  denyTools?: string[];
};

/**
 * Resolve stage-aware steering for the current direction node.
 *
 * - Only mutation tasks get tool deny (inquiry tasks get annotations only)
 * - Only write/edit/apply_patch are denied (exec/process/message stay available)
 * - Auto-relaxes after 2 stuck attempts at the same node to prevent jamming
 */
export function resolveStageSteering(params: {
  currentNode: string;
  goalType: "inquiry" | "mutation";
  stuckAttempts: number;
  denyEnabled: boolean;
}): StageSteering | null {
  const node = params.currentNode.toLowerCase();
  const isMutation = params.goalType === "mutation";
  const canDeny = params.denyEnabled && isMutation && params.stuckAttempts < 2;
  const mutationDeny = canDeny ? ["write", "edit", "apply_patch"] : undefined;

  if (node.includes("understand") || node.includes("clarify")) {
    return {
      annotation: "Focus on reading and gathering evidence before making changes.",
      denyTools: mutationDeny,
    };
  }
  if (node.includes("identify")) {
    return {
      annotation: "Identify the smallest safe change before applying it.",
      denyTools: mutationDeny,
    };
  }
  if (node.includes("validate") || node.includes("verify")) {
    return {
      annotation: "Verify results before making more changes.",
      denyTools: mutationDeny,
    };
  }
  // "apply" and "wrap up" stages: no restrictions, no special annotation
  return null;
}
