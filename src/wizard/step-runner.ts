import { WizardBackSignal } from "./prompts.js";

/**
 * A wizard step: receives state + context, returns updated state.
 * State must be JSON-serializable (plain data); context holds runtime objects.
 */
export type WizardStepFn<S, C> = (state: S, ctx: C) => Promise<S>;

export interface WizardStepDef<S, C> {
  /** Human-readable name (for debugging / metrics). */
  name: string;
  run: WizardStepFn<S, C>;
  /**
   * Optional guard: if provided and returns false, the step is skipped.
   * Evaluated against the current state before each run attempt.
   */
  shouldRun?: (state: S) => boolean;
}

type Checkpoint<S> = { index: number; state: string; parsed?: S };

/**
 * Run wizard steps sequentially with back-navigation support.
 *
 * - When a step throws `WizardBackSignal`, the runner restores state to
 *   the checkpoint before the *previous* completed step and re-runs it.
 * - Steps with `shouldRun` returning false are silently skipped in both
 *   forward and backward directions.
 * - State is snapshotted via JSON round-trip; keep it JSON-serializable.
 * - Any other error (including `WizardCancelledError`) propagates normally.
 */
export async function runStepsWithBack<S extends object, C>(
  steps: WizardStepDef<S, C>[],
  initialState: S,
  ctx: C,
): Promise<S> {
  if (steps.length === 0) {
    return initialState;
  }

  // Stack of checkpoints for completed steps.
  // Each entry records the step index and state *before* that step ran.
  const completedStack: Checkpoint<S>[] = [];
  let state: S = JSON.parse(JSON.stringify(initialState)) as S;
  let index = 0;

  while (index < steps.length) {
    const step = steps[index];

    // Skip steps whose guard returns false.
    if (step.shouldRun && !step.shouldRun(state)) {
      index++;
      continue;
    }

    const checkpoint: Checkpoint<S> = { index, state: JSON.stringify(state) };

    try {
      state = await step.run(state, ctx);
      completedStack.push(checkpoint);
      index++;
    } catch (err) {
      if (err instanceof WizardBackSignal) {
        if (completedStack.length === 0) {
          // At the very first step — can't go further back.
          throw err;
        }
        const prev = completedStack.pop()!;
        index = prev.index;
        state = JSON.parse(prev.state) as S;
      } else {
        throw err;
      }
    }
  }

  return state;
}
