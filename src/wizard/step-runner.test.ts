import { describe, expect, it } from "vitest";
import { WizardBackSignal } from "./prompts.js";
import { runStepsWithBack, type WizardStepDef } from "./step-runner.js";

type State = { values: string[] };
type Ctx = { log: string[] };

function makeStep(
  name: string,
  value: string,
  opts?: { shouldRun?: (s: State) => boolean },
): WizardStepDef<State, Ctx> {
  return {
    name,
    run: async (state, ctx) => {
      ctx.log.push(name);
      return { values: [...state.values, value] };
    },
    shouldRun: opts?.shouldRun,
  };
}

describe("runStepsWithBack", () => {
  it("runs steps sequentially", async () => {
    const ctx: Ctx = { log: [] };
    const result = await runStepsWithBack(
      [makeStep("a", "A"), makeStep("b", "B"), makeStep("c", "C")],
      { values: [] },
      ctx,
    );
    expect(result.values).toEqual(["A", "B", "C"]);
    expect(ctx.log).toEqual(["a", "b", "c"]);
  });

  it("goes back one step on WizardBackSignal", async () => {
    const ctx: Ctx = { log: [] };
    let backCount = 0;
    const steps: WizardStepDef<State, Ctx>[] = [
      makeStep("a", "A"),
      {
        name: "b",
        run: async (state, ctx) => {
          ctx.log.push("b");
          if (backCount === 0) {
            backCount++;
            throw new WizardBackSignal();
          }
          return { values: [...state.values, "B2"] };
        },
      },
    ];
    const result = await runStepsWithBack(steps, { values: [] }, ctx);
    // Step a runs, then b throws back, then a runs again, then b runs successfully.
    expect(ctx.log).toEqual(["a", "b", "a", "b"]);
    expect(result.values).toEqual(["A", "B2"]);
  });

  it("skips steps where shouldRun returns false", async () => {
    const ctx: Ctx = { log: [] };
    const steps: WizardStepDef<State, Ctx>[] = [
      makeStep("a", "A"),
      makeStep("skip", "SKIP", { shouldRun: () => false }),
      makeStep("c", "C"),
    ];
    const result = await runStepsWithBack(steps, { values: [] }, ctx);
    expect(result.values).toEqual(["A", "C"]);
    expect(ctx.log).toEqual(["a", "c"]);
  });

  it("back skips over steps that were not run", async () => {
    const ctx: Ctx = { log: [] };
    let backCount = 0;
    const steps: WizardStepDef<State, Ctx>[] = [
      makeStep("a", "A"),
      makeStep("skip", "SKIP", { shouldRun: () => false }),
      {
        name: "c",
        run: async (state, ctx) => {
          ctx.log.push("c");
          if (backCount === 0) {
            backCount++;
            throw new WizardBackSignal();
          }
          return { values: [...state.values, "C2"] };
        },
      },
    ];
    const result = await runStepsWithBack(steps, { values: [] }, ctx);
    // c backs to a (skipping the "skip" step), then a runs, then c runs.
    expect(ctx.log).toEqual(["a", "c", "a", "c"]);
    expect(result.values).toEqual(["A", "C2"]);
  });

  it("re-throws WizardBackSignal at the first step", async () => {
    const ctx: Ctx = { log: [] };
    const steps: WizardStepDef<State, Ctx>[] = [
      {
        name: "a",
        run: async () => {
          throw new WizardBackSignal();
        },
      },
    ];
    await expect(runStepsWithBack(steps, { values: [] }, ctx)).rejects.toThrow(WizardBackSignal);
  });

  it("returns initialState for empty steps", async () => {
    const ctx: Ctx = { log: [] };
    const result = await runStepsWithBack([], { values: ["init"] }, ctx);
    expect(result.values).toEqual(["init"]);
  });

  it("restores state snapshot on back", async () => {
    const ctx: Ctx = { log: [] };
    let backCount = 0;
    const steps: WizardStepDef<State, Ctx>[] = [
      {
        name: "a",
        run: async (state) => ({ values: [...state.values, "A"] }),
      },
      {
        name: "b",
        run: async (state, ctx) => {
          ctx.log.push(`b:${JSON.stringify(state.values)}`);
          if (backCount === 0) {
            backCount++;
            throw new WizardBackSignal();
          }
          return { values: [...state.values, "B"] };
        },
      },
    ];
    const result = await runStepsWithBack(steps, { values: [] }, ctx);
    // When b runs the second time, state should be restored to before a ran (empty).
    expect(ctx.log).toEqual(['b:["A"]', 'b:["A"]']);
    expect(result.values).toEqual(["A", "B"]);
  });
});
