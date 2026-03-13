import { describe, expect, it, vi } from "vitest";
import {
  buildWizardStagePlan,
  createWizardStageController,
  formatWizardStageCard,
} from "./onboarding.stages.js";
import type { WizardPrompter } from "./prompts.js";

function createPrompter(): WizardPrompter {
  return {
    intro: vi.fn(async () => {}),
    outro: vi.fn(async () => {}),
    note: vi.fn(async () => {}),
    select: vi.fn(async () => "quickstart"),
    multiselect: vi.fn(async () => []),
    text: vi.fn(async () => ""),
    confirm: vi.fn(async () => true),
    progress: vi.fn(() => ({ update: vi.fn(), stop: vi.fn() })),
  };
}

describe("wizard stages", () => {
  it("keeps the local plan explicitly model-first", () => {
    expect(buildWizardStagePlan({ mode: "local" })).toEqual([
      "environment",
      "model",
      "setup",
      "review",
    ]);
  });

  it("skips the model stage for remote-only onboarding", () => {
    expect(buildWizardStagePlan({ mode: "remote" })).toEqual(["environment", "setup", "review"]);
  });

  it("formats reuse-aware cards for the local model stage", () => {
    expect(
      formatWizardStageCard("model", {
        flow: "advanced",
        mode: "local",
        reuseExistingLocalSetup: true,
      }),
    ).toEqual(
      expect.objectContaining({
        title: "Stage 2 - Model-first activation",
        message: expect.stringContaining("validate your existing default model"),
      }),
    );
  });

  it("emits each stage card at most once even if entered repeatedly", async () => {
    const prompter = createPrompter();
    const controller = createWizardStageController({
      prompter,
      plan: buildWizardStagePlan({ mode: "local" }),
      initialStage: "environment",
    });

    await controller.enter("model", { mode: "local" });
    await controller.enter("model", { mode: "local" });
    await controller.enter("setup", { mode: "local" });

    const noteMock = vi.mocked(prompter.note);
    expect(noteMock).toHaveBeenCalledTimes(2);
    expect(noteMock.mock.calls[0]?.[1]).toBe("Stage 2 - Model-first activation");
    expect(noteMock.mock.calls[1]?.[1]).toBe("Stage 3 - Structured setup");
  });
});
