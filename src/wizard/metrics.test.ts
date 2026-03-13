import { describe, expect, it, vi } from "vitest";
import { createWizardPrompter } from "../commands/test-wizard-helpers.js";
import { formatWizardMetrics, instrumentWizardPrompter } from "./metrics.js";

describe("wizard metrics", () => {
  it("counts interactive and non-interactive prompt calls separately", async () => {
    const base = createWizardPrompter(
      {
        select: vi.fn(async () => "quickstart") as never,
        text: vi.fn(async () => "/tmp/workspace") as never,
        confirm: vi.fn(async () => true),
        note: vi.fn(async () => {}),
        intro: vi.fn(async () => {}),
        outro: vi.fn(async () => {}),
      },
      { defaultSelect: "quickstart" },
    );

    const instrumented = instrumentWizardPrompter(base);
    await instrumented.prompter.intro("Intro");
    await instrumented.prompter.note("Heads up");
    await instrumented.prompter.select({
      message: "Mode",
      options: [{ value: "quickstart", label: "QuickStart" }],
    });
    await instrumented.prompter.text({ message: "Workspace" });
    await instrumented.prompter.confirm({ message: "Continue?" });
    await instrumented.prompter.outro("Done");

    expect(instrumented.getMetrics()).toEqual({
      totalSteps: 6,
      interactionSteps: 3,
      counts: {
        intro: 1,
        outro: 1,
        note: 1,
        select: 1,
        multiselect: 0,
        text: 1,
        confirm: 1,
        progress: 0,
      },
    });
  });

  it("formats a compact metrics summary", () => {
    const line = formatWizardMetrics({
      totalSteps: 8,
      interactionSteps: 5,
      counts: {
        intro: 1,
        outro: 1,
        note: 1,
        select: 2,
        multiselect: 0,
        text: 1,
        confirm: 2,
        progress: 0,
      },
    });
    expect(line).toContain("Wizard metrics: interactions=5 total=8");
    expect(line).toContain("select=2");
    expect(line).toContain("confirm=2");
  });
});
