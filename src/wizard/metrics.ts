import type {
  WizardConfirmParams,
  WizardMultiSelectParams,
  WizardProgress,
  WizardPrompter,
  WizardSelectParams,
  WizardTextParams,
} from "./prompts.js";

export type WizardInteractionType =
  | "intro"
  | "outro"
  | "note"
  | "select"
  | "multiselect"
  | "text"
  | "confirm"
  | "progress";

export type WizardMetrics = {
  totalSteps: number;
  interactionSteps: number;
  counts: Record<WizardInteractionType, number>;
};

const INTERACTIVE_TYPES = new Set<WizardInteractionType>([
  "select",
  "multiselect",
  "text",
  "confirm",
]);

function createEmptyCounts(): Record<WizardInteractionType, number> {
  return {
    intro: 0,
    outro: 0,
    note: 0,
    select: 0,
    multiselect: 0,
    text: 0,
    confirm: 0,
    progress: 0,
  };
}

export function instrumentWizardPrompter(prompter: WizardPrompter): {
  prompter: WizardPrompter;
  getMetrics: () => WizardMetrics;
} {
  const counts = createEmptyCounts();
  const record = (type: WizardInteractionType) => {
    counts[type] += 1;
  };

  return {
    prompter: {
      intro: async (title: string) => {
        record("intro");
        await prompter.intro(title);
      },
      outro: async (message: string) => {
        record("outro");
        await prompter.outro(message);
      },
      note: async (message: string, title?: string) => {
        record("note");
        await prompter.note(message, title);
      },
      select: async <T>(params: WizardSelectParams<T>) => {
        record("select");
        return await prompter.select(params);
      },
      multiselect: async <T>(params: WizardMultiSelectParams<T>) => {
        record("multiselect");
        return await prompter.multiselect(params);
      },
      text: async (params: WizardTextParams) => {
        record("text");
        return await prompter.text(params);
      },
      confirm: async (params: WizardConfirmParams) => {
        record("confirm");
        return await prompter.confirm(params);
      },
      progress: (label: string): WizardProgress => {
        record("progress");
        return prompter.progress(label);
      },
    },
    getMetrics: () => {
      const totalSteps = Object.values(counts).reduce((sum, value) => sum + value, 0);
      const interactionSteps = Object.entries(counts)
        .filter(([type]) => INTERACTIVE_TYPES.has(type as WizardInteractionType))
        .reduce((sum, [, value]) => sum + value, 0);
      return {
        totalSteps,
        interactionSteps,
        counts: { ...counts },
      };
    },
  };
}

export function formatWizardMetrics(metrics: WizardMetrics): string {
  return [
    `Wizard metrics: interactions=${metrics.interactionSteps} total=${metrics.totalSteps}`,
    `select=${metrics.counts.select}`,
    `multiselect=${metrics.counts.multiselect}`,
    `text=${metrics.counts.text}`,
    `confirm=${metrics.counts.confirm}`,
    `note=${metrics.counts.note}`,
    `intro=${metrics.counts.intro}`,
    `outro=${metrics.counts.outro}`,
    `progress=${metrics.counts.progress}`,
  ].join(" ");
}
