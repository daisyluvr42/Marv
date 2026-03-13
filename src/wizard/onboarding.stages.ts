import type { OnboardMode } from "../commands/onboard-types.js";
import type { WizardFlow, WizardStage } from "./onboarding.types.js";
import type { WizardPrompter } from "./prompts.js";

type WizardStageCardParams = {
  flow?: WizardFlow;
  mode?: OnboardMode;
  reuseExistingLocalSetup?: boolean;
};

type WizardStageCard = {
  title: string;
  message: string;
};

const WIZARD_STAGE_TITLES: Record<WizardStage, string> = {
  environment: "Stage 1 - Environment and risk",
  model: "Stage 2 - Model-first activation",
  setup: "Stage 3 - Structured setup",
  review: "Stage 4 - Review and activation",
};

export function buildWizardStagePlan(params: { mode: OnboardMode }): WizardStage[] {
  return params.mode === "remote"
    ? ["environment", "setup", "review"]
    : ["environment", "model", "setup", "review"];
}

function describeFlow(flow: WizardFlow | undefined): string {
  if (flow === "advanced") {
    return "Manual keeps the full set of gateway and routing controls visible.";
  }
  if (flow === "quickstart") {
    return "QuickStart keeps the common path short and leans on safer defaults.";
  }
  return "We will keep the setup path concise and only ask for the next meaningful choice.";
}

function buildEnvironmentStageCard(params: WizardStageCardParams): WizardStageCard {
  return {
    title: WIZARD_STAGE_TITLES.environment,
    message: [
      "First we confirm the safety baseline and choose the setup path for this machine.",
      describeFlow(params.flow),
      "You can still revise the resulting setup before the wizard finishes.",
    ].join("\n"),
  };
}

function buildModelStageCard(params: WizardStageCardParams): WizardStageCard {
  const body = params.reuseExistingLocalSetup
    ? "We will validate your existing default model before touching the rest of the local setup."
    : "We will pick one working default model first, then carry that through the rest of onboarding.";
  return {
    title: WIZARD_STAGE_TITLES.model,
    message: [
      body,
      "This keeps model readiness ahead of workspace, channel, and gateway polish.",
    ].join("\n"),
  };
}

function buildSetupStageCard(params: WizardStageCardParams): WizardStageCard {
  if (params.mode === "remote") {
    return {
      title: WIZARD_STAGE_TITLES.setup,
      message: [
        "This path only captures remote gateway details.",
        "Local model, workspace, and channel bootstrap steps are skipped on purpose.",
      ].join("\n"),
    };
  }
  if (params.reuseExistingLocalSetup) {
    return {
      title: WIZARD_STAGE_TITLES.setup,
      message: [
        "We will keep your current workspace, gateway, channels, and hooks intact.",
        "Only the missing local wiring is refreshed before the final review.",
      ].join("\n"),
    };
  }
  return {
    title: WIZARD_STAGE_TITLES.setup,
    message: [
      "Now we apply the structured local setup: workspace, routing, gateway, channels, and hooks.",
      "QuickStart keeps defaults tight; Manual keeps the advanced switches available.",
    ].join("\n"),
  };
}

function buildReviewStageCard(params: WizardStageCardParams): WizardStageCard {
  if (params.mode === "remote") {
    return {
      title: WIZARD_STAGE_TITLES.review,
      message: [
        "We will review the remote gateway coordinates before saving them.",
        "No local bootstrap runs on the remote-only path.",
      ].join("\n"),
    };
  }
  return {
    title: WIZARD_STAGE_TITLES.review,
    message: [
      "Finally we summarize the exact capabilities enabled by this setup.",
      "From there you can hatch immediately or make a bounded revision without reopening the whole wizard.",
    ].join("\n"),
  };
}

export function formatWizardStageCard(
  stage: WizardStage,
  params: WizardStageCardParams = {},
): WizardStageCard {
  switch (stage) {
    case "environment":
      return buildEnvironmentStageCard(params);
    case "model":
      return buildModelStageCard(params);
    case "setup":
      return buildSetupStageCard(params);
    case "review":
      return buildReviewStageCard(params);
  }
}

export async function presentWizardStage(
  prompter: WizardPrompter,
  stage: WizardStage,
  params: WizardStageCardParams = {},
) {
  const card = formatWizardStageCard(stage, params);
  await prompter.note(card.message, card.title);
}

export function createWizardStageController(params: {
  prompter: WizardPrompter;
  plan: WizardStage[];
  initialStage?: WizardStage;
}) {
  const stageIndex = new Map(params.plan.map((stage, index) => [stage, index]));
  let currentStage = params.initialStage;
  let currentIndex =
    params.initialStage === undefined ? -1 : (stageIndex.get(params.initialStage) ?? -1);
  return {
    currentStage: () => currentStage,
    async enter(stage: WizardStage, cardParams: WizardStageCardParams = {}) {
      const nextIndex = stageIndex.get(stage);
      if (nextIndex === undefined) {
        return;
      }
      if (nextIndex <= currentIndex) {
        return;
      }
      currentIndex = nextIndex;
      currentStage = stage;
      await presentWizardStage(params.prompter, stage, cardParams);
    },
  };
}
