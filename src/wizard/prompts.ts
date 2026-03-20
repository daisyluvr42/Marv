export type WizardSelectOption<T = string> = {
  value: T;
  label: string;
  hint?: string;
};

export type WizardSelectParams<T = string> = {
  message: string;
  options: Array<WizardSelectOption<T>>;
  initialValue?: T;
};

export type WizardMultiSelectParams<T = string> = {
  message: string;
  options: Array<WizardSelectOption<T>>;
  initialValues?: T[];
  searchable?: boolean;
};

export type WizardTextParams = {
  message: string;
  initialValue?: string;
  placeholder?: string;
  validate?: (value: string) => string | undefined;
};

export type WizardConfirmParams = {
  message: string;
  initialValue?: boolean;
};

export type WizardProgress = {
  update: (message: string) => void;
  stop: (message?: string) => void;
};

export type WizardPrompter = {
  intro: (title: string) => Promise<void>;
  outro: (message: string) => Promise<void>;
  note: (message: string, title?: string) => Promise<void>;
  select: <T>(params: WizardSelectParams<T>) => Promise<T>;
  multiselect: <T>(params: WizardMultiSelectParams<T>) => Promise<T[]>;
  text: (params: WizardTextParams) => Promise<string>;
  confirm: (params: WizardConfirmParams) => Promise<boolean>;
  progress: (label: string) => WizardProgress;
};

export class WizardCancelledError extends Error {
  constructor(message = "wizard cancelled") {
    super(message);
    this.name = "WizardCancelledError";
  }
}

/**
 * Thrown when the user chooses "← Back" at a prompt.
 * The step runner catches this and re-runs the previous step.
 */
export class WizardBackSignal extends Error {
  constructor() {
    super("wizard back");
    this.name = "WizardBackSignal";
  }
}

/** Sentinel value injected into select options to represent "go back". */
export const WIZARD_BACK_VALUE = "__wizard_back__" as const;

/**
 * Wrap a prompter so that `select` prompts include a "← Back" option.
 * When the user picks it, `WizardBackSignal` is thrown.
 *
 * All other methods delegate unchanged.
 */
export function withBackSupport(inner: WizardPrompter): WizardPrompter {
  return {
    ...inner,
    select: async <T>(params: WizardSelectParams<T>): Promise<T> => {
      const backOption = {
        value: WIZARD_BACK_VALUE as unknown as T,
        label: "← Back",
        hint: "Return to previous step",
      };
      const result = await inner.select<T>({
        ...params,
        options: [...params.options, backOption],
      });
      if ((result as unknown) === WIZARD_BACK_VALUE) {
        throw new WizardBackSignal();
      }
      return result;
    },
  };
}
