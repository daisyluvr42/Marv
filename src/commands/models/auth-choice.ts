import { normalizeProviderId } from "../../agents/model/model-selection.js";
import {
  type AuthChoiceGroup,
  type AuthChoiceOption,
  buildAuthChoiceGroups,
} from "../auth-choice-options.js";
import { resolvePreferredProviderForAuthChoice } from "../auth-choice.preferred-provider.js";
import type { AuthChoice } from "../onboard-types.js";

function normalizeMatchValue(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function normalizeGroupValue(value: string): string {
  return value.trim().toLowerCase();
}

function normalizeProviderValue(value: string): string {
  return normalizeProviderId(value.trim().toLowerCase());
}

function optionMatches(option: AuthChoiceOption, raw: string): boolean {
  const normalizedRaw = normalizeMatchValue(raw);
  if (normalizeMatchValue(option.value) === normalizedRaw) {
    return true;
  }
  if (normalizeMatchValue(option.label) === normalizedRaw) {
    return true;
  }
  return false;
}

function groupMatches(group: AuthChoiceGroup, raw: string): boolean {
  const normalizedRaw = normalizeGroupValue(raw);
  if (normalizeGroupValue(group.value) === normalizedRaw) {
    return true;
  }
  return normalizeMatchValue(group.label) === normalizeMatchValue(raw);
}

function dedupeOptions(options: AuthChoiceOption[]): AuthChoiceOption[] {
  const seen = new Set<string>();
  return options.filter((option) => {
    if (seen.has(option.value)) {
      return false;
    }
    seen.add(option.value);
    return true;
  });
}

export function listModelsAuthChoiceGroups(): AuthChoiceGroup[] {
  return buildAuthChoiceGroups({
    includeSkip: false,
  }).groups;
}

export function resolveModelsAuthChoiceOptions(rawProvider?: string): AuthChoiceOption[] {
  const groups = listModelsAuthChoiceGroups();
  const requested = rawProvider?.trim();
  if (!requested) {
    return [];
  }

  const exactOption = groups
    .flatMap((group) => group.options)
    .find((option) => optionMatches(option, requested));
  if (exactOption) {
    return [exactOption];
  }

  const matchedGroups = groups.filter((group) => groupMatches(group, requested));
  if (matchedGroups.length > 0) {
    return dedupeOptions(matchedGroups.flatMap((group) => group.options));
  }

  const providerMatches = groups
    .flatMap((group) => group.options)
    .filter((option) => {
      const preferred = resolvePreferredProviderForAuthChoice(option.value);
      return preferred
        ? normalizeProviderValue(preferred) === normalizeProviderValue(requested)
        : false;
    });
  return dedupeOptions(providerMatches);
}

export function resolveModelsAuthChoice(params: { provider?: string; method?: string }): {
  choice: AuthChoice | null;
  options: AuthChoiceOption[];
} {
  const options = resolveModelsAuthChoiceOptions(params.provider);
  if (options.length === 0) {
    return { choice: null, options: [] };
  }
  const requestedMethod = params.method?.trim();
  if (!requestedMethod) {
    return {
      choice: options.length === 1 ? options[0].value : null,
      options,
    };
  }

  const matched = options.find((option) => optionMatches(option, requestedMethod));
  return { choice: matched?.value ?? null, options };
}
