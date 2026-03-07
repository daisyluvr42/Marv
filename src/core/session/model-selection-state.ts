import type { SessionEntry } from "../config/sessions.js";

export type SessionModelSelectionMode = "auto" | "manual";

export type SessionModelSelectionState = {
  mode: SessionModelSelectionMode;
  manualModelRef?: string;
  source: "session" | "legacy" | "default";
};

export function resolveSessionModelSelectionState(
  entry?: Pick<
    SessionEntry,
    "selectionMode" | "manualModelRef" | "providerOverride" | "modelOverride"
  >,
): SessionModelSelectionState {
  const manualModelRef = entry?.manualModelRef?.trim();
  const selectionMode = entry?.selectionMode;
  if (selectionMode === "manual" && manualModelRef) {
    return { mode: "manual", manualModelRef, source: "session" };
  }

  const legacyModel = entry?.modelOverride?.trim();
  if (legacyModel) {
    const legacyProvider = entry?.providerOverride?.trim();
    const legacyRef = legacyProvider ? `${legacyProvider}/${legacyModel}` : legacyModel;
    return { mode: "manual", manualModelRef: legacyRef, source: "legacy" };
  }

  return { mode: "auto", source: "default" };
}

export function setSessionManualModelSelection(entry: SessionEntry, modelRef: string): boolean {
  const trimmed = modelRef.trim();
  if (!trimmed) {
    return false;
  }
  let updated = false;
  if (entry.selectionMode !== "manual") {
    entry.selectionMode = "manual";
    updated = true;
  }
  if (entry.manualModelRef !== trimmed) {
    entry.manualModelRef = trimmed;
    updated = true;
  }
  if (updated) {
    entry.updatedAt = Date.now();
  }
  return updated;
}

export function clearSessionManualModelSelection(entry: SessionEntry): boolean {
  let updated = false;
  if (entry.selectionMode && entry.selectionMode !== "auto") {
    entry.selectionMode = "auto";
    updated = true;
  }
  if (entry.manualModelRef) {
    delete entry.manualModelRef;
    updated = true;
  }
  if (updated) {
    entry.updatedAt = Date.now();
  }
  return updated;
}
