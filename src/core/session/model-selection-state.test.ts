import { describe, expect, it } from "vitest";
import type { SessionEntry } from "../config/sessions.js";
import {
  clearSessionManualModelSelection,
  resolveSessionModelSelectionState,
  setSessionManualModelSelection,
} from "./model-selection-state.js";

describe("session model selection state", () => {
  it("prefers explicit manual selection state", () => {
    const state = resolveSessionModelSelectionState({
      selectionMode: "manual",
      manualModelRef: "openai/gpt-4o",
      providerOverride: "anthropic",
      modelOverride: "claude-sonnet-4-5",
    });

    expect(state).toEqual({
      mode: "manual",
      manualModelRef: "openai/gpt-4o",
      source: "session",
    });
  });

  it("maps legacy override fields into manual mode", () => {
    const state = resolveSessionModelSelectionState({
      modelOverride: "gpt-4o",
      providerOverride: "openai",
    });

    expect(state).toEqual({
      mode: "manual",
      manualModelRef: "openai/gpt-4o",
      source: "legacy",
    });
  });

  it("writes and clears manual selection fields", () => {
    const entry: SessionEntry = {
      sessionId: "session-1",
      updatedAt: 1,
    };

    expect(setSessionManualModelSelection(entry, "openai/gpt-4o")).toBe(true);
    expect(entry.selectionMode).toBe("manual");
    expect(entry.manualModelRef).toBe("openai/gpt-4o");

    expect(clearSessionManualModelSelection(entry)).toBe(true);
    expect(entry.selectionMode).toBe("auto");
    expect(entry.manualModelRef).toBeUndefined();
  });
});
