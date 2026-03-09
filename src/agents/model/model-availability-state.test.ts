import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getRuntimeModelAvailability,
  markRuntimeModelFailure,
  readRuntimeModelAvailability,
} from "./model-availability-state.js";

describe("model availability state", () => {
  const realStateDir = process.env.MARV_STATE_DIR;
  let tempStateDir = "";

  beforeEach(() => {
    tempStateDir = fs.mkdtempSync(path.join(os.tmpdir(), "marv-model-availability-"));
    process.env.MARV_STATE_DIR = tempStateDir;
  });

  afterEach(() => {
    if (realStateDir === undefined) {
      delete process.env.MARV_STATE_DIR;
    } else {
      process.env.MARV_STATE_DIR = realStateDir;
    }
    fs.rmSync(tempStateDir, { recursive: true, force: true });
    vi.restoreAllMocks();
  });

  it("permanently removes models that are invalid or no longer accessible", () => {
    const status = markRuntimeModelFailure({
      ref: "openai/gpt-4.1",
      error: Object.assign(new Error("You do not have access to model gpt-4.1"), {
        status: 403,
      }),
    });

    expect(status).toBe("unsupported");
    expect(getRuntimeModelAvailability("openai/gpt-4.1")).toMatchObject({
      status: "unsupported",
    });
  });

  it("stores retryAfter for temporary limit failures", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000);

    const status = markRuntimeModelFailure({
      ref: "google/gemini-2.5-pro",
      error: Object.assign(new Error("429 quota exceeded"), { status: 429 }),
    });

    expect(status).toBe("temporary_unavailable");
    expect(getRuntimeModelAvailability("google/gemini-2.5-pro")).toMatchObject({
      status: "temporary_unavailable",
      retryAfter: 1_700_000_900_000,
    });
  });

  it("automatically clears expired cooldown entries so models can be retried", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000);
    markRuntimeModelFailure({
      ref: "anthropic/claude-sonnet-4-5",
      error: Object.assign(new Error("429 rate limit reached"), { status: 429 }),
    });

    vi.spyOn(Date, "now").mockReturnValue(1_700_000_900_001);
    expect(getRuntimeModelAvailability("anthropic/claude-sonnet-4-5")).toBeUndefined();
    expect(readRuntimeModelAvailability().models["anthropic/claude-sonnet-4-5"]).toBeUndefined();
  });
});
