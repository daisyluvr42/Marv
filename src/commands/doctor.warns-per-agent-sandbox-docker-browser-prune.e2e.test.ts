import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { createDoctorRuntime, mockDoctorConfigSnapshot, note } from "./doctor.e2e-harness.js";

describe("doctor command", () => {
  it("does not warn for removed per-agent sandbox overrides", async () => {
    mockDoctorConfigSnapshot({
      config: {
        agents: {
          defaults: {
            sandbox: {
              mode: "all",
              scope: "shared",
            },
          },
          list: [
            {
              id: "work",
              workspace: "~/marv-work",
              sandbox: {
                mode: "all",
                scope: "shared",
                docker: {
                  setupCommand: "echo work",
                },
              },
            },
          ],
        },
      },
    });

    note.mockClear();

    const { doctorCommand } = await import("./doctor.js");
    await doctorCommand(createDoctorRuntime(), { nonInteractive: true });

    expect(note.mock.calls.some(([_, title]) => title === "Sandbox")).toBe(false);
  }, 30_000);

  it("does not warn when only the active workspace is present", async () => {
    mockDoctorConfigSnapshot({
      config: {
        agents: { defaults: { workspace: "/Users/monadlab/marv" } },
      },
    });

    note.mockClear();
    const homedirSpy = vi.spyOn(os, "homedir").mockReturnValue("/Users/monadlab");
    const realExists = fs.existsSync;
    const legacyPath = path.join("/Users/monadlab", "marv");
    const legacyAgentsPath = path.join(legacyPath, "AGENTS.md");
    const existsSpy = vi.spyOn(fs, "existsSync").mockImplementation((value) => {
      if (value === "/Users/monadlab/marv" || value === legacyPath || value === legacyAgentsPath) {
        return true;
      }
      return realExists(value as never);
    });

    const { doctorCommand } = await import("./doctor.js");
    await doctorCommand(createDoctorRuntime(), { nonInteractive: true });

    expect(note.mock.calls.some(([_, title]) => title === "Extra workspace")).toBe(false);

    homedirSpy.mockRestore();
    existsSpy.mockRestore();
  });
});
