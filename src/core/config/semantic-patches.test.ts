import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  compileSemanticPatch,
  createSemanticConfigRevision,
  createSemanticPatchProposal,
  getSemanticConfigRevision,
  getSemanticPatchProposal,
  listSemanticConfigRevisions,
  updateSemanticConfigRevisionStatus,
  updateSemanticPatchProposalStatus,
} from "./semantic-patches.js";

let stateDir = "";
let prevStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-semantic-patch-"));
  prevStateDir = process.env.MARV_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (prevStateDir == null) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = prevStateDir;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
});

describe("semantic patch compilation", () => {
  it("maps concise intent to low thinking default", () => {
    const compiled = compileSemanticPatch("请更简洁一点");
    expect(compiled.riskLevel).toBe("L1");
    expect(compiled.needsApproval).toBe(false);
    expect(compiled.patch).toMatchObject({
      agents: {
        defaults: {
          thinkingDefault: "low",
        },
      },
    });
  });

  it("accepts explicit JSON object patch", () => {
    const compiled = compileSemanticPatch('{"tools":{"fs":{"workspaceOnly":true}}}');
    expect(compiled.riskLevel).toBe("L2");
    expect(compiled.needsApproval).toBe(true);
    expect(compiled.patch).toMatchObject({
      tools: {
        fs: {
          workspaceOnly: true,
        },
      },
    });
  });
});

describe("semantic patch persistence", () => {
  it("creates proposal and revision records", () => {
    const proposal = createSemanticPatchProposal({
      scopeType: "global",
      scopeId: "gateway",
      naturalLanguage: "请更简洁一点",
      actorId: "tester",
      nowMs: 123,
    });
    expect(proposal.proposalId).toMatch(/^pp_/);
    expect(proposal.status).toBe("open");

    const loadedProposal = getSemanticPatchProposal(proposal.proposalId);
    expect(loadedProposal?.naturalLanguage).toBe("请更简洁一点");

    const committedProposal = updateSemanticPatchProposalStatus({
      proposalId: proposal.proposalId,
      status: "committed",
    });
    expect(committedProposal?.status).toBe("committed");

    const revision = createSemanticConfigRevision({
      proposalId: proposal.proposalId,
      scopeType: proposal.scopeType,
      scopeId: proposal.scopeId,
      actorId: "tester",
      patch: proposal.patch,
      explanation: proposal.explanation,
      riskLevel: proposal.riskLevel,
      nowMs: 456,
      beforeConfig: { commands: { restart: false } },
      afterConfig: { commands: { restart: true } },
    });
    expect(revision.revision).toMatch(/^rev_/);
    expect(revision.status).toBe("committed");

    const loadedRevision = getSemanticConfigRevision(revision.revision);
    expect(loadedRevision?.beforeConfig).toMatchObject({
      commands: {
        restart: false,
      },
    });

    const rolledBack = updateSemanticConfigRevisionStatus({
      revision: revision.revision,
      status: "rolled_back",
    });
    expect(rolledBack?.status).toBe("rolled_back");

    const listed = listSemanticConfigRevisions({ scopeType: "global", scopeId: "gateway" });
    expect(listed.length).toBeGreaterThanOrEqual(1);
  });
});
