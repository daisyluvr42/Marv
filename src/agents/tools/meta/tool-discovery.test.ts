import { describe, expect, it } from "vitest";
import type { SkillEntry } from "../../skills.js";
import { ToolDiscoveryService } from "./tool-discovery.js";

function makeEntry(params: {
  name: string;
  description: string;
  source: string;
  requiresBins?: string[];
  installBins?: string[];
  frontmatterDescription?: string;
}): SkillEntry {
  return {
    skill: {
      name: params.name,
      description: params.description,
      source: params.source,
      filePath: `/tmp/${params.name}/SKILL.md`,
      baseDir: `/tmp/${params.name}`,
    },
    frontmatter: params.frontmatterDescription
      ? { description: params.frontmatterDescription }
      : {},
    metadata: {
      requires: params.requiresBins ? { bins: params.requiresBins } : undefined,
      install: params.installBins
        ? [
            {
              kind: "node",
              bins: params.installBins,
            },
          ]
        : undefined,
    },
  } as unknown as SkillEntry;
}

describe("ToolDiscoveryService", () => {
  it("discovers best matching skills by capability description", () => {
    const service = new ToolDiscoveryService();
    const entries = [
      makeEntry({
        name: "github-repos",
        description: "Search GitHub repositories and issues",
        source: "marv-managed",
        requiresBins: ["gh"],
        installBins: ["gh"],
      }),
      makeEntry({
        name: "pdf-reader",
        description: "Read PDF files",
        source: "marv-bundled",
      }),
    ];

    const discovered = service.discover({
      workspaceDir: "/tmp/workspace",
      capability: {
        description: "search github repositories",
        suggestedTools: ["gh", "github"],
      },
      entries,
    });

    expect(discovered.length).toBeGreaterThanOrEqual(1);
    expect(discovered[0]?.skillId).toBe("github-repos");
    expect(discovered[0]?.source).toBe("managed");
  });

  it("respects discovery scope from autonomy config", () => {
    const service = new ToolDiscoveryService();
    const entries = [
      makeEntry({
        name: "one",
        description: "GitHub helper",
        source: "marv-managed",
      }),
      makeEntry({
        name: "two",
        description: "GitHub helper",
        source: "marv-bundled",
      }),
    ];

    const discovered = service.discover({
      workspaceDir: "/tmp/workspace",
      capability: {
        description: "github helper",
      },
      config: {
        autonomy: {
          discovery: {
            scope: "bundled",
          },
        },
      },
      entries,
    });

    expect(discovered).toHaveLength(1);
    expect(discovered[0]?.skillId).toBe("two");
    expect(discovered[0]?.source).toBe("bundled");
  });

  it("returns empty when discovery is disabled", () => {
    const service = new ToolDiscoveryService();
    const entries = [
      makeEntry({
        name: "github-repos",
        description: "Search GitHub repositories",
        source: "marv-managed",
      }),
    ];

    const discovered = service.discover({
      workspaceDir: "/tmp/workspace",
      capability: { description: "github" },
      config: {
        autonomy: {
          discovery: {
            enabled: false,
          },
        },
      },
      entries,
    });

    expect(discovered).toEqual([]);
  });

  it("skips quarantined skills", () => {
    const service = new ToolDiscoveryService();
    const entries = [
      makeEntry({
        name: "github-repos",
        description: "Search GitHub repositories",
        source: "marv-managed",
      }),
      makeEntry({
        name: "pdf-reader",
        description: "Read PDF files",
        source: "marv-bundled",
      }),
    ];

    const discovered = service.discover({
      workspaceDir: "/tmp/workspace",
      capability: { description: "github" },
      entries,
      usageRecords: {
        "github-repos": {
          skillId: "github-repos",
          installedAt: Date.now(),
          successCount: 0,
          failureCount: 0,
          ok: false,
          quarantined: true,
        },
      },
    });

    expect(discovered).toHaveLength(0);
  });
});
