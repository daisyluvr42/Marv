import { describe, expect, it } from "vitest";
import {
  AGENTS_SECTIONS,
  iconForTab,
  inferBasePathFromPathname,
  NAV_TABS,
  normalizeBasePath,
  normalizePath,
  OPERATIONS_SECTIONS,
  pathForAgentsSection,
  pathForOperationsSection,
  pathForTab,
  pathForWorkspaceSection,
  resolveRoute,
  subtitleForTab,
  tabFromPath,
  titleForTab,
  WORKSPACE_SECTIONS,
  type Tab,
} from "./navigation.js";

describe("iconForTab", () => {
  it("returns a non-empty string for every tab", () => {
    for (const tab of NAV_TABS) {
      const icon = iconForTab(tab);
      expect(icon).toBeTruthy();
      expect(typeof icon).toBe("string");
      expect(icon.length).toBeGreaterThan(0);
    }
  });

  it("returns stable icons for known tabs", () => {
    expect(iconForTab("overview")).toBe("barChart");
    expect(iconForTab("operations")).toBe("radio");
    expect(iconForTab("channels")).toBe("link");
    expect(iconForTab("agents")).toBe("folder");
    expect(iconForTab("workspace")).toBe("book");
    expect(iconForTab("chat")).toBe("messageSquare");
    expect(iconForTab("settings")).toBe("settings");
  });

  it("returns a fallback icon for unknown tab", () => {
    const unknownTab = "unknown" as Tab;
    expect(iconForTab(unknownTab)).toBe("folder");
  });
});

describe("titleForTab", () => {
  it("returns a non-empty string for every tab", () => {
    for (const tab of NAV_TABS) {
      const title = titleForTab(tab);
      expect(title).toBeTruthy();
      expect(typeof title).toBe("string");
    }
  });

  it("returns expected titles", () => {
    expect(titleForTab("overview")).toBe("Overview");
    expect(titleForTab("operations")).toBe("Operations");
    expect(titleForTab("workspace")).toBe("Workspace");
  });
});

describe("subtitleForTab", () => {
  it("returns a string for every tab", () => {
    for (const tab of NAV_TABS) {
      const subtitle = subtitleForTab(tab);
      expect(typeof subtitle).toBe("string");
    }
  });

  it("returns descriptive subtitles", () => {
    expect(subtitleForTab("chat")).toContain("chat session");
    expect(subtitleForTab("settings")).toContain("configuration");
    expect(subtitleForTab("workspace")).toContain("Projects");
  });
});

describe("normalizeBasePath", () => {
  it("returns empty string for falsy input", () => {
    expect(normalizeBasePath("")).toBe("");
  });

  it("adds leading slash if missing", () => {
    expect(normalizeBasePath("ui")).toBe("/ui");
  });

  it("removes trailing slash", () => {
    expect(normalizeBasePath("/ui/")).toBe("/ui");
  });

  it("returns empty string for root path", () => {
    expect(normalizeBasePath("/")).toBe("");
  });
});

describe("normalizePath", () => {
  it("returns / for falsy input", () => {
    expect(normalizePath("")).toBe("/");
  });

  it("adds leading slash if missing", () => {
    expect(normalizePath("chat")).toBe("/chat");
  });

  it("removes trailing slash except for root", () => {
    expect(normalizePath("/chat/")).toBe("/chat");
    expect(normalizePath("/")).toBe("/");
  });
});

describe("path helpers", () => {
  it("returns correct top-level paths", () => {
    expect(pathForTab("overview")).toBe("/overview");
    expect(pathForTab("operations")).toBe("/operations");
    expect(pathForTab("workspace")).toBe("/workspace");
  });

  it("returns correct section paths", () => {
    expect(pathForOperationsSection("sessions")).toBe("/sessions");
    expect(pathForAgentsSection("skills")).toBe("/skills");
    expect(pathForWorkspaceSection("documents")).toBe("/documents");
  });

  it("prepends base paths", () => {
    expect(pathForTab("channels", "/ui")).toBe("/ui/channels");
    expect(pathForOperationsSection("cron", "/apps/marv")).toBe("/apps/marv/cron");
  });
});

describe("resolveRoute", () => {
  it("maps root and top-level paths", () => {
    expect(resolveRoute("/overview")?.tab).toBe("overview");
    expect(resolveRoute("/operations")?.tab).toBe("operations");
    expect(resolveRoute("/")?.tab).toBe("overview");
  });

  it("maps legacy detail paths to parent tabs", () => {
    expect(resolveRoute("/sessions")).toMatchObject({
      tab: "operations",
      operationsSection: "sessions",
    });
    expect(resolveRoute("/skills")).toMatchObject({
      tab: "agents",
      agentsSection: "skills",
    });
    expect(resolveRoute("/projects")).toMatchObject({
      tab: "workspace",
      workspaceSection: "projects",
    });
    expect(resolveRoute("/config")).toMatchObject({
      tab: "settings",
      settingsSection: "config",
    });
  });

  it("handles base paths", () => {
    expect(resolveRoute("/ui/cron", "/ui")).toMatchObject({
      tab: "operations",
      operationsSection: "cron",
    });
    expect(resolveRoute("/apps/marv/projects", "/apps/marv")).toMatchObject({
      tab: "workspace",
      workspaceSection: "projects",
    });
  });
});

describe("tabFromPath", () => {
  it("returns parent tabs for current paths", () => {
    expect(tabFromPath("/chat")).toBe("chat");
    expect(tabFromPath("/overview")).toBe("overview");
    expect(tabFromPath("/sessions")).toBe("operations");
    expect(tabFromPath("/projects")).toBe("workspace");
  });

  it("returns overview for root path", () => {
    expect(tabFromPath("/")).toBe("overview");
  });

  it("returns null for unknown path", () => {
    expect(tabFromPath("/unknown")).toBeNull();
  });
});

describe("inferBasePathFromPathname", () => {
  it("returns empty string for root", () => {
    expect(inferBasePathFromPathname("/")).toBe("");
  });

  it("returns empty string for direct tab path", () => {
    expect(inferBasePathFromPathname("/overview")).toBe("");
    expect(inferBasePathFromPathname("/operations")).toBe("");
    expect(inferBasePathFromPathname("/workspace")).toBe("");
  });

  it("infers base path from nested paths", () => {
    expect(inferBasePathFromPathname("/ui/cron")).toBe("/ui");
    expect(inferBasePathFromPathname("/apps/marv/projects")).toBe("/apps/marv");
  });
});

describe("section constants", () => {
  it("keep sections unique", () => {
    expect(new Set(OPERATIONS_SECTIONS).size).toBe(OPERATIONS_SECTIONS.length);
    expect(new Set(AGENTS_SECTIONS).size).toBe(AGENTS_SECTIONS.length);
    expect(new Set(WORKSPACE_SECTIONS).size).toBe(WORKSPACE_SECTIONS.length);
  });
});
