import { t } from "../i18n/index.js";
import type { IconName } from "./icons.js";

export const NAV_TABS = [
  "overview",
  "operations",
  "channels",
  "agents",
  "workspace",
  "chat",
  "settings",
] as const;

export type Tab = (typeof NAV_TABS)[number];

export const OPERATIONS_SECTIONS = [
  "sessions",
  "instances",
  "usage",
  "cron",
  "logs",
  "debug",
] as const;
export type OperationsSection = (typeof OPERATIONS_SECTIONS)[number];

export const AGENTS_SECTIONS = ["agents", "skills", "nodes"] as const;
export type AgentsSection = (typeof AGENTS_SECTIONS)[number];

export const WORKSPACE_SECTIONS = [
  "projects",
  "workbench",
  "memory",
  "documents",
  "calendar",
] as const;
export type WorkspaceSection = (typeof WORKSPACE_SECTIONS)[number];

export const SETTINGS_SECTIONS = ["config"] as const;
export type SettingsSection = (typeof SETTINGS_SECTIONS)[number];

export type RouteState = {
  tab: Tab;
  operationsSection: OperationsSection;
  agentsSection: AgentsSection;
  workspaceSection: WorkspaceSection;
  settingsSection: SettingsSection;
  path: string;
};

const DEFAULT_ROUTE_STATE: RouteState = {
  tab: "overview",
  operationsSection: "sessions",
  agentsSection: "agents",
  workspaceSection: "projects",
  settingsSection: "config",
  path: "/overview",
};

const TAB_PATHS: Record<Tab, string> = {
  overview: "/overview",
  operations: "/operations",
  channels: "/channels",
  agents: "/agents",
  workspace: "/workspace",
  chat: "/chat",
  settings: "/settings",
};

const OPERATIONS_SECTION_PATHS: Record<OperationsSection, string> = {
  sessions: "/sessions",
  instances: "/instances",
  usage: "/usage",
  cron: "/cron",
  logs: "/logs",
  debug: "/debug",
};

const AGENTS_SECTION_PATHS: Record<AgentsSection, string> = {
  agents: "/agents",
  skills: "/skills",
  nodes: "/nodes",
};

const WORKSPACE_SECTION_PATHS: Record<WorkspaceSection, string> = {
  projects: "/projects",
  workbench: "/workbench",
  memory: "/memory",
  documents: "/documents",
  calendar: "/calendar",
};

const SETTINGS_SECTION_PATHS: Record<SettingsSection, string> = {
  config: "/config",
};

const PATH_TO_ROUTE = new Map<string, RouteState>([
  ["/", DEFAULT_ROUTE_STATE],
  ["/index.html", DEFAULT_ROUTE_STATE],
  ["/overview", { ...DEFAULT_ROUTE_STATE, tab: "overview", path: "/overview" }],
  ["/operations", { ...DEFAULT_ROUTE_STATE, tab: "operations", path: "/operations" }],
  ["/channels", { ...DEFAULT_ROUTE_STATE, tab: "channels", path: "/channels" }],
  ["/agents", { ...DEFAULT_ROUTE_STATE, tab: "agents", agentsSection: "agents", path: "/agents" }],
  [
    "/workspace",
    { ...DEFAULT_ROUTE_STATE, tab: "workspace", workspaceSection: "projects", path: "/workspace" },
  ],
  ["/chat", { ...DEFAULT_ROUTE_STATE, tab: "chat", path: "/chat" }],
  [
    "/settings",
    { ...DEFAULT_ROUTE_STATE, tab: "settings", settingsSection: "config", path: "/settings" },
  ],
  [
    "/sessions",
    { ...DEFAULT_ROUTE_STATE, tab: "operations", operationsSection: "sessions", path: "/sessions" },
  ],
  [
    "/instances",
    {
      ...DEFAULT_ROUTE_STATE,
      tab: "operations",
      operationsSection: "instances",
      path: "/instances",
    },
  ],
  [
    "/usage",
    { ...DEFAULT_ROUTE_STATE, tab: "operations", operationsSection: "usage", path: "/usage" },
  ],
  [
    "/cron",
    { ...DEFAULT_ROUTE_STATE, tab: "operations", operationsSection: "cron", path: "/cron" },
  ],
  [
    "/logs",
    { ...DEFAULT_ROUTE_STATE, tab: "operations", operationsSection: "logs", path: "/logs" },
  ],
  [
    "/debug",
    { ...DEFAULT_ROUTE_STATE, tab: "operations", operationsSection: "debug", path: "/debug" },
  ],
  ["/skills", { ...DEFAULT_ROUTE_STATE, tab: "agents", agentsSection: "skills", path: "/skills" }],
  ["/nodes", { ...DEFAULT_ROUTE_STATE, tab: "agents", agentsSection: "nodes", path: "/nodes" }],
  [
    "/projects",
    { ...DEFAULT_ROUTE_STATE, tab: "workspace", workspaceSection: "projects", path: "/projects" },
  ],
  [
    "/workbench",
    {
      ...DEFAULT_ROUTE_STATE,
      tab: "workspace",
      workspaceSection: "workbench",
      path: "/workbench",
    },
  ],
  [
    "/memory",
    { ...DEFAULT_ROUTE_STATE, tab: "workspace", workspaceSection: "memory", path: "/memory" },
  ],
  [
    "/documents",
    { ...DEFAULT_ROUTE_STATE, tab: "workspace", workspaceSection: "documents", path: "/documents" },
  ],
  [
    "/calendar",
    { ...DEFAULT_ROUTE_STATE, tab: "workspace", workspaceSection: "calendar", path: "/calendar" },
  ],
  [
    "/config",
    { ...DEFAULT_ROUTE_STATE, tab: "settings", settingsSection: "config", path: "/config" },
  ],
]);

export function normalizeBasePath(basePath: string): string {
  if (!basePath) {
    return "";
  }
  let base = basePath.trim();
  if (!base.startsWith("/")) {
    base = `/${base}`;
  }
  if (base === "/") {
    return "";
  }
  if (base.endsWith("/")) {
    base = base.slice(0, -1);
  }
  return base;
}

export function normalizePath(path: string): string {
  if (!path) {
    return "/";
  }
  let normalized = path.trim();
  if (!normalized.startsWith("/")) {
    normalized = `/${normalized}`;
  }
  if (normalized.length > 1 && normalized.endsWith("/")) {
    normalized = normalized.slice(0, -1);
  }
  return normalized;
}

export function pathForTab(tab: Tab, basePath = ""): string {
  const base = normalizeBasePath(basePath);
  const path = TAB_PATHS[tab];
  return base ? `${base}${path}` : path;
}

export function pathForOperationsSection(section: OperationsSection, basePath = ""): string {
  const base = normalizeBasePath(basePath);
  const path = OPERATIONS_SECTION_PATHS[section];
  return base ? `${base}${path}` : path;
}

export function pathForAgentsSection(section: AgentsSection, basePath = ""): string {
  const base = normalizeBasePath(basePath);
  const path = AGENTS_SECTION_PATHS[section];
  return base ? `${base}${path}` : path;
}

export function pathForWorkspaceSection(section: WorkspaceSection, basePath = ""): string {
  const base = normalizeBasePath(basePath);
  const path = WORKSPACE_SECTION_PATHS[section];
  return base ? `${base}${path}` : path;
}

export function pathForSettingsSection(section: SettingsSection, basePath = ""): string {
  const base = normalizeBasePath(basePath);
  const path = SETTINGS_SECTION_PATHS[section];
  return base ? `${base}${path}` : path;
}

export function tabFromPath(pathname: string, basePath = ""): Tab | null {
  return resolveRoute(pathname, basePath)?.tab ?? null;
}

export function resolveRoute(pathname: string, basePath = ""): RouteState | null {
  const base = normalizeBasePath(basePath);
  let path = pathname || "/";
  if (base) {
    if (path === base) {
      path = "/";
    } else if (path.startsWith(`${base}/`)) {
      path = path.slice(base.length);
    }
  }
  let normalized = normalizePath(path).toLowerCase();
  if (normalized.endsWith("/index.html")) {
    normalized = "/";
  }
  return PATH_TO_ROUTE.get(normalized) ?? null;
}

export function inferBasePathFromPathname(pathname: string): string {
  let normalized = normalizePath(pathname);
  if (normalized.endsWith("/index.html")) {
    normalized = normalizePath(normalized.slice(0, -"/index.html".length));
  }
  if (normalized === "/") {
    return "";
  }
  const segments = normalized.split("/").filter(Boolean);
  if (segments.length === 0) {
    return "";
  }
  for (let i = 0; i < segments.length; i++) {
    const candidate = `/${segments.slice(i).join("/")}`.toLowerCase();
    if (PATH_TO_ROUTE.has(candidate)) {
      const prefix = segments.slice(0, i);
      return prefix.length ? `/${prefix.join("/")}` : "";
    }
  }
  return `/${segments.join("/")}`;
}

export function iconForTab(tab: Tab): IconName {
  switch (tab) {
    case "overview":
      return "barChart";
    case "operations":
      return "radio";
    case "channels":
      return "link";
    case "agents":
      return "folder";
    case "workspace":
      return "book";
    case "chat":
      return "messageSquare";
    case "settings":
      return "settings";
    default:
      return "folder";
  }
}

export function titleForTab(tab: Tab) {
  return t(`tabs.${tab}`);
}

export function subtitleForTab(tab: Tab) {
  return t(`subtitles.${tab}`);
}

export function titleForOperationsSection(section: OperationsSection) {
  return t(`operationsSections.${section}`);
}

export function titleForAgentsSection(section: AgentsSection) {
  return t(`agentsSections.${section}`);
}

export function titleForWorkspaceSection(section: WorkspaceSection) {
  return t(`workspaceSections.${section}`);
}

export function titleForSettingsSection(section: SettingsSection) {
  return t(`settingsSections.${section}`);
}

export function isWorkspaceTab(tab: Tab): boolean {
  return tab === "workspace";
}

export function isLogsView(tab: Tab, operationsSection: OperationsSection): boolean {
  return tab === "operations" && operationsSection === "logs";
}

export function isDebugView(tab: Tab, operationsSection: OperationsSection): boolean {
  return tab === "operations" && operationsSection === "debug";
}

export function isCronView(tab: Tab, operationsSection: OperationsSection): boolean {
  return tab === "operations" && operationsSection === "cron";
}

export function isWorkbenchView(tab: Tab, workspaceSection: WorkspaceSection): boolean {
  return tab === "workspace" && workspaceSection === "workbench";
}
