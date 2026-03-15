import { refreshChat } from "./app-chat.js";
import {
  startLogsPolling,
  stopLogsPolling,
  startDebugPolling,
  stopDebugPolling,
} from "./app-polling.js";
import { scheduleChatScroll, scheduleLogsScroll } from "./app-scroll.js";
import type { MarvApp } from "./app.js";
import { loadAgentFiles } from "./controllers/agent-files.js";
import { loadAgentIdentities, loadAgentIdentity } from "./controllers/agent-identity.js";
import { loadAgentSkills } from "./controllers/agent-skills.js";
import { loadAgents } from "./controllers/agents.js";
import { loadWorkspaceCalendar } from "./controllers/calendar.js";
import { loadChannels } from "./controllers/channels.js";
import { loadConfig, loadConfigSchema } from "./controllers/config.js";
import { loadCronJobs, loadCronStatus } from "./controllers/cron.js";
import { loadDashboard } from "./controllers/dashboard.js";
import { loadDebug } from "./controllers/debug.js";
import { loadDevices } from "./controllers/devices.js";
import { loadWorkspaceDocuments } from "./controllers/documents.js";
import { loadExecApprovals } from "./controllers/exec-approvals.js";
import { loadLogs } from "./controllers/logs.js";
import { loadWorkspaceMemory, searchWorkspaceMemory } from "./controllers/memory.js";
import { loadNodes } from "./controllers/nodes.js";
import { loadPresence } from "./controllers/presence.js";
import { loadWorkspaceProjects } from "./controllers/projects.js";
import { loadSessions } from "./controllers/sessions.js";
import { loadSkills } from "./controllers/skills.js";
import { loadWorkspaceSummary } from "./controllers/workspace-summary.js";
import {
  inferBasePathFromPathname,
  isDebugView,
  isLogsView,
  normalizeBasePath,
  normalizePath,
  pathForAgentsSection,
  pathForOperationsSection,
  pathForSettingsSection,
  pathForTab,
  pathForWorkspaceSection,
  resolveRoute,
  type AgentsSection,
  type OperationsSection,
  type SettingsSection,
  type Tab,
  type WorkspaceSection,
} from "./navigation.js";
import { saveSettings, type UiSettings } from "./storage.js";
import { startThemeTransition, type ThemeTransitionContext } from "./theme-transition.js";
import { resolveTheme, type ResolvedTheme, type ThemeMode } from "./theme.js";
import type { AgentsListResult } from "./types.js";

type SettingsHost = {
  settings: UiSettings;
  password?: string;
  theme: ThemeMode;
  themeResolved: ResolvedTheme;
  applySessionKey: string;
  sessionKey: string;
  tab: Tab;
  operationsSection: OperationsSection;
  agentsSection: AgentsSection;
  workspaceSection: WorkspaceSection;
  settingsSection: SettingsSection;
  connected: boolean;
  chatHasAutoScrolled: boolean;
  logsAtBottom: boolean;
  eventLog: unknown[];
  eventLogBuffer: unknown[];
  basePath: string;
  agentsList?: AgentsListResult | null;
  agentsSelectedId?: string | null;
  agentsPanel?: "overview" | "p0" | "files" | "tools" | "skills" | "channels" | "cron";
  themeMedia: MediaQueryList | null;
  themeMediaHandler: ((event: MediaQueryListEvent) => void) | null;
  pendingGatewayUrl?: string | null;
};

function updatePollingForActiveView(host: SettingsHost) {
  if (isLogsView(host.tab, host.operationsSection)) {
    startLogsPolling(host as unknown as Parameters<typeof startLogsPolling>[0]);
  } else {
    stopLogsPolling(host as unknown as Parameters<typeof stopLogsPolling>[0]);
  }
  if (isDebugView(host.tab, host.operationsSection)) {
    startDebugPolling(host as unknown as Parameters<typeof startDebugPolling>[0]);
  } else {
    stopDebugPolling(host as unknown as Parameters<typeof stopDebugPolling>[0]);
  }
}

function applySectionSettings(host: SettingsHost) {
  applySettings(host, {
    ...host.settings,
    operationsSection: host.operationsSection,
    agentsSection: host.agentsSection,
    workspaceSection: host.workspaceSection,
    settingsSection: host.settingsSection,
  });
}

function pathForCurrentView(host: SettingsHost) {
  switch (host.tab) {
    case "operations":
      return pathForOperationsSection(host.operationsSection, host.basePath);
    case "agents":
      return pathForAgentsSection(host.agentsSection, host.basePath);
    case "workspace":
      return pathForWorkspaceSection(host.workspaceSection, host.basePath);
    case "settings":
      return pathForSettingsSection(host.settingsSection, host.basePath);
    default:
      return pathForTab(host.tab, host.basePath);
  }
}

export function applySettings(host: SettingsHost, next: UiSettings) {
  const normalized = {
    ...next,
    lastActiveSessionKey: next.lastActiveSessionKey?.trim() || next.sessionKey.trim() || "main",
  };
  host.settings = normalized;
  host.operationsSection = normalized.operationsSection;
  host.agentsSection = normalized.agentsSection;
  host.workspaceSection = normalized.workspaceSection;
  host.settingsSection = normalized.settingsSection;
  saveSettings(normalized);
  if (next.theme !== host.theme) {
    host.theme = next.theme;
    applyResolvedTheme(host, resolveTheme(next.theme));
  }
  host.applySessionKey = host.settings.lastActiveSessionKey;
}

export function setLastActiveSessionKey(host: SettingsHost, next: string) {
  const trimmed = next.trim();
  if (!trimmed) {
    return;
  }
  if (host.settings.lastActiveSessionKey === trimmed) {
    return;
  }
  applySettings(host, { ...host.settings, lastActiveSessionKey: trimmed });
}

export function applySettingsFromUrl(host: SettingsHost) {
  if (!window.location.search && !window.location.hash) {
    return;
  }
  const url = new URL(window.location.href);
  const params = new URLSearchParams(url.search);
  const hashParams = new URLSearchParams(url.hash.startsWith("#") ? url.hash.slice(1) : url.hash);

  const tokenRaw = params.get("token") ?? hashParams.get("token");
  const passwordRaw = params.get("password") ?? hashParams.get("password");
  const sessionRaw = params.get("session") ?? hashParams.get("session");
  const gatewayUrlRaw = params.get("gatewayUrl") ?? hashParams.get("gatewayUrl");
  let shouldCleanUrl = false;

  if (tokenRaw != null) {
    const token = tokenRaw.trim();
    if (token && token !== host.settings.token) {
      applySettings(host, { ...host.settings, token });
    }
    params.delete("token");
    hashParams.delete("token");
    shouldCleanUrl = true;
  }

  if (passwordRaw != null) {
    params.delete("password");
    hashParams.delete("password");
    shouldCleanUrl = true;
  }

  if (sessionRaw != null) {
    const session = sessionRaw.trim();
    if (session) {
      host.sessionKey = session;
      applySettings(host, {
        ...host.settings,
        sessionKey: session,
        lastActiveSessionKey: session,
      });
    }
  }

  if (gatewayUrlRaw != null) {
    const gatewayUrl = gatewayUrlRaw.trim();
    if (gatewayUrl && gatewayUrl !== host.settings.gatewayUrl) {
      host.pendingGatewayUrl = gatewayUrl;
    }
    params.delete("gatewayUrl");
    hashParams.delete("gatewayUrl");
    shouldCleanUrl = true;
  }

  if (!shouldCleanUrl) {
    return;
  }
  url.search = params.toString();
  const nextHash = hashParams.toString();
  url.hash = nextHash ? `#${nextHash}` : "";
  window.history.replaceState({}, "", url.toString());
}

export function setTab(host: SettingsHost, next: Tab) {
  if (host.tab !== next) {
    host.tab = next;
  }
  if (next === "chat") {
    host.chatHasAutoScrolled = false;
  }
  updatePollingForActiveView(host);
  void refreshActiveTab(host);
  syncUrlWithTab(host, next, false);
}

export function selectOperationsSection(host: SettingsHost, next: OperationsSection) {
  host.tab = "operations";
  host.operationsSection = next;
  applySectionSettings(host);
  updatePollingForActiveView(host);
  void refreshActiveTab(host);
  syncUrlWithCurrentView(host, false);
}

export function selectAgentsSection(host: SettingsHost, next: AgentsSection) {
  host.tab = "agents";
  host.agentsSection = next;
  applySectionSettings(host);
  updatePollingForActiveView(host);
  void refreshActiveTab(host);
  syncUrlWithCurrentView(host, false);
}

export function selectWorkspaceSection(host: SettingsHost, next: WorkspaceSection) {
  host.tab = "workspace";
  host.workspaceSection = next;
  applySectionSettings(host);
  updatePollingForActiveView(host);
  void refreshActiveTab(host);
  syncUrlWithCurrentView(host, false);
}

export function selectSettingsSection(host: SettingsHost, next: SettingsSection) {
  host.tab = "settings";
  host.settingsSection = next;
  applySectionSettings(host);
  updatePollingForActiveView(host);
  void refreshActiveTab(host);
  syncUrlWithCurrentView(host, false);
}

export function setTheme(host: SettingsHost, next: ThemeMode, context?: ThemeTransitionContext) {
  const applyTheme = () => {
    host.theme = next;
    applySettings(host, { ...host.settings, theme: next });
    applyResolvedTheme(host, resolveTheme(next));
  };
  startThemeTransition({
    nextTheme: next,
    applyTheme,
    context,
    currentTheme: host.theme,
  });
}

export async function refreshActiveTab(host: SettingsHost) {
  if (host.tab === "overview") {
    await loadOverview(host);
    return;
  }
  if (host.tab === "channels") {
    await loadChannelsTab(host);
    return;
  }
  if (host.tab === "operations") {
    if (host.operationsSection === "instances") {
      await loadPresence(host as unknown as MarvApp);
      return;
    }
    if (host.operationsSection === "sessions") {
      await loadSessions(host as unknown as MarvApp);
      return;
    }
    if (host.operationsSection === "cron") {
      await loadCron(host);
      return;
    }
    if (host.operationsSection === "usage") {
      return;
    }
    if (host.operationsSection === "debug") {
      await loadDebug(host as unknown as MarvApp);
      host.eventLog = host.eventLogBuffer;
      return;
    }
    if (host.operationsSection === "logs") {
      host.logsAtBottom = true;
      await loadLogs(host as unknown as MarvApp, { reset: true });
      scheduleLogsScroll(host as unknown as Parameters<typeof scheduleLogsScroll>[0], true);
    }
    return;
  }
  if (host.tab === "workspace") {
    await loadWorkspaceSummary(host as unknown as Parameters<typeof loadWorkspaceSummary>[0]);
    if (host.workspaceSection === "projects") {
      await loadWorkspaceProjects(host as unknown as Parameters<typeof loadWorkspaceProjects>[0]);
      return;
    }
    if (host.workspaceSection === "calendar") {
      await loadWorkspaceCalendar(host as unknown as Parameters<typeof loadWorkspaceCalendar>[0]);
      return;
    }
    if (host.workspaceSection === "memory") {
      if (
        "workspaceMemoryQuery" in host &&
        typeof host.workspaceMemoryQuery === "string" &&
        host.workspaceMemoryQuery.trim()
      ) {
        await searchWorkspaceMemory(host as unknown as Parameters<typeof searchWorkspaceMemory>[0]);
      } else {
        await loadWorkspaceMemory(host as unknown as Parameters<typeof loadWorkspaceMemory>[0]);
      }
      return;
    }
    if (host.workspaceSection === "documents") {
      await loadWorkspaceDocuments(host as unknown as Parameters<typeof loadWorkspaceDocuments>[0]);
    }
    return;
  }
  if (host.tab === "agents") {
    if (host.agentsSection === "skills") {
      await loadSkills(host as unknown as MarvApp);
      return;
    }
    if (host.agentsSection === "nodes") {
      await loadNodes(host as unknown as MarvApp);
      await loadDevices(host as unknown as MarvApp);
      await loadConfig(host as unknown as MarvApp);
      await loadExecApprovals(host as unknown as MarvApp);
      return;
    }
    await loadAgents(host as unknown as MarvApp);
    await loadConfig(host as unknown as MarvApp);
    const agentIds = host.agentsList?.agents?.map((entry) => entry.id) ?? [];
    if (agentIds.length > 0) {
      void loadAgentIdentities(host as unknown as MarvApp, agentIds);
    }
    const agentId =
      host.agentsSelectedId ?? host.agentsList?.defaultId ?? host.agentsList?.agents?.[0]?.id;
    if (agentId) {
      void loadAgentIdentity(host as unknown as MarvApp, agentId);
      if (host.agentsPanel === "skills") {
        void loadAgentSkills(host as unknown as MarvApp, agentId);
      }
      if (host.agentsPanel === "p0" || host.agentsPanel === "files") {
        void loadAgentFiles(host as unknown as MarvApp, agentId);
      }
      if (host.agentsPanel === "channels") {
        void loadChannels(host as unknown as MarvApp, false);
      }
      if (host.agentsPanel === "cron") {
        void loadCron(host);
      }
    }
    return;
  }
  if (host.tab === "chat") {
    await refreshChat(host as unknown as Parameters<typeof refreshChat>[0]);
    scheduleChatScroll(
      host as unknown as Parameters<typeof scheduleChatScroll>[0],
      !host.chatHasAutoScrolled,
    );
    return;
  }
  if (host.tab === "settings") {
    await loadConfigSchema(host as unknown as MarvApp);
    await loadConfig(host as unknown as MarvApp);
  }
}

export function inferBasePath() {
  if (typeof window === "undefined") {
    return "";
  }
  const configured = window.__MARV_CONTROL_UI_BASE_PATH__;
  if (typeof configured === "string" && configured.trim()) {
    return normalizeBasePath(configured);
  }
  return inferBasePathFromPathname(window.location.pathname);
}

export function syncThemeWithSettings(host: SettingsHost) {
  host.theme = host.settings.theme ?? "system";
  applyResolvedTheme(host, resolveTheme(host.theme));
}

export function applyResolvedTheme(host: SettingsHost, resolved: ResolvedTheme) {
  host.themeResolved = resolved;
  if (typeof document === "undefined") {
    return;
  }
  const root = document.documentElement;
  root.dataset.theme = resolved;
  root.style.colorScheme = resolved;
}

export function attachThemeListener(host: SettingsHost) {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return;
  }
  host.themeMedia = window.matchMedia("(prefers-color-scheme: dark)");
  host.themeMediaHandler = (event) => {
    if (host.theme !== "system") {
      return;
    }
    applyResolvedTheme(host, event.matches ? "dark" : "light");
  };
  if (typeof host.themeMedia.addEventListener === "function") {
    host.themeMedia.addEventListener("change", host.themeMediaHandler);
    return;
  }
  const legacy = host.themeMedia as MediaQueryList & {
    addListener: (cb: (event: MediaQueryListEvent) => void) => void;
  };
  legacy.addListener(host.themeMediaHandler);
}

export function detachThemeListener(host: SettingsHost) {
  if (!host.themeMedia || !host.themeMediaHandler) {
    return;
  }
  if (typeof host.themeMedia.removeEventListener === "function") {
    host.themeMedia.removeEventListener("change", host.themeMediaHandler);
    return;
  }
  const legacy = host.themeMedia as MediaQueryList & {
    removeListener: (cb: (event: MediaQueryListEvent) => void) => void;
  };
  legacy.removeListener(host.themeMediaHandler);
  host.themeMedia = null;
  host.themeMediaHandler = null;
}

export function syncTabWithLocation(host: SettingsHost, replace: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  const resolved = resolveRoute(window.location.pathname, host.basePath) ?? {
    tab: "overview" as const,
    operationsSection: host.settings.operationsSection,
    agentsSection: host.settings.agentsSection,
    workspaceSection: host.settings.workspaceSection,
    settingsSection: host.settings.settingsSection,
    path: "/overview",
  };
  setTabFromRoute(host, resolved);
  syncUrlWithCurrentView(host, replace, resolved.path);
}

export function onPopState(host: SettingsHost) {
  if (typeof window === "undefined") {
    return;
  }
  const resolved = resolveRoute(window.location.pathname, host.basePath);
  if (!resolved) {
    return;
  }

  const url = new URL(window.location.href);
  const session = url.searchParams.get("session")?.trim();
  if (session) {
    host.sessionKey = session;
    applySettings(host, {
      ...host.settings,
      sessionKey: session,
      lastActiveSessionKey: session,
    });
  }

  setTabFromRoute(host, resolved);
}

export function setTabFromRoute(
  host: SettingsHost,
  next: {
    tab: Tab;
    operationsSection: OperationsSection;
    agentsSection: AgentsSection;
    workspaceSection: WorkspaceSection;
    settingsSection: SettingsSection;
  },
) {
  host.tab = next.tab;
  host.operationsSection = next.operationsSection;
  host.agentsSection = next.agentsSection;
  host.workspaceSection = next.workspaceSection;
  host.settingsSection = next.settingsSection;
  host.settings = {
    ...host.settings,
    operationsSection: next.operationsSection,
    agentsSection: next.agentsSection,
    workspaceSection: next.workspaceSection,
    settingsSection: next.settingsSection,
  };
  if (next.tab === "chat") {
    host.chatHasAutoScrolled = false;
  }
  updatePollingForActiveView(host);
  if (host.connected) {
    void refreshActiveTab(host);
  }
}

export function syncUrlWithCurrentView(
  host: SettingsHost,
  replace: boolean,
  pathOverride?: string,
) {
  if (typeof window === "undefined") {
    return;
  }
  const targetPath = normalizePath(
    pathOverride ? `${normalizeBasePath(host.basePath)}${pathOverride}` : pathForCurrentView(host),
  );
  const currentPath = normalizePath(window.location.pathname);
  const url = new URL(window.location.href);

  if (host.tab === "chat" && host.sessionKey) {
    url.searchParams.set("session", host.sessionKey);
  } else {
    url.searchParams.delete("session");
  }

  if (currentPath !== targetPath) {
    url.pathname = targetPath;
  }

  if (replace) {
    window.history.replaceState({}, "", url.toString());
  } else {
    window.history.pushState({}, "", url.toString());
  }
}

export function syncUrlWithTab(host: SettingsHost, tab: Tab, replace: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.pathname = normalizePath(pathForTab(tab, host.basePath));
  if (tab === "chat" && host.sessionKey) {
    url.searchParams.set("session", host.sessionKey);
  } else {
    url.searchParams.delete("session");
  }
  if (replace) {
    window.history.replaceState({}, "", url.toString());
  } else {
    window.history.pushState({}, "", url.toString());
  }
}

export function syncUrlWithSessionKey(host: SettingsHost, sessionKey: string, replace: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("session", sessionKey);
  if (replace) {
    window.history.replaceState({}, "", url.toString());
  } else {
    window.history.pushState({}, "", url.toString());
  }
}

export async function loadOverview(host: SettingsHost) {
  await Promise.all([
    loadChannels(host as unknown as MarvApp, false),
    loadPresence(host as unknown as MarvApp),
    loadSessions(host as unknown as MarvApp),
    loadCronStatus(host as unknown as MarvApp),
    loadDashboard(host as unknown as MarvApp),
    loadDebug(host as unknown as MarvApp),
  ]);
}

export async function loadChannelsTab(host: SettingsHost) {
  await Promise.all([
    loadChannels(host as unknown as MarvApp, true),
    loadConfigSchema(host as unknown as MarvApp),
    loadConfig(host as unknown as MarvApp),
  ]);
}

export async function loadCron(host: SettingsHost) {
  await Promise.all([
    loadChannels(host as unknown as MarvApp, false),
    loadCronStatus(host as unknown as MarvApp),
    loadCronJobs(host as unknown as MarvApp),
  ]);
}
