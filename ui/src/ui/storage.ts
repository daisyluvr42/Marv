const KEY = "marv.control.settings.v2";

import { isSupportedLocale } from "../i18n/index.js";
import type {
  AgentsSection,
  OperationsSection,
  SettingsSection,
  WorkspaceSection,
} from "./navigation.js";
import type { ThemeMode } from "./theme.js";

export type UiSettings = {
  gatewayUrl: string;
  token: string;
  sessionKey: string;
  lastActiveSessionKey: string;
  theme: ThemeMode;
  chatFocusMode: boolean;
  chatShowThinking: boolean;
  splitRatio: number; // Sidebar split ratio (0.4 to 0.7, default 0.6)
  navCollapsed: boolean; // Collapsible sidebar state
  operationsSection: OperationsSection;
  agentsSection: AgentsSection;
  workspaceSection: WorkspaceSection;
  settingsSection: SettingsSection;
  locale?: string;
};

export function loadSettings(): UiSettings {
  const defaultUrl = (() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}`;
  })();

  const defaults: UiSettings = {
    gatewayUrl: defaultUrl,
    token: "",
    sessionKey: "main",
    lastActiveSessionKey: "main",
    theme: "system",
    chatFocusMode: false,
    chatShowThinking: true,
    splitRatio: 0.6,
    navCollapsed: false,
    operationsSection: "sessions",
    agentsSection: "agents",
    workspaceSection: "projects",
    settingsSection: "config",
  };

  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) {
      return defaults;
    }
    const parsed = JSON.parse(raw) as Partial<UiSettings>;
    return {
      gatewayUrl:
        typeof parsed.gatewayUrl === "string" && parsed.gatewayUrl.trim()
          ? parsed.gatewayUrl.trim()
          : defaults.gatewayUrl,
      // Shared gateway tokens are treated as bootstrap credentials and should
      // not be rehydrated from durable browser storage by default.
      token: defaults.token,
      sessionKey:
        typeof parsed.sessionKey === "string" && parsed.sessionKey.trim()
          ? parsed.sessionKey.trim()
          : defaults.sessionKey,
      lastActiveSessionKey:
        typeof parsed.lastActiveSessionKey === "string" && parsed.lastActiveSessionKey.trim()
          ? parsed.lastActiveSessionKey.trim()
          : (typeof parsed.sessionKey === "string" && parsed.sessionKey.trim()) ||
            defaults.lastActiveSessionKey,
      theme:
        parsed.theme === "light" || parsed.theme === "dark" || parsed.theme === "system"
          ? parsed.theme
          : defaults.theme,
      chatFocusMode:
        typeof parsed.chatFocusMode === "boolean" ? parsed.chatFocusMode : defaults.chatFocusMode,
      chatShowThinking:
        typeof parsed.chatShowThinking === "boolean"
          ? parsed.chatShowThinking
          : defaults.chatShowThinking,
      splitRatio:
        typeof parsed.splitRatio === "number" &&
        parsed.splitRatio >= 0.4 &&
        parsed.splitRatio <= 0.7
          ? parsed.splitRatio
          : defaults.splitRatio,
      navCollapsed:
        typeof parsed.navCollapsed === "boolean" ? parsed.navCollapsed : defaults.navCollapsed,
      operationsSection:
        parsed.operationsSection === "sessions" ||
        parsed.operationsSection === "instances" ||
        parsed.operationsSection === "usage" ||
        parsed.operationsSection === "cron" ||
        parsed.operationsSection === "logs" ||
        parsed.operationsSection === "debug"
          ? parsed.operationsSection
          : defaults.operationsSection,
      agentsSection:
        parsed.agentsSection === "agents" ||
        parsed.agentsSection === "skills" ||
        parsed.agentsSection === "nodes"
          ? parsed.agentsSection
          : defaults.agentsSection,
      workspaceSection:
        parsed.workspaceSection === "projects" ||
        parsed.workspaceSection === "memory" ||
        parsed.workspaceSection === "documents" ||
        parsed.workspaceSection === "calendar"
          ? parsed.workspaceSection
          : defaults.workspaceSection,
      settingsSection:
        parsed.settingsSection === "config" ? parsed.settingsSection : defaults.settingsSection,
      locale: isSupportedLocale(parsed.locale) ? parsed.locale : undefined,
    };
  } catch {
    return defaults;
  }
}

export function saveSettings(next: UiSettings) {
  const persisted: Omit<UiSettings, "token"> = {
    gatewayUrl: next.gatewayUrl,
    sessionKey: next.sessionKey,
    lastActiveSessionKey: next.lastActiveSessionKey,
    theme: next.theme,
    chatFocusMode: next.chatFocusMode,
    chatShowThinking: next.chatShowThinking,
    splitRatio: next.splitRatio,
    navCollapsed: next.navCollapsed,
    operationsSection: next.operationsSection,
    agentsSection: next.agentsSection,
    workspaceSection: next.workspaceSection,
    settingsSection: next.settingsSection,
    locale: next.locale,
  };
  localStorage.setItem(KEY, JSON.stringify(persisted));
}
