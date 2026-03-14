import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setTabFromRoute } from "./app-settings.js";
import type { Tab } from "./navigation.js";

type SettingsHost = Parameters<typeof setTabFromRoute>[0] & {
  logsPollInterval: number | null;
  debugPollInterval: number | null;
};

const createHost = (tab: Tab): SettingsHost => ({
  settings: {
    gatewayUrl: "",
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
  },
  theme: "system",
  themeResolved: "dark",
  applySessionKey: "main",
  sessionKey: "main",
  tab,
  operationsSection: "sessions",
  agentsSection: "agents",
  workspaceSection: "projects",
  settingsSection: "config",
  connected: false,
  chatHasAutoScrolled: false,
  logsAtBottom: false,
  eventLog: [],
  eventLogBuffer: [],
  basePath: "",
  themeMedia: null,
  themeMediaHandler: null,
  logsPollInterval: null,
  debugPollInterval: null,
});

describe("setTabFromRoute", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts and stops log polling based on the active operations section", () => {
    const host = createHost("chat");

    setTabFromRoute(host, {
      tab: "operations",
      operationsSection: "logs",
      agentsSection: "agents",
      workspaceSection: "projects",
      settingsSection: "config",
    });
    expect(host.logsPollInterval).not.toBeNull();
    expect(host.debugPollInterval).toBeNull();

    setTabFromRoute(host, {
      tab: "chat",
      operationsSection: "sessions",
      agentsSection: "agents",
      workspaceSection: "projects",
      settingsSection: "config",
    });
    expect(host.logsPollInterval).toBeNull();
  });

  it("starts and stops debug polling based on the active operations section", () => {
    const host = createHost("chat");

    setTabFromRoute(host, {
      tab: "operations",
      operationsSection: "debug",
      agentsSection: "agents",
      workspaceSection: "projects",
      settingsSection: "config",
    });
    expect(host.debugPollInterval).not.toBeNull();
    expect(host.logsPollInterval).toBeNull();

    setTabFromRoute(host, {
      tab: "chat",
      operationsSection: "sessions",
      agentsSection: "agents",
      workspaceSection: "projects",
      settingsSection: "config",
    });
    expect(host.debugPollInterval).toBeNull();
  });
});
