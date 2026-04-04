import type { MarvApp } from "./app.js";
import { loadDebug } from "./controllers/debug.js";
import { loadLogs } from "./controllers/logs.js";
import { loadNodes } from "./controllers/nodes.js";
import { loadWorkbench, WORKBENCH_POLL_INTERVAL_MS } from "./controllers/workbench.js";
import {
  isDebugView,
  isLogsView,
  isWorkbenchView,
  type OperationsSection,
  type Tab,
  type WorkspaceSection,
} from "./navigation.js";

type PollingHost = {
  nodesPollInterval: number | null;
  logsPollInterval: number | null;
  debugPollInterval: number | null;
  workbenchPollInterval: number | null;
  tab: Tab;
  operationsSection?: OperationsSection;
  workspaceSection?: WorkspaceSection;
};

export function startNodesPolling(host: PollingHost) {
  if (host.nodesPollInterval != null) {
    return;
  }
  host.nodesPollInterval = window.setInterval(
    () => void loadNodes(host as unknown as MarvApp, { quiet: true }),
    5000,
  );
}

export function stopNodesPolling(host: PollingHost) {
  if (host.nodesPollInterval == null) {
    return;
  }
  clearInterval(host.nodesPollInterval);
  host.nodesPollInterval = null;
}

export function startLogsPolling(host: PollingHost) {
  if (host.logsPollInterval != null) {
    return;
  }
  host.logsPollInterval = window.setInterval(() => {
    if (!isLogsView(host.tab, host.operationsSection ?? "sessions")) {
      return;
    }
    void loadLogs(host as unknown as MarvApp, { quiet: true });
  }, 2000);
}

export function stopLogsPolling(host: PollingHost) {
  if (host.logsPollInterval == null) {
    return;
  }
  clearInterval(host.logsPollInterval);
  host.logsPollInterval = null;
}

export function startDebugPolling(host: PollingHost) {
  if (host.debugPollInterval != null) {
    return;
  }
  host.debugPollInterval = window.setInterval(() => {
    if (!isDebugView(host.tab, host.operationsSection ?? "sessions")) {
      return;
    }
    void loadDebug(host as unknown as MarvApp);
  }, 3000);
}

export function stopDebugPolling(host: PollingHost) {
  if (host.debugPollInterval == null) {
    return;
  }
  clearInterval(host.debugPollInterval);
  host.debugPollInterval = null;
}

export function startWorkbenchPolling(host: PollingHost) {
  if (host.workbenchPollInterval != null) {
    return;
  }
  host.workbenchPollInterval = window.setInterval(() => {
    if (!isWorkbenchView(host.tab, host.workspaceSection ?? "projects")) {
      return;
    }
    void loadWorkbench(host as unknown as MarvApp);
  }, WORKBENCH_POLL_INTERVAL_MS);
}

export function stopWorkbenchPolling(host: PollingHost) {
  if (host.workbenchPollInterval == null) {
    return;
  }
  clearInterval(host.workbenchPollInterval);
  host.workbenchPollInterval = null;
}
