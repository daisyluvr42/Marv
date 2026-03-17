import type { PluginRuntime } from "agentmarv/plugin-sdk";

let _runtime: PluginRuntime | undefined;

export function setWeChatRuntime(runtime: PluginRuntime): void {
  _runtime = runtime;
}

export function getWeChatRuntime(): PluginRuntime {
  if (!_runtime) {
    throw new Error("WeChat runtime not initialized — plugin not registered yet");
  }
  return _runtime;
}
