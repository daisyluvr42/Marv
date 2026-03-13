import { describe, expect, it } from "vitest";
import { CHANNEL_POLICY_CONTRACT_VERSION as channelPluginVersion } from "../channels/plugins/index.js";
import { CHANNEL_POLICY_CONTRACT_VERSION as pluginSdkVersion } from "./index.js";

describe("plugin sdk channel policy contract", () => {
  it("re-exports the shared channel policy contract version without drift", () => {
    expect(pluginSdkVersion).toBe(channelPluginVersion);
  });
});
