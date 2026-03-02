import type { ChannelPlugin, MarvPluginApi } from "marv/plugin-sdk";
import { emptyPluginConfigSchema } from "marv/plugin-sdk";
import { ircPlugin } from "./src/channel.js";
import { setIrcRuntime } from "./src/runtime.js";

const plugin = {
  id: "irc",
  name: "IRC",
  description: "IRC channel plugin",
  configSchema: emptyPluginConfigSchema(),
  register(api: MarvPluginApi) {
    setIrcRuntime(api.runtime);
    api.registerChannel({ plugin: ircPlugin as ChannelPlugin });
  },
};

export default plugin;
