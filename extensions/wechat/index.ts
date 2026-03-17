import type { MarvPluginApi } from "agentmarv/plugin-sdk";
import { emptyPluginConfigSchema } from "agentmarv/plugin-sdk";
import { wechatPlugin } from "./src/channel.js";
import { setWeChatRuntime } from "./src/runtime.js";

export { wechatPlugin } from "./src/channel.js";
export { probeWeChat, sendWeChatText } from "./src/api.js";

const plugin = {
  id: "wechat",
  name: "WeChat",
  description: "WeChat channel plugin via Wechaty",
  configSchema: emptyPluginConfigSchema(),
  register(api: MarvPluginApi) {
    setWeChatRuntime(api.runtime);
    api.registerChannel({ plugin: wechatPlugin as never });
  },
};

export default plugin;
