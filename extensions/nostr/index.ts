import type { MarvPluginApi } from "agentmarv/plugin-sdk";
import { emptyPluginConfigSchema } from "agentmarv/plugin-sdk";
import { nostrPlugin } from "./src/channel.js";
import type { NostrProfile } from "./src/config-schema.js";
import { createNostrProfileHttpHandler } from "./src/nostr-profile-http.js";
import { setNostrRuntime, getNostrRuntime } from "./src/runtime.js";
import {
  listNostrAccountIds,
  resolveDefaultNostrAccountId,
  resolveNostrAccount,
} from "./src/types.js";

const plugin = {
  id: "nostr",
  name: "Nostr",
  description: "Nostr DM channel plugin via NIP-04",
  configSchema: emptyPluginConfigSchema(),
  register(api: MarvPluginApi) {
    setNostrRuntime(api.runtime);
    api.registerChannel({ plugin: nostrPlugin });

    // Register HTTP handler for profile management
    const httpHandler = createNostrProfileHttpHandler({
      getConfigProfile: (accountId: string) => {
        const runtime = getNostrRuntime();
        const cfg = runtime.config.loadConfig();
        const account = resolveNostrAccount({ cfg, accountId });
        return account.profile;
      },
      updateConfigProfile: async (accountId: string, profile: NostrProfile) => {
        const runtime = getNostrRuntime();
        const cfg = runtime.config.loadConfig();

        // Build the config patch for channels.nostr.profile
        const channels = (cfg.channels ?? {}) as Record<string, unknown>;
        const nostrConfig = (channels.nostr ?? {}) as Record<string, unknown>;

        const updatedNostrConfig = {
          ...nostrConfig,
          profile,
        };

        const updatedChannels = {
          ...channels,
          nostr: updatedNostrConfig,
        };

        await runtime.config.writeConfigFile({
          ...cfg,
          channels: updatedChannels,
        });
      },
      getAccountInfo: (accountId: string) => {
        const runtime = getNostrRuntime();
        const cfg = runtime.config.loadConfig();
        const account = resolveNostrAccount({ cfg, accountId });
        if (!account.configured || !account.publicKey) {
          return null;
        }
        return {
          pubkey: account.publicKey,
          relays: account.relays,
        };
      },
      log: api.logger,
    });

    const accountIds = new Set([
      resolveDefaultNostrAccountId(api.config),
      ...listNostrAccountIds(api.config),
    ]);

    for (const accountId of accountIds) {
      const encodedAccountId = encodeURIComponent(accountId);
      const profilePath = `/api/channels/nostr/${encodedAccountId}/profile`;
      const importPath = `/api/channels/nostr/${encodedAccountId}/profile/import`;

      api.registerHttpRoute({
        path: profilePath,
        handler: async (req, res) => {
          const handled = await httpHandler(req, res);
          if (!handled && !res.headersSent) {
            res.statusCode = 404;
            res.end("not found");
          }
        },
      });
      api.registerHttpRoute({
        path: importPath,
        handler: async (req, res) => {
          const handled = await httpHandler(req, res);
          if (!handled && !res.headersSent) {
            res.statusCode = 404;
            res.end("not found");
          }
        },
      });
    }
  },
};

export default plugin;
