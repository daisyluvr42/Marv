import type {
  ChannelConfigAdapter,
  ChannelGatewayAdapter,
  ChannelId,
  ChannelMessagingAdapter,
  ChannelMeta,
  ChannelOutboundAdapter,
  ChannelPlugin,
  ChannelSetupAdapter,
  ChannelStatusAdapter,
} from "./plugins/types.js";

// Stable channel contract for shared code. Prefer this entrypoint over
// direct imports from `src/channels/plugins/*` when wiring new channels.
export type ChannelAdapter<ResolvedAccount = unknown, Probe = unknown, Audit = unknown> = Pick<
  ChannelPlugin<ResolvedAccount, Probe, Audit>,
  "id" | "meta" | "config" | "setup" | "outbound" | "messaging" | "gateway" | "status"
>;

export type {
  ChannelConfigAdapter,
  ChannelGatewayAdapter,
  ChannelId,
  ChannelMessagingAdapter,
  ChannelMeta,
  ChannelOutboundAdapter,
  ChannelSetupAdapter,
  ChannelStatusAdapter,
};
