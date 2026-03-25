/**
 * Shared channel config types used by both `types.channels.ts` and individual
 * channel type files (e.g. `types.discord.ts`). Extracted to break circular
 * imports between the aggregate and per-channel modules.
 */

export type ChannelHeartbeatVisibilityConfig = {
  /** Show HEARTBEAT_OK acknowledgments in chat (default: false). */
  showOk?: boolean;
  /** Show heartbeat alerts with actual content (default: true). */
  showAlerts?: boolean;
  /** Emit indicator events for UI status display (default: true). */
  useIndicator?: boolean;
};
