/**
 * Shared hook types extracted to break the circular dependency
 * between hooks.ts and hooks-mapping.ts.
 */
import type { ChannelId } from "../../channels/plugins/types.js";

export type HookMessageChannel = ChannelId | "last";
