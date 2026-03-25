import type {
  CommandHandler,
  CommandHandlerResult,
  HandleCommandsParams,
} from "./handler-types.js";
import type { ChatCommandDefinition } from "./types.js";

export type CommandRegistryEntry = {
  key: string;
  handler: CommandHandler;
  meta?: ChatCommandDefinition;
};

/**
 * Centralized command registry. Handlers self-register with a key,
 * handler function, and optional metadata (ChatCommandDefinition).
 *
 * Dispatch iterates handlers in registration order (chain-of-responsibility):
 * each handler returns a result if it matches, or null to pass to the next.
 */
export class CommandRegistry {
  private entries: CommandRegistryEntry[] = [];
  private keyIndex = new Map<string, CommandRegistryEntry>();

  /**
   * Register a command handler.
   *
   * @param key   Unique identifier for the handler (e.g. "help", "bash", "__plugin").
   * @param handler  The handler function.
   * @param meta  Optional ChatCommandDefinition metadata for the command.
   */
  register(key: string, handler: CommandHandler, meta?: ChatCommandDefinition): void {
    const normalized = key.toLowerCase();
    if (this.keyIndex.has(normalized)) {
      throw new Error(`Duplicate command handler key: ${key}`);
    }
    const entry: CommandRegistryEntry = { key: normalized, handler, meta };
    this.entries.push(entry);
    this.keyIndex.set(normalized, entry);
  }

  /**
   * Dispatch a command through the handler chain in registration order.
   * Returns the first non-null handler result, or null if no handler matched.
   */
  async dispatch(
    params: HandleCommandsParams,
    allowTextCommands: boolean,
  ): Promise<CommandHandlerResult | null> {
    for (const entry of this.entries) {
      const result = await entry.handler(params, allowTextCommands);
      if (result) {
        return result;
      }
    }
    return null;
  }

  /** Check whether a handler with the given key is registered. */
  has(key: string): boolean {
    return this.keyIndex.has(key.toLowerCase());
  }

  /** Return the handler entry for a key, or undefined. */
  get(key: string): CommandRegistryEntry | undefined {
    return this.keyIndex.get(key.toLowerCase());
  }

  /** Return metadata for all registered commands that have it. */
  listMeta(): ChatCommandDefinition[] {
    return this.entries.filter((e) => e.meta != null).map((e) => e.meta!);
  }

  /** Return all registered entries in order. */
  list(): CommandRegistryEntry[] {
    return [...this.entries];
  }

  /** Number of registered handlers. */
  get size(): number {
    return this.entries.length;
  }
}
