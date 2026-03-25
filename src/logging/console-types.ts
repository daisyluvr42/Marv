/**
 * Shared console logging types extracted to break the circular dependency
 * between logger.ts and console.ts.
 */

export type ConsoleStyle = "pretty" | "compact" | "json";
