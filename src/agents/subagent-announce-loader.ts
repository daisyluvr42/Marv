/**
 * Thin indirection module to break the circular dependency between
 * subagent-registry and subagent-announce while staying mockable by Vitest.
 */
type AnnounceModule = { runSubagentAnnounceFlow: (...args: unknown[]) => Promise<boolean> };

let cached: AnnounceModule | null = null;

export async function loadAnnounceModule(): Promise<AnnounceModule> {
  if (!cached) {
    cached = (await import("./subagent-announce.js")) as AnnounceModule;
  }
  return cached;
}

/** @internal test-only */
export function resetAnnounceLoaderForTests() {
  cached = null;
}
