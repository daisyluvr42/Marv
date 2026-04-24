# Prompt Cache Optimization Plan

## Problem Statement

Marv's system prompt (~25+ sections) is sent as a single text block to the Anthropic API. Pi-ai's Anthropic provider wraps it in one `cache_control: { type: "ephemeral" }` block. When volatile sections (Project Context, Recalled Context) change between turns, the entire system prompt cache is invalidated because the byte sequence changes. This wastes cache-read tokens on every turn where context files are updated.

## Architecture Analysis

### Current Flow

1. `buildAgentSystemPrompt()` in `src/agents/prompt/system-prompt.ts` returns a single string with 33+ sections concatenated via `lines.filter(Boolean).join("\n")` (line 782).
2. `buildEmbeddedSystemPrompt()` in `src/agents/pi-embedded-runner/system-prompt.ts` is a thin wrapper that calls `buildAgentSystemPrompt()`.
3. `createSystemPromptOverride()` locks the string via a closure (line 91-96 of same file).
4. `applySystemPromptOverrideToSession()` sets it on `session.agent` (line 98-110).
5. Pi-ai's Anthropic provider in `node_modules/@mariozechner/pi-ai/dist/providers/anthropic.js` line 461-469 converts `context.systemPrompt` (a string) into `params.system = [{ type: "text", text: systemPrompt, cache_control: cacheControl }]` -- a single-element array.
6. `onPayload` callback fires with the full `params` object BEFORE the API call.

### Key Insight

The `onPayload` hook (used extensively via `createPayloadFieldWrapper()` in `extra-params.ts`) can mutate `params.system` from a 1-element array to a 2-element array, splitting at the volatile boundary. The stable prefix block keeps `cache_control: { type: "ephemeral" }` and the volatile suffix block gets no `cache_control` (or a separate one). This way Anthropic's prefix caching sees the stable portion as a cacheable prefix.

### Existing Patterns to Follow

- `createPayloadFieldWrapper()` at `src/agents/pi-embedded-runner/extra-params.ts:111-136` -- onPayload wrapper pattern
- `createAnthropicBetaHeadersWrapper()` -- provider-gated streamFn wrapper
- `createCacheTrace()` at `src/agents/cache-trace.ts` -- diagnostic streamFn wrapper
- `createAnthropicPayloadLogger()` at `src/agents/anthropic-payload-log.ts` -- payload inspection wrapper
- `applyExtraParamsToAgent()` at `extra-params.ts:376-449` -- wrapper composition site

## Implementation Plan

### Phase 1: Boundary Marker in System Prompt (Zero Risk)

**File: `src/agents/prompt/system-prompt.ts`**

1. Define a constant:

   ```ts
   export const PROMPT_CACHE_BOUNDARY = "__MARV_PROMPT_CACHE_BOUNDARY__";
   ```

2. At line 723 (the existing `// --- Volatile content below` comment), insert the boundary marker into the `lines` array immediately before the volatile content:

   ```ts
   lines.push(PROMPT_CACHE_BOUNDARY);
   ```

   This goes right before the `const contextFiles = params.contextFiles ?? [];` block that builds Project Context and Recalled Context sections.

3. The marker must NOT be filtered out by `lines.filter(Boolean)` on line 782 (it won't be, since it's a non-empty string).

4. The marker is invisible to the LLM because the `onPayload` wrapper (Phase 2) strips it during the split. If the wrapper is not active (non-Anthropic provider), the marker appears as a harmless line in the prompt.

**Impact**: No behavioral change. The marker is a plain text line that does not affect LLM behavior.

### Phase 2: System Prompt Block Splitter (Core Feature)

**New file: `src/agents/pi-embedded-runner/cache-split.ts`**

Create a new streamFn wrapper that intercepts the Anthropic API payload and splits the system prompt into two blocks at the boundary marker.

```ts
// Key exports:
export function createCacheSplitWrapper(
  baseStreamFn: StreamFn | undefined,
  provider: string,
): StreamFn | undefined;

// Internal logic:
function splitSystemPromptBlocks(
  systemBlocks: Array<{ type: string; text: string; cache_control?: unknown }>,
  boundary: string,
): Array<{ type: string; text: string; cache_control?: unknown }> | null;
```

**Behavior**:

1. Only activates for `provider === "anthropic"` (or openrouter with anthropic model).
2. In the `onPayload` callback:
   - Access `payload.system` (the array of text blocks constructed by pi-ai).
   - For each text block, check if `block.text` contains `PROMPT_CACHE_BOUNDARY`.
   - If found, split that block's `text` at the boundary into two parts:
     - **Stable prefix block**: `{ type: "text", text: prefixText, cache_control: { type: "ephemeral" } }` -- the stable portion retains cache control.
     - **Volatile suffix block**: `{ type: "text", text: suffixText }` -- no `cache_control`, so it doesn't anchor the cache to volatile content.
   - Replace the original block in the array with these two blocks.
   - Strip the boundary marker string itself from both halves.
3. If no boundary marker found, pass through unchanged (backward compatible).
4. If the prefix or suffix is empty after trimming, skip splitting.

**Why two blocks works for caching**: Anthropic's prompt caching works on prefix matching. The `cache_control` marker on block 1 tells Anthropic "cache up to here." Block 2 follows but doesn't have its own cache marker, so it's processed fresh each time. When block 1 is byte-identical across turns (which it will be for the stable sections), Anthropic serves it from cache.

**File: `src/agents/pi-embedded-runner/extra-params.ts`**

In `applyExtraParamsToAgent()` (line 376+), add the cache split wrapper to the wrapper chain. It should be applied:

- AFTER `createStreamFnWithExtraParams` (which sets `cacheRetention`)
- BEFORE `cacheTrace.wrapStreamFn` (so the trace sees the split payload)

Insert approximately at line 420 (after the think payload wrapper, before the beta headers wrapper):

```ts
// Cache split: split system prompt at boundary for better prefix caching
if (provider === "anthropic") {
  const cacheSplitWrapper = createCacheSplitWrapper(agent.streamFn, provider);
  if (cacheSplitWrapper) {
    agent.streamFn = cacheSplitWrapper;
  }
}
```

### Phase 3: Stable Prefix Memoization

**File: `src/agents/prompt/system-prompt.ts`**

Refactor `buildAgentSystemPrompt()` to return a structured result instead of just a string, while maintaining backward compatibility:

1. Extract the stable prefix construction (sections 1-31) into a helper:

   ```ts
   function buildStablePromptSections(params: ...): string[]
   ```

2. Extract the volatile suffix construction (sections 32-33) into a helper:

   ```ts
   function buildVolatilePromptSections(params: ...): string[]
   ```

3. Keep `buildAgentSystemPrompt()` returning a string (concatenation of stable + boundary + volatile) for backward compatibility.

4. Add a new export for structured access:

   ```ts
   export type StructuredSystemPrompt = {
     stablePrefix: string;
     volatileSuffix: string;
     combined: string; // stablePrefix + boundary + volatileSuffix
   };

   export function buildStructuredSystemPrompt(params: ...): StructuredSystemPrompt
   ```

**Session-level memoization** (in `src/agents/pi-embedded-runner/run/attempt.ts`):

The stable prefix depends on: tools, mode, scaffold level, runtime info, and other session-stable parameters. These don't change between turns within a session. However, the current architecture rebuilds the full prompt on each `runEmbeddedAttempt()` call.

For now, the boundary marker approach (Phases 1-2) is sufficient without explicit memoization, because:

- The stable sections ARE byte-identical across turns (same tools, same mode, same runtime info).
- The `cache_control: { type: "ephemeral" }` on the stable prefix tells Anthropic to cache it.
- Anthropic's prefix caching matches on byte-identical prefixes automatically.

Explicit memoization becomes valuable only if there's concern about subtle byte-level differences sneaking in (e.g., timestamps). The `userTime` field is passed to `buildAgentSystemPrompt()` but currently only generates a hint to "run session_status" rather than embedding the actual time. So this is safe.

**Deferred**: Full memoization can be added later if cache-trace diagnostics reveal unexpected cache misses in the stable prefix.

### Phase 4: Cache Break Detection (Monitoring)

**File: `src/agents/pi-embedded-runner/cache-split.ts`** (extend)

Add monitoring that logs when cache read tokens drop significantly between turns:

```ts
export type CacheSplitMonitor = {
  recordUsage(usage: { cacheRead?: number; cacheWrite?: number; input?: number }): void;
  getStats(): { turns: number; cacheHits: number; cacheMisses: number; avgCacheReadRatio: number };
};

export function createCacheSplitMonitor(sessionId: string): CacheSplitMonitor;
```

**Integration point**: In `attempt.ts`, after the run completes, extract usage from the last assistant message and feed it to the monitor:

```ts
// After run completes (around line 1058+):
cacheSplitMonitor?.recordUsage({
  cacheRead: lastUsage?.cacheRead,
  cacheWrite: lastUsage?.cacheWrite,
  input: lastUsage?.input,
});
```

The monitor compares `cacheRead / (cacheRead + input)` ratio between turns. A significant drop (e.g., from >80% to <20%) indicates a cache break, which should be logged at `warn` level.

**Integration with existing diagnostics**: The `createCacheTrace()` system already tracks `systemDigest` (SHA256 of system prompt). The cache split monitor complements this by tracking actual cache hit rates from the API response.

### Phase 5: Subagent Support

**File: `src/agents/subagent-announce.ts`** (function `buildSubagentSystemPrompt`)

Subagent prompts use `promptMode: "minimal"`, which already skips many sections. The boundary marker approach works here too since `buildAgentSystemPrompt()` is the shared builder. The boundary marker will be inserted at the same point, and the `onPayload` wrapper operates at the streaming layer regardless of which prompt builder was used.

No additional changes needed -- the wrapper in `applyExtraParamsToAgent()` applies to all sessions including subagent sessions.

## Sequencing and Dependencies

```
Phase 1 (boundary marker) ─── no dependencies, can ship alone
    │
    v
Phase 2 (onPayload splitter) ─── depends on Phase 1
    │
    v
Phase 3 (memoization) ─── optional, depends on Phase 2 monitoring data
    │
    v
Phase 4 (monitoring) ─── can ship with Phase 2
    │
    v
Phase 5 (subagent) ─── free, comes with Phase 2
```

Phases 1+2+4 should be shipped together as a single PR. Phase 3 is deferred.

## Testing Strategy

1. **Unit tests for `splitSystemPromptBlocks()`**: Verify correct splitting, boundary stripping, edge cases (no boundary, empty prefix/suffix, multiple boundaries).

2. **Unit tests for `createCacheSplitWrapper()`**: Verify it only activates for Anthropic provider, correctly mutates payload.system, passes through for non-Anthropic.

3. **E2E test**: Enable `MARV_ANTHROPIC_PAYLOAD_LOG=true`, run a 2-turn conversation, verify the logged payload shows `system` as a 2-element array with `cache_control` only on the first element.

4. **Existing test compatibility**: `src/agents/prompt/system-prompt.e2e.test.ts` -- verify boundary marker appears in output, doesn't break existing assertions.

## Risk Assessment

- **Low risk**: Phase 1 is a string literal addition with no behavioral impact.
- **Low risk**: Phase 2 operates in `onPayload` which fires after pi-ai builds the payload. If the split fails or is skipped, behavior is identical to today.
- **Provider-gated**: The wrapper only activates for `provider === "anthropic"`, so OpenRouter, Google, OpenAI paths are unaffected.
- **No pi-ai changes required**: Everything works through the existing `onPayload` hook.
- **Backward compatible**: If the boundary marker is absent (e.g., `promptMode: "none"`), the splitter is a no-op.

## Expected Impact

- **Stable prefix size**: ~15-20KB of text (sections 1-31) that remains byte-identical across turns.
- **Volatile suffix size**: Variable (depends on context files), typically 1-10KB.
- **Cache savings**: On a typical multi-turn conversation, the stable prefix is cached after turn 1. Subsequent turns read ~15-20KB from cache instead of re-processing it. At Anthropic's 90% discount for cached tokens, this saves ~$0.001-0.003 per turn for a 20KB prefix.
- **Latency improvement**: Cached tokens have ~80% lower time-to-first-token.

## Files to Create/Modify

| File                                                | Action | Description                                                               |
| --------------------------------------------------- | ------ | ------------------------------------------------------------------------- |
| `src/agents/prompt/system-prompt.ts`                | Modify | Add `PROMPT_CACHE_BOUNDARY` constant and insert marker at line 723        |
| `src/agents/pi-embedded-runner/cache-split.ts`      | Create | New file with `createCacheSplitWrapper()` and `splitSystemPromptBlocks()` |
| `src/agents/pi-embedded-runner/extra-params.ts`     | Modify | Add cache split wrapper to `applyExtraParamsToAgent()` chain              |
| `src/agents/pi-embedded-runner/cache-split.test.ts` | Create | Unit tests for splitting logic                                            |
| `src/agents/prompt/system-prompt.e2e.test.ts`       | Modify | Add test for boundary marker presence                                     |
