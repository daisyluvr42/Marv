/**
 * Anthropic Skills Bridge
 *
 * Bridges Anthropic's managed Agent Skills (DOCX, XLSX, PPTX, PDF processing)
 * for use within profession packs. Falls back to local tools when the
 * Anthropic API key is not configured.
 *
 * Skills API: POST /v1/skills/{skillId}/execute
 * Managed skills: docx, xlsx, pptx, pdf
 */

import type { MarvConfig } from "../core/config/config.js";

// ============================================================================
// Types
// ============================================================================

export type AnthropicSkillId = "docx" | "xlsx" | "pptx" | "pdf";

export type SkillExecutionInput = {
  /** The file content (base64-encoded for binary files, or raw text) */
  content: string;
  /** MIME type of the input */
  mimeType?: string;
  /** Action to perform (skill-specific) */
  action: string;
  /** Additional parameters */
  params?: Record<string, unknown>;
};

export type SkillExecutionResult = {
  success: boolean;
  output?: string;
  error?: string;
  metadata?: Record<string, unknown>;
};

// ============================================================================
// API Client
// ============================================================================

const ANTHROPIC_API_BASE = "https://api.anthropic.com";

/**
 * Execute an Anthropic managed skill.
 *
 * Requires an Anthropic API key with skills access.
 * Falls back gracefully if the key is missing or the skill is unavailable.
 */
export async function executeAnthropicSkill(
  skillId: AnthropicSkillId,
  input: SkillExecutionInput,
  apiKey: string,
): Promise<SkillExecutionResult> {
  const url = `${ANTHROPIC_API_BASE}/v1/skills/${skillId}/execute`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "skills-2025-10-02",
      },
      body: JSON.stringify({
        content: input.content,
        mime_type: input.mimeType,
        action: input.action,
        params: input.params,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        error: `Anthropic Skills API error (${response.status}): ${errorText}`,
      };
    }

    const result = (await response.json()) as Record<string, unknown>;
    return {
      success: true,
      output: result.output as string | undefined,
      metadata: result.metadata as Record<string, unknown> | undefined,
    };
  } catch (err) {
    return {
      success: false,
      error: `Failed to call Anthropic Skills API: ${String(err)}`,
    };
  }
}

// ============================================================================
// Helper
// ============================================================================

/**
 * Resolve the Anthropic API key from config.
 * Returns undefined if not configured.
 */
export function resolveAnthropicApiKey(config: MarvConfig): string | undefined {
  // Check provider config first
  const providers = config as Record<string, unknown>;
  const anthropicProvider = (providers.providers as Record<string, unknown>)?.anthropic as
    | Record<string, unknown>
    | undefined;

  if (anthropicProvider?.apiKey && typeof anthropicProvider.apiKey === "string") {
    return anthropicProvider.apiKey;
  }

  // Fall back to environment variable
  return process.env.ANTHROPIC_API_KEY;
}

/**
 * Check if Anthropic Skills are available (API key configured).
 */
export function isAnthropicSkillsAvailable(config: MarvConfig): boolean {
  return !!resolveAnthropicApiKey(config);
}

/**
 * High-level helper for packs to use Anthropic skills with fallback.
 *
 * Usage in a pack tool:
 * ```ts
 * const result = await useAnthropicSkill("docx", {
 *   content: base64Content,
 *   mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
 *   action: "extract_text",
 * }, config);
 *
 * if (result.success) {
 *   // Use result.output (extracted text)
 * } else {
 *   // Fall back to local processing
 * }
 * ```
 */
export async function useAnthropicSkill(
  skillId: AnthropicSkillId,
  input: SkillExecutionInput,
  config: MarvConfig,
): Promise<SkillExecutionResult> {
  const apiKey = resolveAnthropicApiKey(config);

  if (!apiKey) {
    return {
      success: false,
      error:
        "Anthropic API key not configured. Set providers.anthropic.apiKey in config or ANTHROPIC_API_KEY env var.",
    };
  }

  return executeAnthropicSkill(skillId, input, apiKey);
}
