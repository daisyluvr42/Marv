import type { PromptMediaRef } from "../../contracts/media-ref.js";
import { buildCompatibilityPromptMedia } from "../../contracts/multimodal-routing-compat.js";
import type { TurnContext } from "./templating.js";

function formatMediaAttachedLine(params: {
  path: string;
  url?: string;
  type?: string;
  index?: number;
  total?: number;
}): string {
  const prefix =
    typeof params.index === "number" && typeof params.total === "number"
      ? `[media attached ${params.index}/${params.total}: `
      : "[media attached: ";
  const typePart = params.type?.trim() ? ` (${params.type.trim()})` : "";
  const urlRaw = params.url?.trim();
  const urlPart = urlRaw ? ` | ${urlRaw}` : "";
  return `${prefix}${params.path}${typePart}${urlPart}]`;
}

function formatPromptMediaNote(promptMedia: PromptMediaRef[]): string | undefined {
  if (promptMedia.length === 0) {
    return undefined;
  }
  const entries = promptMedia.map((entry) => ({
    path: entry.path?.trim() || entry.url?.trim() || `[${entry.kind}]`,
    type: entry.contentType,
    url: entry.url,
  }));
  if (entries.length === 1) {
    return formatMediaAttachedLine(entries[0]);
  }

  const count = entries.length;
  const lines: string[] = [`[media attached: ${count} files]`];
  for (const [idx, entry] of entries.entries()) {
    lines.push(
      formatMediaAttachedLine({
        path: entry.path,
        index: idx + 1,
        total: count,
        type: entry.type,
        url: entry.url,
      }),
    );
  }
  return lines.join("\n");
}

export function buildInboundMediaNote(ctx: TurnContext): string | undefined {
  // Treat media-note as a compatibility adapter over the newer routing contract.
  if (ctx.MultimodalRouting) {
    return formatPromptMediaNote(ctx.MultimodalRouting.promptMedia);
  }
  return formatPromptMediaNote(
    buildCompatibilityPromptMedia({
      mediaPath: ctx.MediaPath,
      mediaPaths: ctx.MediaPaths,
      mediaUrl: ctx.MediaUrl,
      mediaUrls: ctx.MediaUrls,
      mediaType: ctx.MediaType,
      mediaTypes: ctx.MediaTypes,
      transcript: ctx.Transcript,
      mediaUnderstanding: ctx.MediaUnderstanding,
      mediaUnderstandingDecisions: ctx.MediaUnderstandingDecisions,
    }),
  );
}
