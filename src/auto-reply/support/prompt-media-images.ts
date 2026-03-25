import { fileURLToPath } from "node:url";
import type { ImageContent } from "@mariozechner/pi-ai";
import {
  loadImageFromRef,
  type DetectedImageRef,
} from "../../agents/pi-embedded-runner/run/images.js";
import type { PromptMediaRef } from "../../contracts/media-ref.js";
import type { TurnContext } from "./templating.js";

function promptMediaToImageRef(entry: PromptMediaRef): DetectedImageRef | null {
  if (entry.kind !== "image") {
    return null;
  }
  if (entry.path?.trim()) {
    const resolved = entry.path.trim();
    return {
      raw: resolved,
      resolved,
      type: "path",
    };
  }
  const url = entry.url?.trim();
  if (!url) {
    return null;
  }
  if (url.startsWith("file://")) {
    const resolved = fileURLToPath(url);
    return {
      raw: url,
      resolved,
      type: "path",
    };
  }
  return {
    raw: url,
    resolved: url,
    type: "url",
  };
}

function resolvePromptMediaEntries(ctx: TurnContext): PromptMediaRef[] {
  return ctx.MultimodalRouting?.promptMedia ?? ctx.PromptMedia ?? [];
}

export async function loadPromptMediaImages(params: {
  ctx: TurnContext;
  workspaceDir: string;
  existingImages?: ImageContent[];
}): Promise<ImageContent[] | undefined> {
  const refs = resolvePromptMediaEntries(params.ctx)
    .map(promptMediaToImageRef)
    .filter((entry): entry is DetectedImageRef => Boolean(entry));
  if (refs.length === 0) {
    return params.existingImages;
  }

  const seen = new Set(
    (params.existingImages ?? []).map((image) => `${image.mimeType}:${image.data.slice(0, 48)}`),
  );
  const images: ImageContent[] = [...(params.existingImages ?? [])];
  for (const ref of refs) {
    const image = await loadImageFromRef(ref, params.workspaceDir);
    if (!image) {
      continue;
    }
    const key = `${image.mimeType}:${image.data.slice(0, 48)}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    images.push(image);
  }
  return images.length > 0 ? images : undefined;
}
