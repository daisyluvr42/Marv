import type {
  MediaUnderstandingDecision,
  MediaUnderstandingOutput,
} from "../media-understanding/types.js";
import type { PromptMediaKind, PromptMediaRef } from "./media-ref.js";

type CompatibilityPromptMediaInput = {
  mediaPath?: string;
  mediaPaths?: string[];
  mediaUrl?: string;
  mediaUrls?: string[];
  mediaType?: string;
  mediaTypes?: string[];
  transcript?: string;
  mediaUnderstanding?: MediaUnderstandingOutput[];
  mediaUnderstandingDecisions?: MediaUnderstandingDecision[];
};

const AUDIO_EXTENSIONS = new Set([
  ".ogg",
  ".opus",
  ".mp3",
  ".m4a",
  ".wav",
  ".webm",
  ".flac",
  ".aac",
  ".wma",
  ".aiff",
  ".alac",
  ".oga",
]);

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".svg",
  ".heic",
  ".heif",
  ".tif",
  ".tiff",
  ".avif",
]);

const VIDEO_EXTENSIONS = new Set([
  ".mp4",
  ".mov",
  ".m4v",
  ".avi",
  ".mkv",
  ".webm",
  ".mpeg",
  ".mpg",
  ".wmv",
  ".3gp",
]);

type CompatibilityPromptMediaEntry = {
  path?: string;
  url?: string;
  type?: string;
  index: number;
};

function hasExtension(path: string | undefined, extensions: Set<string>): boolean {
  if (!path) {
    return false;
  }
  const lower = path.toLowerCase();
  for (const ext of extensions) {
    if (lower.endsWith(ext)) {
      return true;
    }
  }
  return false;
}

function inferPromptMediaKind(entry: CompatibilityPromptMediaEntry): PromptMediaKind {
  const mime = entry.type?.toLowerCase();
  if (mime?.startsWith("image/") || hasExtension(entry.path ?? entry.url, IMAGE_EXTENSIONS)) {
    return "image";
  }
  if (mime?.startsWith("audio/") || hasExtension(entry.path ?? entry.url, AUDIO_EXTENSIONS)) {
    return "audio";
  }
  if (mime?.startsWith("video/") || hasExtension(entry.path ?? entry.url, VIDEO_EXTENSIONS)) {
    return "video";
  }
  return "file";
}

function collectSuppressedAttachmentState(params: {
  mediaUnderstanding?: MediaUnderstandingOutput[];
  mediaUnderstandingDecisions?: MediaUnderstandingDecision[];
}) {
  const suppressed = new Set<number>();
  const transcribedAudioIndices = new Set<number>();
  if (Array.isArray(params.mediaUnderstanding)) {
    for (const output of params.mediaUnderstanding) {
      suppressed.add(output.attachmentIndex);
      if (output.kind === "audio.transcription") {
        transcribedAudioIndices.add(output.attachmentIndex);
      }
    }
  }
  if (Array.isArray(params.mediaUnderstandingDecisions)) {
    for (const decision of params.mediaUnderstandingDecisions) {
      if (decision.outcome !== "success") {
        continue;
      }
      for (const attachment of decision.attachments) {
        if (attachment.chosen?.outcome === "success") {
          suppressed.add(attachment.attachmentIndex);
          if (decision.capability === "audio") {
            transcribedAudioIndices.add(attachment.attachmentIndex);
          }
        }
      }
    }
  }
  return { suppressed, transcribedAudioIndices };
}

function resolveCompatibilityEntries(
  input: CompatibilityPromptMediaInput,
): CompatibilityPromptMediaEntry[] {
  const pathsFromArray = Array.isArray(input.mediaPaths) ? input.mediaPaths : undefined;
  const paths =
    pathsFromArray && pathsFromArray.length > 0
      ? pathsFromArray
      : input.mediaPath?.trim()
        ? [input.mediaPath.trim()]
        : [];
  if (paths.length === 0) {
    return [];
  }

  const urls =
    Array.isArray(input.mediaUrls) && input.mediaUrls.length === paths.length
      ? input.mediaUrls
      : undefined;
  const types =
    Array.isArray(input.mediaTypes) && input.mediaTypes.length === paths.length
      ? input.mediaTypes
      : undefined;
  const canStripSingleAttachmentByTranscript =
    Boolean(input.transcript?.trim()) && paths.length === 1;
  const { suppressed, transcribedAudioIndices } = collectSuppressedAttachmentState(input);

  return paths
    .map((path, index) => ({
      path: path ?? "",
      type: types?.[index] ?? input.mediaType,
      url: urls?.[index] ?? input.mediaUrl,
      index,
    }))
    .filter((entry) => {
      if (suppressed.has(entry.index)) {
        return false;
      }
      const hasPerEntryType = types !== undefined;
      const isAudioByMime = hasPerEntryType && entry.type?.toLowerCase().startsWith("audio/");
      const isAudioEntry = hasExtension(entry.path, AUDIO_EXTENSIONS) || isAudioByMime;
      if (!isAudioEntry) {
        return true;
      }
      if (
        transcribedAudioIndices.has(entry.index) ||
        (canStripSingleAttachmentByTranscript && entry.index === 0)
      ) {
        return false;
      }
      return true;
    });
}

export function buildCompatibilityPromptMedia(
  input: CompatibilityPromptMediaInput,
): PromptMediaRef[] {
  return resolveCompatibilityEntries(input).map((entry) => ({
    attachmentIndex: entry.index,
    kind: inferPromptMediaKind(entry),
    source: "native",
    path: entry.path,
    url: entry.url,
    contentType: entry.type,
  }));
}
