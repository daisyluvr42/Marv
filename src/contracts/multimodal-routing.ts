import type {
  MediaUnderstandingDecision,
  MediaUnderstandingOutput,
} from "../media-understanding/types.js";
import type { PromptMediaRef } from "./media-ref.js";

export type StreamingDerivedTextProducer = {
  kind: "audio-stt";
  onSettled: (transcript: string) => void;
  onPartial?: (partial: string) => void;
};

export type MultimodalRoutingDerivedText = {
  transcript?: string;
  mediaOutputs?: MediaUnderstandingOutput[];
  fileBlocks?: string[];
};

export type MultimodalRoutingResult = {
  promptMedia: PromptMediaRef[];
  derivedText: MultimodalRoutingDerivedText;
  decisions: MediaUnderstandingDecision[];
  settled: boolean;
  producer?: StreamingDerivedTextProducer;
};

// Keep one compatibility contract for media prompt shaping instead of
// introducing a second, overlapping abstraction.
export type MediaPromptCompatibility = MultimodalRoutingResult;
