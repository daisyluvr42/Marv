import { describe, expect, it } from "vitest";
import { buildCompatibilityPromptMedia } from "./multimodal-routing-compat.js";

describe("buildCompatibilityPromptMedia", () => {
  it("preserves visible image attachments as native prompt media", () => {
    expect(
      buildCompatibilityPromptMedia({
        mediaPaths: ["/tmp/a.png", "/tmp/b.jpg"],
        mediaTypes: ["image/png", "image/jpeg"],
        mediaUrls: ["https://example.com/a.png", "https://example.com/b.jpg"],
      }),
    ).toEqual([
      {
        attachmentIndex: 0,
        kind: "image",
        source: "native",
        path: "/tmp/a.png",
        url: "https://example.com/a.png",
        contentType: "image/png",
      },
      {
        attachmentIndex: 1,
        kind: "image",
        source: "native",
        path: "/tmp/b.jpg",
        url: "https://example.com/b.jpg",
        contentType: "image/jpeg",
      },
    ]);
  });

  it("strips audio attachments once transcription is available", () => {
    expect(
      buildCompatibilityPromptMedia({
        mediaPaths: ["/tmp/voice.ogg", "/tmp/photo.png"],
        mediaTypes: ["audio/ogg", "image/png"],
        mediaUnderstanding: [
          {
            kind: "audio.transcription",
            attachmentIndex: 0,
            text: "hello",
            provider: "whisper",
          },
        ],
      }),
    ).toEqual([
      {
        attachmentIndex: 1,
        kind: "image",
        source: "native",
        path: "/tmp/photo.png",
        url: undefined,
        contentType: "image/png",
      },
    ]);
  });
});
