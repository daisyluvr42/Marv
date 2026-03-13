import { describe, expect, it } from "vitest";
import {
  buildCompatibilityPromptMedia,
  renderCommandArgs,
  type CommandArgRenderer,
} from "./index.js";

describe("contracts index", () => {
  it("re-exports the shared command renderer helper", () => {
    const renderers: Record<string, CommandArgRenderer> = {
      deploy: ({ values }) => `target=${String(values.target ?? "")}`,
    };
    expect(
      renderCommandArgs({
        commandName: "deploy",
        values: { target: "prod" },
        renderers,
      }),
    ).toBe("target=prod");
  });

  it("re-exports the multimodal compatibility helper", () => {
    expect(
      buildCompatibilityPromptMedia({
        mediaPaths: ["/tmp/example.png"],
        mediaTypes: ["image/png"],
      }),
    ).toEqual([
      {
        attachmentIndex: 0,
        contentType: "image/png",
        kind: "image",
        path: "/tmp/example.png",
        source: "native",
      },
    ]);
  });
});
