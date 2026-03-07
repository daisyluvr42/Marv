import { html, render } from "lit";
import { describe, expect, it } from "vitest";
import { buildModelOptions } from "./agents-utils.js";

function renderOptions(params: {
  configForm?: Record<string, unknown> | null;
  availableModels?: Array<{ provider: string; id: string; name: string }> | null;
  current?: string | null;
  loading?: boolean;
}) {
  const container = document.createElement("div");
  render(
    html`<select>
      ${buildModelOptions(
        params.configForm ?? null,
        params.availableModels ?? null,
        params.current,
        params.loading ?? false,
      )}
    </select>`,
    container,
  );
  return Array.from(container.querySelectorAll("option")).map((option) => ({
    value: option.value,
    label: option.textContent?.trim() ?? "",
  }));
}

describe("buildModelOptions", () => {
  it("prefers gateway model catalog entries over a config-only list", () => {
    const options = renderOptions({
      availableModels: [{ provider: "openai", id: "gpt-5.2", name: "GPT-5.2" }],
    });

    expect(options).toEqual([{ value: "openai/gpt-5.2", label: "GPT-5.2 (openai/gpt-5.2)" }]);
  });

  it("keeps config-only models that are missing from the catalog", () => {
    const options = renderOptions({
      configForm: {
        agents: {
          defaults: {
            models: {
              "custom/foo-large": { alias: "Foo Large" },
            },
          },
        },
      },
      availableModels: [{ provider: "openai", id: "gpt-5.2", name: "GPT-5.2" }],
    });

    expect(options).toEqual([
      { value: "openai/gpt-5.2", label: "GPT-5.2 (openai/gpt-5.2)" },
      { value: "custom/foo-large", label: "Foo Large (custom/foo-large)" },
    ]);
  });

  it("keeps the current model visible when it is absent from config and catalog", () => {
    const options = renderOptions({
      current: "anthropic/claude-opus-4-6",
      availableModels: [{ provider: "openai", id: "gpt-5.2", name: "GPT-5.2" }],
    });

    expect(options[0]).toEqual({
      value: "anthropic/claude-opus-4-6",
      label: "Current (anthropic/claude-opus-4-6)",
    });
  });
});
