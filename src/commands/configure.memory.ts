import type { MarvConfig } from "../core/config/config.js";
import type { MemorySearchConfig } from "../core/config/types.tools.js";
import type { RuntimeEnv } from "../runtime.js";
import { note } from "../terminal/note.js";
import type { WizardPrompter } from "../wizard/prompts.js";
import { confirm, select, text } from "./configure.shared.js";
import { guardCancel } from "./onboard-helpers.js";

type EmbeddingProviderChoice = "openai" | "gemini" | "voyage" | "local" | "script" | "auto";

const EMBEDDING_PROVIDER_OPTIONS: Array<{
  value: EmbeddingProviderChoice;
  label: string;
  hint: string;
}> = [
  { value: "auto", label: "Auto-detect", hint: "Try available providers in order" },
  { value: "openai", label: "OpenAI", hint: "text-embedding-3-small (requires API key)" },
  { value: "gemini", label: "Gemini", hint: "gemini-embedding-001 (requires API key)" },
  { value: "voyage", label: "Voyage AI", hint: "voyage-4-large (requires API key)" },
  { value: "local", label: "Local (node-llama-cpp)", hint: "GGUF model, runs on-device" },
  { value: "script", label: "Local (script)", hint: "Zero-dependency hash vectorizer fallback" },
];

const DEFAULT_MODELS: Record<string, string> = {
  openai: "text-embedding-3-small",
  gemini: "gemini-embedding-001",
  voyage: "voyage-4-large",
};

type RerankerDeployment = "local" | "remote";

const RERANKER_DEPLOYMENT_OPTIONS: Array<{
  value: RerankerDeployment;
  label: string;
  hint: string;
}> = [
  {
    value: "local",
    label: "Local",
    hint: "Self-hosted reranker (e.g. TEI, vLLM, llama.cpp server)",
  },
  {
    value: "remote",
    label: "Remote",
    hint: "Cloud reranker API (e.g. Cohere, Jina, Voyage)",
  },
];

// Shared reranker config prompt used by both configure and onboarding wizards.
async function promptRerankerConfig(
  memorySearch: MemorySearchConfig,
  existing: MemorySearchConfig | undefined,
  io: {
    select: (params: {
      message: string;
      options: Array<{ value: string; label: string; hint?: string }>;
      initialValue?: string;
    }) => Promise<string>;
    text: (params: {
      message: string;
      initialValue?: string;
      placeholder?: string;
    }) => Promise<string>;
  },
): Promise<MemorySearchConfig> {
  const existingApiUrl = existing?.query?.hybrid?.reranker?.apiUrl ?? "";
  const isExistingLocal =
    existingApiUrl.includes("localhost") || existingApiUrl.includes("127.0.0.1");
  const defaultDeployment: RerankerDeployment = isExistingLocal ? "local" : "remote";

  const deployment = (await io.select({
    message: "Reranker deployment",
    options: RERANKER_DEPLOYMENT_OPTIONS,
    initialValue: defaultDeployment,
  })) as RerankerDeployment;

  const isLocal = deployment === "local";
  const defaultUrl = isLocal ? "http://localhost:8080/rerank" : "";
  const urlPlaceholder = isLocal
    ? "http://localhost:8080/rerank"
    : "https://api.cohere.com/v2/rerank";
  const modelPlaceholder = isLocal ? "BAAI/bge-reranker-v2-m3" : "rerank-v3.5";

  const rerankerApiUrl = await io.text({
    message: isLocal ? "Reranker endpoint URL" : "Reranker API URL",
    initialValue: existingApiUrl || defaultUrl,
    placeholder: urlPlaceholder,
  });

  const rerankerModel = await io.text({
    message: "Reranker model",
    initialValue: existing?.query?.hybrid?.reranker?.model ?? "",
    placeholder: modelPlaceholder,
  });

  const hasRerankerKey = Boolean(existing?.query?.hybrid?.reranker?.apiKey);
  const apiKeyMessage = isLocal
    ? hasRerankerKey
      ? "Reranker API key (leave blank to keep current, or clear if not needed)"
      : "Reranker API key (leave blank if your local server doesn't require one)"
    : hasRerankerKey
      ? "Reranker API key (leave blank to keep current)"
      : "Reranker API key";

  const rerankerApiKey = await io.text({
    message: apiKeyMessage,
    placeholder: hasRerankerKey ? "Leave blank to keep current" : isLocal ? "" : "sk-...",
  });

  return {
    ...memorySearch,
    query: {
      ...memorySearch.query,
      hybrid: {
        ...memorySearch.query?.hybrid,
        reranker: {
          ...memorySearch.query?.hybrid?.reranker,
          enabled: true,
          apiUrl: rerankerApiUrl.trim() || undefined,
          model: rerankerModel.trim() || undefined,
          apiKey: rerankerApiKey.trim() || existing?.query?.hybrid?.reranker?.apiKey,
        },
      },
    },
  };
}

export async function promptMemorySearchConfig(
  nextConfig: MarvConfig,
  runtime: RuntimeEnv,
): Promise<MarvConfig> {
  const existing = nextConfig.agents?.defaults?.memorySearch;

  note(
    [
      "Memory search uses embeddings to find relevant memories and session history.",
      "An embedding provider converts text into vectors for semantic search.",
      "",
      "Options:",
      "- Remote providers (OpenAI, Gemini, Voyage) need an API key.",
      "- Local (node-llama-cpp) runs a GGUF model on-device.",
      "- Local (script) is a zero-dependency fallback using hash vectorization.",
      "- Auto-detect tries available providers in order.",
      "",
      "Reranking is optional and improves search quality by reordering results.",
    ].join("\n"),
    "Memory search",
  );

  // --- Embedding provider ---
  const currentProvider = existing?.provider ?? "auto";
  const providerChoice = guardCancel(
    await select<EmbeddingProviderChoice>({
      message: "Embedding provider",
      options: EMBEDDING_PROVIDER_OPTIONS,
      initialValue: currentProvider as EmbeddingProviderChoice,
    }),
    runtime,
  ) as EmbeddingProviderChoice;

  let memorySearch: MemorySearchConfig = {
    ...existing,
    provider: providerChoice === "auto" ? undefined : providerChoice,
  };

  // Provider-specific prompts
  if (providerChoice === "openai" || providerChoice === "gemini" || providerChoice === "voyage") {
    const defaultModel = DEFAULT_MODELS[providerChoice] ?? "";
    const modelInput = guardCancel(
      await text({
        message: `Embedding model (${providerChoice})`,
        initialValue: existing?.model ?? defaultModel,
        placeholder: defaultModel,
      }),
      runtime,
    );
    const model = String(modelInput ?? "").trim() || defaultModel;
    memorySearch = { ...memorySearch, model };

    const hasApiKey = Boolean(existing?.remote?.apiKey);
    const apiKeyInput = guardCancel(
      await text({
        message: hasApiKey
          ? "Embedding API key (leave blank to keep current)"
          : "Embedding API key (leave blank to use provider env var)",
        placeholder: hasApiKey ? "Leave blank to keep current" : "sk-...",
      }),
      runtime,
    );
    const apiKey = String(apiKeyInput ?? "").trim();
    if (apiKey) {
      memorySearch = {
        ...memorySearch,
        remote: { ...memorySearch.remote, apiKey },
      };
    }

    if (providerChoice === "openai") {
      const hasBaseUrl = Boolean(existing?.remote?.baseUrl);
      const baseUrlInput = guardCancel(
        await text({
          message: "Custom base URL (leave blank for default OpenAI endpoint)",
          initialValue: existing?.remote?.baseUrl ?? "",
          placeholder: "https://api.openai.com/v1",
        }),
        runtime,
      );
      const baseUrl = String(baseUrlInput ?? "").trim();
      if (baseUrl) {
        memorySearch = {
          ...memorySearch,
          remote: { ...memorySearch.remote, baseUrl },
        };
      } else if (hasBaseUrl) {
        // Clear previous custom URL
        const { baseUrl: _, ...rest } = memorySearch.remote ?? {};
        memorySearch = { ...memorySearch, remote: rest };
      }
    }
  } else if (providerChoice === "local") {
    const modelPathInput = guardCancel(
      await text({
        message: "Local GGUF model path or hf: URI",
        initialValue:
          existing?.local?.modelPath ??
          "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf",
        placeholder: "hf:ggml-org/...",
      }),
      runtime,
    );
    const modelPath = String(modelPathInput ?? "").trim();
    if (modelPath) {
      memorySearch = {
        ...memorySearch,
        local: { ...memorySearch.local, modelPath },
      };
    }
  }
  // "script" and "auto" need no extra prompts

  // --- Fallback ---
  if (providerChoice !== "auto" && providerChoice !== "script") {
    const fallbackChoice = guardCancel(
      await select({
        message: "Fallback if primary embedding fails",
        options: [
          { value: "script", label: "Local (script)", hint: "Hash vectorizer, always available" },
          { value: "local", label: "Local (node-llama-cpp)", hint: "On-device GGUF model" },
          { value: "none", label: "None", hint: "Fall back to keyword-only search" },
        ],
        initialValue: existing?.fallback ?? "script",
      }),
      runtime,
    );
    memorySearch = { ...memorySearch, fallback: fallbackChoice as MemorySearchConfig["fallback"] };
  }

  // --- Reranking (optional) ---
  const enableReranker = guardCancel(
    await confirm({
      message: "Enable reranking? (improves search quality via a reranker model)",
      initialValue: existing?.query?.hybrid?.reranker?.enabled ?? false,
    }),
    runtime,
  );

  if (enableReranker) {
    memorySearch = await promptRerankerConfig(memorySearch, existing, {
      select: async (params) => guardCancel(await select(params), runtime),
      text: async (params) => String(guardCancel(await text(params), runtime) ?? ""),
    });
  } else if (existing?.query?.hybrid?.reranker?.enabled) {
    // Disable previously enabled reranker
    memorySearch = {
      ...memorySearch,
      query: {
        ...memorySearch.query,
        hybrid: {
          ...memorySearch.query?.hybrid,
          reranker: {
            ...memorySearch.query?.hybrid?.reranker,
            enabled: false,
          },
        },
      },
    };
  }

  return {
    ...nextConfig,
    agents: {
      ...nextConfig.agents,
      defaults: {
        ...nextConfig.agents?.defaults,
        memorySearch,
      },
    },
  };
}

/**
 * Onboarding variant using WizardPrompter (supports non-interactive and TUI prompters).
 */
export async function promptMemorySearchForOnboarding(params: {
  config: MarvConfig;
  prompter: WizardPrompter;
}): Promise<MarvConfig> {
  const { prompter } = params;
  const existing = params.config.agents?.defaults?.memorySearch;

  await prompter.note(
    [
      "Memory search uses embeddings to find relevant memories.",
      "Remote providers (OpenAI, Gemini, Voyage) need an API key.",
      "Local (script) is a zero-dependency fallback that always works.",
    ].join("\n"),
    "Memory search",
  );

  const currentProvider = existing?.provider ?? "auto";
  const providerChoice = await prompter.select<EmbeddingProviderChoice>({
    message: "Embedding provider",
    options: EMBEDDING_PROVIDER_OPTIONS,
    initialValue: currentProvider as EmbeddingProviderChoice,
  });

  let memorySearch: MemorySearchConfig = {
    ...existing,
    provider: providerChoice === "auto" ? undefined : providerChoice,
  };

  if (providerChoice === "openai" || providerChoice === "gemini" || providerChoice === "voyage") {
    const defaultModel = DEFAULT_MODELS[providerChoice] ?? "";
    const modelInput = await prompter.text({
      message: `Embedding model (${providerChoice})`,
      initialValue: existing?.model ?? defaultModel,
      placeholder: defaultModel,
    });
    memorySearch = { ...memorySearch, model: modelInput.trim() || defaultModel };

    const hasApiKey = Boolean(existing?.remote?.apiKey);
    const apiKeyInput = await prompter.text({
      message: hasApiKey
        ? "Embedding API key (leave blank to keep current)"
        : "Embedding API key (leave blank to use provider env var)",
      placeholder: hasApiKey ? "Leave blank to keep current" : "sk-...",
    });
    const apiKey = apiKeyInput.trim();
    if (apiKey) {
      memorySearch = {
        ...memorySearch,
        remote: { ...memorySearch.remote, apiKey },
      };
    }
  } else if (providerChoice === "local") {
    const modelPathInput = await prompter.text({
      message: "Local GGUF model path or hf: URI",
      initialValue:
        existing?.local?.modelPath ??
        "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf",
      placeholder: "hf:ggml-org/...",
    });
    const modelPath = modelPathInput.trim();
    if (modelPath) {
      memorySearch = {
        ...memorySearch,
        local: { ...memorySearch.local, modelPath },
      };
    }
  }

  // Fallback
  if (providerChoice !== "auto" && providerChoice !== "script") {
    const fallbackChoice = await prompter.select({
      message: "Fallback if primary embedding fails",
      options: [
        { value: "script", label: "Local (script)", hint: "Hash vectorizer, always available" },
        { value: "local", label: "Local (node-llama-cpp)", hint: "On-device GGUF model" },
        { value: "none", label: "None", hint: "Fall back to keyword-only search" },
      ],
      initialValue: existing?.fallback ?? "script",
    });
    memorySearch = { ...memorySearch, fallback: fallbackChoice as MemorySearchConfig["fallback"] };
  }

  // Reranking
  const enableReranker = await prompter.confirm({
    message: "Enable reranking? (optional, improves search quality via a reranker model)",
    initialValue: existing?.query?.hybrid?.reranker?.enabled ?? false,
  });

  if (enableReranker) {
    memorySearch = await promptRerankerConfig(memorySearch, existing, {
      select: async (params) => await prompter.select(params),
      text: async (params) => await prompter.text(params),
    });
  } else if (existing?.query?.hybrid?.reranker?.enabled) {
    memorySearch = {
      ...memorySearch,
      query: {
        ...memorySearch.query,
        hybrid: {
          ...memorySearch.query?.hybrid,
          reranker: {
            ...memorySearch.query?.hybrid?.reranker,
            enabled: false,
          },
        },
      },
    };
  }

  return {
    ...params.config,
    agents: {
      ...params.config.agents,
      defaults: {
        ...params.config.agents?.defaults,
        memorySearch,
      },
    },
  };
}
