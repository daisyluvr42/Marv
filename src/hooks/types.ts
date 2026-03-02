export type HookInstallSpec = {
  id?: string;
  kind: "bundled" | "npm" | "git";
  label?: string;
  package?: string;
  repository?: string;
  bins?: string[];
};

export type MarvHookMetadata = {
  always?: boolean;
  hookKey?: string;
  emoji?: string;
  homepage?: string;
  /** Events this hook handles (e.g., ["command:new", "session:start"]) */
  events: string[];
  /** Optional export name (default: "default") */
  export?: string;
  os?: string[];
  requires?: {
    bins?: string[];
    anyBins?: string[];
    env?: string[];
    config?: string[];
  };
  install?: HookInstallSpec[];
};

export type MarvHookMetadata = MarvHookMetadata;

export type HookInvocationPolicy = {
  enabled: boolean;
};

export type ParsedHookFrontmatter = Record<string, string>;

export type MarvHookSource = "marv-bundled" | "marv-managed" | "marv-workspace" | "marv-plugin";

export type LegacyMarvHookSource =
  | "marv-bundled"
  | "marv-managed"
  | "marv-workspace"
  | "marv-plugin";

export type HookSource = MarvHookSource | LegacyMarvHookSource;

const LEGACY_HOOK_SOURCE_ALIASES: Record<LegacyMarvHookSource, MarvHookSource> = {
  "marv-bundled": "marv-bundled",
  "marv-managed": "marv-managed",
  "marv-workspace": "marv-workspace",
  "marv-plugin": "marv-plugin",
};

export function normalizeHookSource(source: HookSource): MarvHookSource {
  return LEGACY_HOOK_SOURCE_ALIASES[source as LegacyMarvHookSource] ?? source;
}

export function isPluginHookSource(source: HookSource): boolean {
  return normalizeHookSource(source) === "marv-plugin";
}

export type Hook = {
  name: string;
  description: string;
  source: HookSource;
  pluginId?: string;
  filePath: string; // Path to HOOK.md
  baseDir: string; // Directory containing hook
  handlerPath: string; // Path to handler module (handler.ts/js)
};

export type HookEntry = {
  hook: Hook;
  frontmatter: ParsedHookFrontmatter;
  metadata?: MarvHookMetadata;
  invocation?: HookInvocationPolicy;
};

export type HookEligibilityContext = {
  remote?: {
    platforms: string[];
    hasBin: (bin: string) => boolean;
    hasAnyBin: (bins: string[]) => boolean;
    note?: string;
  };
};

export type HookSnapshot = {
  hooks: Array<{ name: string; events: string[] }>;
  resolvedHooks?: Hook[];
  version?: number;
};
