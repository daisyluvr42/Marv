export type ManagedCliOutputMode = "text" | "json" | "jsonl";

export type ManagedCliInputMode = "none" | "stdin";

export type ManagedCliTier = "script-wrapper" | "full-cli";

export type ManagedCliLifecycleState =
  | "draft"
  | "verified"
  | "active"
  | "quarantined"
  | "deprecated";

export type ManagedCliNetworkMode = "none" | "optional" | "required";

export type ManagedCliEntry = {
  command: string;
  staticArgs?: string[];
  argsTemplate?: string[];
  inputMode?: ManagedCliInputMode;
  outputMode: ManagedCliOutputMode;
  env?: Record<string, string>;
};

export type ManagedCliPolicy = {
  writesWorkspace?: boolean;
  network?: ManagedCliNetworkMode;
  sandboxSafe?: boolean;
};

export type ManagedCliVerification = {
  helpArgs?: string[];
  smokeArgs?: string[];
  lastVerifiedAt?: string;
  lastResult?: "pass" | "fail";
  lastError?: string;
};

export type ManagedCliSource = {
  kind: "script" | "command";
  originalPath?: string;
  originalCommand?: string;
};

export type ManagedCliManifest = {
  manifestVersion: 1;
  id: string;
  name: string;
  description: string;
  version: string;
  tier: ManagedCliTier;
  toolDir: string;
  createdAt: string;
  updatedAt: string;
  entry: ManagedCliEntry;
  capabilities?: string[];
  source?: ManagedCliSource;
  policy?: ManagedCliPolicy;
  verification?: ManagedCliVerification;
  lifecycle: {
    state: ManagedCliLifecycleState;
  };
};

export type ManagedCliRegistryEntry = {
  id: string;
  manifestPath: string;
  state: ManagedCliLifecycleState;
  tier: ManagedCliTier;
  name: string;
  description: string;
  capabilities: string[];
  createdAt: string;
  updatedAt: string;
};

export type ManagedCliRegistry = {
  version: 1;
  profiles: Record<string, ManagedCliRegistryEntry>;
};

export type ManagedCliProfileRecord = {
  entry: ManagedCliRegistryEntry;
  manifest: ManagedCliManifest;
};
