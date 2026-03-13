export type MediaRefScope = "inbound" | "outbound" | "browser" | "plugin" | "transient";

export type MediaRefLifecycle = "transient" | "session" | "hosted";

export type StoredMediaRef = {
  id: string;
  path: string;
  size: number;
  contentType?: string;
  scope?: MediaRefScope;
  lifecycle?: MediaRefLifecycle;
};

export type HostedMediaRef = {
  id: string;
  url: string;
  size?: number;
  expiresAt?: number;
};

export type PromptMediaKind = "image" | "audio" | "video" | "file";

export type PromptMediaRef = {
  attachmentIndex?: number;
  kind: PromptMediaKind;
  source: "native" | "derived";
  path?: string;
  url?: string;
  contentType?: string;
};
