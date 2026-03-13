export type MediaRefScope = "inbound" | "outbound" | "browser" | "plugin" | "transient";

export type MediaRefLifecycle = "transient" | "session" | "hosted";

export type MediaStoragePreset = "transient" | "inbound" | "outbound" | "browser" | "hosted";

export type MediaStorageTarget =
  | {
      kind: "transient";
      subdir: "";
      scope: "transient";
      lifecycle: "transient";
    }
  | {
      kind: "inbound";
      subdir: "inbound";
      scope: "inbound";
      lifecycle: "session";
    }
  | {
      kind: "outbound";
      subdir: "outbound";
      scope: "outbound";
      lifecycle: "session";
    }
  | {
      kind: "browser";
      subdir: "browser";
      scope: "browser";
      lifecycle: "session";
    }
  | {
      kind: "hosted";
      subdir: "";
      scope: "outbound";
      lifecycle: "hosted";
    }
  | {
      kind: "plugin";
      pluginId: string;
      subdir: `plugin/${string}`;
      scope: "plugin";
      lifecycle: "session";
    };

export type MediaStorageHandle = MediaStoragePreset | { pluginId: string };

export type StoredMediaRef = {
  id: string;
  path: string;
  size: number;
  contentType?: string;
  scope: MediaRefScope;
  lifecycle: MediaRefLifecycle;
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
