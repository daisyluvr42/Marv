import { clearStoredDeviceAuth } from "./device-auth.js";
import { clearStoredDeviceIdentity } from "./device-identity.js";
import type { UiSettings } from "./storage.js";

export type TrustedDeviceHost = {
  settings: UiSettings;
  password: string;
  hello: unknown;
  lastError: string | null;
  connected: boolean;
  client?: { stop?: () => void } | null;
  applySettings: (next: UiSettings) => void;
};

export function forgetTrustedDevice(host: TrustedDeviceHost) {
  clearStoredDeviceAuth();
  clearStoredDeviceIdentity();
  host.password = "";
  host.hello = null;
  host.lastError = null;
  host.connected = false;
  host.client?.stop?.();
  host.applySettings({
    ...host.settings,
    token: "",
  });
}
