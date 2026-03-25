import fs from "node:fs/promises";
import path from "node:path";
import { resolveConfigDir } from "../../../utils.js";
import type {
  ManagedCliLifecycleState,
  ManagedCliManifest,
  ManagedCliProfileRecord,
  ManagedCliRegistry,
  ManagedCliRegistryEntry,
} from "./cli-profile-types.js";

const REGISTRY_VERSION = 1 as const;
const PROFILE_ID_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const MAX_PROFILE_ID_LENGTH = 64;

function nowIso(): string {
  return new Date().toISOString();
}

export function validateManagedCliProfileId(raw: string): string {
  const id = raw.trim();
  if (!id) {
    throw new Error("CLI profile id is required.");
  }
  if (id.length > MAX_PROFILE_ID_LENGTH) {
    throw new Error(`CLI profile id must be ${MAX_PROFILE_ID_LENGTH} characters or fewer.`);
  }
  if (!PROFILE_ID_RE.test(id)) {
    throw new Error("CLI profile id must use hyphen-case letters and numbers only.");
  }
  return id;
}

export function resolveManagedCliRootDir(baseDir?: string): string {
  return path.join(baseDir ?? resolveConfigDir(), "tools", "managed-cli");
}

export function resolveManagedCliRegistryPath(baseDir?: string): string {
  return path.join(resolveManagedCliRootDir(baseDir), "registry.json");
}

export function resolveManagedCliToolDir(profileId: string, baseDir?: string): string {
  return path.join(resolveManagedCliRootDir(baseDir), validateManagedCliProfileId(profileId));
}

export function resolveManagedCliManifestPath(profileId: string, baseDir?: string): string {
  return path.join(resolveManagedCliToolDir(profileId, baseDir), "manifest.json");
}

function normalizeRegistry(raw: unknown): ManagedCliRegistry {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { version: REGISTRY_VERSION, profiles: {} };
  }
  const parsed = raw as Partial<ManagedCliRegistry>;
  const profiles =
    parsed.profiles && typeof parsed.profiles === "object" && !Array.isArray(parsed.profiles)
      ? parsed.profiles
      : {};
  const normalized: Record<string, ManagedCliRegistryEntry> = {};
  for (const [key, value] of Object.entries(profiles)) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      continue;
    }
    const entry = value as Partial<ManagedCliRegistryEntry>;
    const id = typeof entry.id === "string" && entry.id.trim() ? entry.id.trim() : key;
    if (!PROFILE_ID_RE.test(id)) {
      continue;
    }
    const manifestPath =
      typeof entry.manifestPath === "string" && entry.manifestPath.trim()
        ? entry.manifestPath.trim()
        : resolveManagedCliManifestPath(id);
    const state = normalizeLifecycleState(entry.state);
    const tier = entry.tier === "full-cli" ? "full-cli" : "script-wrapper";
    normalized[id] = {
      id,
      manifestPath,
      state,
      tier,
      name: typeof entry.name === "string" && entry.name.trim() ? entry.name.trim() : id,
      description:
        typeof entry.description === "string" && entry.description.trim()
          ? entry.description.trim()
          : "",
      capabilities: Array.isArray(entry.capabilities)
        ? entry.capabilities.filter((value): value is string => typeof value === "string")
        : [],
      createdAt:
        typeof entry.createdAt === "string" && entry.createdAt.trim() ? entry.createdAt : nowIso(),
      updatedAt:
        typeof entry.updatedAt === "string" && entry.updatedAt.trim() ? entry.updatedAt : nowIso(),
    };
  }
  return { version: REGISTRY_VERSION, profiles: normalized };
}

export function normalizeLifecycleState(raw: unknown): ManagedCliLifecycleState {
  if (
    raw === "draft" ||
    raw === "verified" ||
    raw === "active" ||
    raw === "quarantined" ||
    raw === "deprecated"
  ) {
    return raw;
  }
  return "draft";
}

export async function readManagedCliRegistry(baseDir?: string): Promise<ManagedCliRegistry> {
  const registryPath = resolveManagedCliRegistryPath(baseDir);
  try {
    const raw = await fs.readFile(registryPath, "utf8");
    return normalizeRegistry(JSON.parse(raw));
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | undefined)?.code;
    if (code === "ENOENT") {
      return { version: REGISTRY_VERSION, profiles: {} };
    }
    throw error;
  }
}

export async function writeManagedCliRegistry(
  registry: ManagedCliRegistry,
  baseDir?: string,
): Promise<void> {
  const registryPath = resolveManagedCliRegistryPath(baseDir);
  await fs.mkdir(path.dirname(registryPath), { recursive: true });
  await fs.writeFile(registryPath, `${JSON.stringify(registry, null, 2)}\n`, "utf8");
}

export async function readManagedCliManifest(manifestPath: string): Promise<ManagedCliManifest> {
  const raw = await fs.readFile(manifestPath, "utf8");
  const parsed = JSON.parse(raw) as ManagedCliManifest;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Invalid CLI manifest: ${manifestPath}`);
  }
  if (parsed.manifestVersion !== 1) {
    throw new Error(`Unsupported CLI manifest version at ${manifestPath}`);
  }
  return parsed;
}

export async function writeManagedCliManifest(manifest: ManagedCliManifest): Promise<void> {
  await fs.mkdir(path.dirname(path.resolve(manifest.toolDir)), { recursive: true });
  const manifestPath = path.join(manifest.toolDir, "manifest.json");
  await fs.mkdir(manifest.toolDir, { recursive: true });
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

export async function registerManagedCliManifest(params: {
  manifest: ManagedCliManifest;
  baseDir?: string;
}): Promise<ManagedCliRegistryEntry> {
  const id = validateManagedCliProfileId(params.manifest.id);
  const registry = await readManagedCliRegistry(params.baseDir);
  const manifestPath = path.join(params.manifest.toolDir, "manifest.json");
  const entry: ManagedCliRegistryEntry = {
    id,
    manifestPath,
    state: normalizeLifecycleState(params.manifest.lifecycle.state),
    tier: params.manifest.tier,
    name: params.manifest.name,
    description: params.manifest.description,
    capabilities: params.manifest.capabilities ?? [],
    createdAt: params.manifest.createdAt,
    updatedAt: params.manifest.updatedAt,
  };
  registry.profiles[id] = entry;
  await writeManagedCliRegistry(registry, params.baseDir);
  return entry;
}

export async function loadManagedCliProfile(params: {
  profileId: string;
  baseDir?: string;
}): Promise<ManagedCliProfileRecord | null> {
  const profileId = validateManagedCliProfileId(params.profileId);
  const registry = await readManagedCliRegistry(params.baseDir);
  const entry = registry.profiles[profileId];
  if (!entry) {
    return null;
  }
  const manifest = await readManagedCliManifest(entry.manifestPath);
  return { entry, manifest };
}

export async function listManagedCliProfiles(params?: {
  baseDir?: string;
}): Promise<ManagedCliProfileRecord[]> {
  const registry = await readManagedCliRegistry(params?.baseDir);
  const out: ManagedCliProfileRecord[] = [];
  for (const entry of Object.values(registry.profiles).toSorted((a, b) =>
    a.id.localeCompare(b.id),
  )) {
    try {
      const manifest = await readManagedCliManifest(entry.manifestPath);
      out.push({ entry, manifest });
    } catch {
      continue;
    }
  }
  return out;
}

export async function updateManagedCliProfileState(params: {
  profileId: string;
  state: ManagedCliLifecycleState;
  baseDir?: string;
  verificationError?: string;
}): Promise<ManagedCliProfileRecord> {
  const record = await loadManagedCliProfile(params);
  if (!record) {
    throw new Error(`CLI profile not found: ${params.profileId}`);
  }
  const timestamp = nowIso();
  record.manifest.lifecycle.state = params.state;
  record.manifest.updatedAt = timestamp;
  if (record.manifest.verification) {
    record.manifest.verification.lastVerifiedAt = timestamp;
    if (params.verificationError !== undefined) {
      record.manifest.verification.lastResult = params.verificationError ? "fail" : "pass";
      record.manifest.verification.lastError = params.verificationError || undefined;
    }
  }
  await writeManagedCliManifest(record.manifest);
  const registry = await readManagedCliRegistry(params.baseDir);
  const existing = registry.profiles[record.entry.id];
  if (!existing) {
    throw new Error(`CLI profile registry entry missing: ${params.profileId}`);
  }
  existing.state = params.state;
  existing.updatedAt = timestamp;
  registry.profiles[record.entry.id] = existing;
  await writeManagedCliRegistry(registry, params.baseDir);
  return {
    entry: existing,
    manifest: record.manifest,
  };
}

export async function removeManagedCliProfile(params: {
  profileId: string;
  baseDir?: string;
}): Promise<void> {
  const profileId = validateManagedCliProfileId(params.profileId);
  const registry = await readManagedCliRegistry(params.baseDir);
  delete registry.profiles[profileId];
  await writeManagedCliRegistry(registry, params.baseDir);
}
