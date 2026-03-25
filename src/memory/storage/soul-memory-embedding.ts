import crypto from "node:crypto";
import { EMBEDDING_DIMS, ENTITY_EN_STOPWORDS, ENTITY_ZH_STOPWORDS } from "./soul-memory-types.js";

export function embedText(value: string): number[] {
  const tokens = tokenize(value);
  if (tokens.size === 0) {
    return [];
  }
  const vec = Array.from({ length: EMBEDDING_DIMS }, () => 0);
  for (const token of tokens) {
    const hash = crypto.createHash("sha256").update(token).digest();
    const indexA = hash[0] % EMBEDDING_DIMS;
    const indexB = hash[3] % EMBEDDING_DIMS;
    const signA = hash[1] % 2 === 0 ? 1 : -1;
    const signB = hash[4] % 2 === 0 ? 1 : -1;
    const weightA = 0.5 + hash[2] / 255;
    const weightB = 0.25 + hash[5] / 255;
    vec[indexA] = (vec[indexA] ?? 0) + signA * weightA;
    vec[indexB] = (vec[indexB] ?? 0) + signB * weightB;
  }
  const magnitude = Math.sqrt(vec.reduce((sum, entry) => sum + entry * entry, 0));
  if (magnitude <= 1e-10) {
    return vec;
  }
  return vec.map((entry) => entry / magnitude);
}

export function tokenize(value: string): Set<string> {
  const out = new Set<string>();
  const matches = value.toLowerCase().match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g) ?? [];
  for (const token of matches) {
    if (token) {
      out.add(token);
    }
  }
  return out;
}

export function extractEntities(value: string): Set<string> {
  const out = new Set<string>();
  const rawTokens =
    value.match(
      /[A-Z][A-Z0-9_-]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[\u4e00-\u9fff]{2,12}|[a-z0-9_]{4,}/g,
    ) ?? [];
  for (const raw of rawTokens) {
    const trimmed = raw.trim();
    if (!trimmed) {
      continue;
    }
    const hasChinese = /[\u4e00-\u9fff]/.test(trimmed);
    if (hasChinese) {
      if (ENTITY_ZH_STOPWORDS.has(trimmed)) {
        continue;
      }
      out.add(trimmed);
      continue;
    }
    const normalized = trimmed.toLowerCase();
    if (ENTITY_EN_STOPWORDS.has(normalized)) {
      continue;
    }
    if (/^[0-9]+$/.test(normalized)) {
      continue;
    }
    out.add(normalized);
  }
  return out;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA <= 0 || normB <= 0) {
    return 0;
  }
  return dot / Math.sqrt(normA * normB);
}

export function lexicalOverlap(a: string, b: string): number {
  const aTokens = tokenize(a);
  const bTokens = tokenize(b);
  if (aTokens.size === 0 || bTokens.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...aTokens, ...bTokens]).size;
  return union > 0 ? intersection / union : 0;
}

export function parseEmbedding(raw: string): number[] {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map((entry) => Number(entry)).filter((entry) => Number.isFinite(entry));
  } catch {
    return [];
  }
}

export function vectorToBlob(embedding: number[]): Buffer {
  return Buffer.from(new Float32Array(embedding).buffer);
}

export function buildFtsMatchQuery(query: string): string | null {
  const tokens = [...tokenize(query)].slice(0, 12);
  if (tokens.length === 0) {
    return null;
  }
  return tokens.map((token) => `"${token.replaceAll('"', '""')}"`).join(" OR ");
}
