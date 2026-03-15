/**
 * Zero-dependency local embedding fallback using hash-based vectorization.
 *
 * Uses character n-gram hashing (the "hashing trick") to produce fixed-dimension
 * vectors without any external model or API. Quality is lower than dedicated
 * embedding models but it always works and is fast.
 *
 * Algorithm:
 * 1. Normalize text (lowercase, collapse whitespace).
 * 2. Extract overlapping character n-grams (trigrams + 5-grams).
 * 3. Hash each n-gram to a dimension index using FNV-1a.
 * 4. Accumulate signed counts (hash bit determines sign for variance).
 * 5. L2-normalize the resulting vector.
 */

import type { EmbeddingProvider } from "./embeddings.js";

const SCRIPT_DIMENSIONS = 384;
const TRIGRAM_N = 3;
const FIVEGRAM_N = 5;

// FNV-1a hash (32-bit) — fast, well-distributed, zero-dependency.
function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function extractNgrams(text: string, n: number): string[] {
  const ngrams: string[] = [];
  if (text.length < n) {
    if (text.length > 0) {
      ngrams.push(text);
    }
    return ngrams;
  }
  for (let i = 0; i <= text.length - n; i++) {
    ngrams.push(text.slice(i, i + n));
  }
  return ngrams;
}

function hashVectorize(text: string): number[] {
  const normalized = normalizeText(text);
  const vec = new Float64Array(SCRIPT_DIMENSIONS);

  const trigrams = extractNgrams(normalized, TRIGRAM_N);
  const fivegrams = extractNgrams(normalized, FIVEGRAM_N);

  for (const ngram of trigrams) {
    const hash = fnv1a(ngram);
    const index = hash % SCRIPT_DIMENSIONS;
    // Use a higher bit to determine sign for better variance
    const sign = hash & 0x8000 ? -1 : 1;
    vec[index] += sign;
  }

  // 5-grams weighted higher for phrase-level signal
  for (const ngram of fivegrams) {
    const hash = fnv1a(ngram);
    const index = hash % SCRIPT_DIMENSIONS;
    const sign = hash & 0x8000 ? -1 : 1;
    vec[index] += sign * 1.5;
  }

  // Also hash word unigrams for term-level signal
  const words = normalized.split(/\s+/).filter((w) => w.length > 1);
  for (const word of words) {
    const hash = fnv1a(`w:${word}`);
    const index = hash % SCRIPT_DIMENSIONS;
    const sign = hash & 0x8000 ? -1 : 1;
    vec[index] += sign * 2;
  }

  // Word bigrams for context
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]} ${words[i + 1]}`;
    const hash = fnv1a(`b:${bigram}`);
    const index = hash % SCRIPT_DIMENSIONS;
    const sign = hash & 0x8000 ? -1 : 1;
    vec[index] += sign * 1.5;
  }

  // L2 normalize
  let magnitude = 0;
  for (let i = 0; i < SCRIPT_DIMENSIONS; i++) {
    magnitude += vec[i] * vec[i];
  }
  magnitude = Math.sqrt(magnitude);
  const result = Array.from<number>({ length: SCRIPT_DIMENSIONS });
  if (magnitude < 1e-10) {
    result.fill(0);
  } else {
    for (let i = 0; i < SCRIPT_DIMENSIONS; i++) {
      result[i] = vec[i] / magnitude;
    }
  }
  return result;
}

export function createScriptEmbeddingProvider(): EmbeddingProvider {
  return {
    id: "script",
    model: "hash-vectorizer",
    dimensions: SCRIPT_DIMENSIONS,
    embedQuery: async (text: string) => hashVectorize(text),
    embedBatch: async (texts: string[]) => texts.map((t) => hashVectorize(t)),
  };
}
