export type HybridRerankerConfig = {
  enabled: boolean;
  apiUrl?: string;
  model?: string;
  apiKey?: string;
  maxCandidates: number;
  /** Parsed for compatibility; inactive in the first reranker release. */
  ftsFirst: boolean;
  timeoutMs?: number;
};

export const DEFAULT_RERANKER_CONFIG: HybridRerankerConfig = {
  enabled: false,
  maxCandidates: 24,
  ftsFirst: false,
  timeoutMs: 15_000,
};

type ParsedRerankerResult = {
  index: number;
  score: number;
};

function toFiniteScore(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseRerankerResults(payload: unknown, count: number): ParsedRerankerResult[] {
  const root = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object"
      ? ((payload as { results?: unknown; data?: unknown }).results ??
        (payload as { results?: unknown; data?: unknown }).data)
      : undefined;
  if (!Array.isArray(root)) {
    throw new Error("reranker response missing results array");
  }

  const parsed: ParsedRerankerResult[] = [];
  for (const item of root) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const rawIndex = (item as { index?: unknown }).index;
    const score =
      toFiniteScore((item as { relevance_score?: unknown }).relevance_score) ??
      toFiniteScore((item as { score?: unknown }).score) ??
      toFiniteScore((item as { relevanceScore?: unknown }).relevanceScore);
    if (
      typeof rawIndex !== "number" ||
      !Number.isInteger(rawIndex) ||
      rawIndex < 0 ||
      rawIndex >= count ||
      score === null
    ) {
      continue;
    }
    parsed.push({ index: rawIndex, score });
  }
  if (parsed.length === 0) {
    throw new Error("reranker response contained no valid ranked results");
  }
  return parsed;
}

export async function rerankHybridResults<
  T extends {
    id: string;
    score: number;
    snippet: string;
    rerankScore?: number;
  },
>(params: {
  query?: string;
  results: T[];
  reranker?: Partial<HybridRerankerConfig>;
  warn?: (message: string) => void;
}): Promise<T[]> {
  const reranker = { ...DEFAULT_RERANKER_CONFIG, ...params.reranker };
  const query = params.query?.trim();
  const apiUrl = reranker.apiUrl?.trim();
  const model = reranker.model?.trim();

  if (!reranker.enabled || !query || !apiUrl || !model || params.results.length <= 1) {
    return [...params.results];
  }

  const topCount = Math.max(1, Math.min(params.results.length, reranker.maxCandidates));
  if (topCount <= 1) {
    return [...params.results];
  }

  const head = params.results.slice(0, topCount);
  const tail = params.results.slice(topCount);
  const controller = new AbortController();
  const timeoutMs = Math.max(1, reranker.timeoutMs ?? DEFAULT_RERANKER_CONFIG.timeoutMs!);
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(reranker.apiKey?.trim() ? { Authorization: `Bearer ${reranker.apiKey.trim()}` } : {}),
      },
      body: JSON.stringify({
        model,
        query,
        documents: head.map((result) => result.snippet),
        top_n: head.length,
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`reranker request failed: ${res.status} ${text}`);
    }

    const ranked = parseRerankerResults(await res.json(), head.length);
    const rankedByIndex = new Map(ranked.map((item) => [item.index, item.score]));
    const rerankedHead = head
      .map((result, index) => ({
        result,
        index,
        rerankScore: rankedByIndex.get(index),
      }))
      .toSorted((a, b) => {
        const scoreA = a.rerankScore ?? Number.NEGATIVE_INFINITY;
        const scoreB = b.rerankScore ?? Number.NEGATIVE_INFINITY;
        if (scoreA !== scoreB) {
          return scoreB - scoreA;
        }
        return a.index - b.index;
      })
      .map(({ result, rerankScore }) =>
        rerankScore === undefined ? result : { ...result, rerankScore },
      );
    return [...rerankedHead, ...tail];
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    params.warn?.(`memory reranker failed; using hybrid order: ${message}`);
    return [...params.results];
  } finally {
    clearTimeout(timer);
  }
}
