export function extractLastJsonObject(raw: string): unknown {
  const trimmed = raw.trim();
  for (let start = trimmed.length - 1; start >= 0; start -= 1) {
    if (trimmed[start] !== "{") {
      continue;
    }
    try {
      return JSON.parse(trimmed.slice(start));
    } catch {
      continue;
    }
  }
  return null;
}

export function extractGeminiResponse(raw: string): string | null {
  const payload = extractLastJsonObject(raw);
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const response = (payload as { response?: unknown }).response;
  if (typeof response !== "string") {
    return null;
  }
  const trimmed = response.trim();
  return trimmed || null;
}
