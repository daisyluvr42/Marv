import { fetchWithPrivateNetworkAccess } from "../../infra/net/private-network-fetch.js";

export async function fetchRemoteEmbeddingVectors(params: {
  url: string;
  headers: Record<string, string>;
  body: unknown;
  errorPrefix: string;
}): Promise<number[][]> {
  const { response, release } = await fetchWithPrivateNetworkAccess({
    url: params.url,
    init: {
      method: "POST",
      headers: params.headers,
      body: JSON.stringify(params.body),
    },
    auditContext: "memory.embeddings.remote",
  });
  try {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${params.errorPrefix}: ${response.status} ${text}`);
    }
    const payload = (await response.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };
    const data = payload.data ?? [];
    return data.map((entry) => entry.embedding ?? []);
  } finally {
    await release();
  }
}
