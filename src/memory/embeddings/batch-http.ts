import { fetchWithPrivateNetworkAccess } from "../../infra/net/private-network-fetch.js";
import { retryAsync } from "../../infra/retry.js";

export async function postJsonWithRetry<T>(params: {
  url: string;
  headers: Record<string, string>;
  body: unknown;
  errorPrefix: string;
}): Promise<T> {
  const res = await retryAsync(
    async () => {
      const { response, release } = await fetchWithPrivateNetworkAccess({
        url: params.url,
        init: {
          method: "POST",
          headers: params.headers,
          body: JSON.stringify(params.body),
        },
        auditContext: "memory.embeddings.batch.post",
      });
      try {
        if (!response.ok) {
          const text = await response.text();
          const err = new Error(`${params.errorPrefix}: ${response.status} ${text}`) as Error & {
            status?: number;
          };
          err.status = response.status;
          throw err;
        }
        return (await response.json()) as T;
      } finally {
        await release();
      }
    },
    {
      attempts: 3,
      minDelayMs: 300,
      maxDelayMs: 2000,
      jitter: 0.2,
      shouldRetry: (err) => {
        const status = (err as { status?: number }).status;
        return status === 429 || (typeof status === "number" && status >= 500);
      },
    },
  );
  return res;
}
