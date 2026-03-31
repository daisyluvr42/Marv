import { fetchWithPrivateNetworkAccess } from "../../infra/net/private-network-fetch.js";
import { hashText } from "../internal.js";
import {
  buildBatchHeaders,
  normalizeBatchBaseUrl,
  type BatchHttpClientConfig,
} from "./batch-utils.js";

export async function uploadBatchJsonlFile(params: {
  client: BatchHttpClientConfig;
  requests: unknown[];
  errorPrefix: string;
}): Promise<string> {
  const baseUrl = normalizeBatchBaseUrl(params.client);
  const jsonl = params.requests.map((request) => JSON.stringify(request)).join("\n");
  const form = new FormData();
  form.append("purpose", "batch");
  form.append(
    "file",
    new Blob([jsonl], { type: "application/jsonl" }),
    `memory-embeddings.${hashText(String(Date.now()))}.jsonl`,
  );

  const { response, release } = await fetchWithPrivateNetworkAccess({
    url: `${baseUrl}/files`,
    init: {
      method: "POST",
      headers: buildBatchHeaders(params.client, { json: false }),
      body: form,
    },
    auditContext: "memory.embeddings.batch.upload",
  });
  try {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${params.errorPrefix}: ${response.status} ${text}`);
    }
    const filePayload = (await response.json()) as { id?: string };
    if (!filePayload.id) {
      throw new Error(`${params.errorPrefix}: missing file id`);
    }
    return filePayload.id;
  } finally {
    await release();
  }
}
