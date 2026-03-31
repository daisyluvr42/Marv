import {
  fetchWithSsrFGuard,
  type GuardedFetchOptions,
  type GuardedFetchResult,
} from "./fetch-guard.js";
import type { SsrFPolicy } from "./ssrf.js";

export const PRIVATE_NETWORK_FETCH_POLICY: SsrFPolicy = {
  allowPrivateNetwork: true,
};

export async function fetchWithPrivateNetworkAccess(
  params: Omit<GuardedFetchOptions, "policy">,
): Promise<GuardedFetchResult> {
  return await fetchWithSsrFGuard({
    ...params,
    policy: PRIVATE_NETWORK_FETCH_POLICY,
  });
}
