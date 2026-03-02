import type { MarvConfig } from "../config/config.js";

export function resolveGatewayProbeAuth(params: {
  cfg: MarvConfig;
  mode: "local" | "remote";
  env?: NodeJS.ProcessEnv;
}): { token?: string; password?: string } {
  const env = params.env ?? process.env;
  const authToken = params.cfg.gateway?.auth?.token;
  const authPassword = params.cfg.gateway?.auth?.password;
  const remote = params.cfg.gateway?.remote;

  const token =
    params.mode === "remote"
      ? typeof remote?.token === "string" && remote.token.trim()
        ? remote.token.trim()
        : undefined
      : env.MARV_GATEWAY_TOKEN?.trim() ||
        env.MARV_GATEWAY_TOKEN?.trim() ||
        (typeof authToken === "string" && authToken.trim() ? authToken.trim() : undefined);

  const password =
    env.MARV_GATEWAY_PASSWORD?.trim() ||
    env.MARV_GATEWAY_PASSWORD?.trim() ||
    (params.mode === "remote"
      ? typeof remote?.password === "string" && remote.password.trim()
        ? remote.password.trim()
        : undefined
      : typeof authPassword === "string" && authPassword.trim()
        ? authPassword.trim()
        : undefined);

  return { token, password };
}
