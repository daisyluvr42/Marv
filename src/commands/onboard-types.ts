import type { ChannelId } from "../channels/plugins/types.js";
import type { GatewayDaemonRuntime } from "./daemon-runtime.js";

export type OnboardMode = "local" | "remote";
export type AuthChoice =
  | "token"
  | "chutes"
  | "ollama"
  | "vllm"
  | "openai-codex"
  | "openai-api-key"
  | "openrouter-api-key"
  | "litellm-api-key"
  | "ai-gateway-api-key"
  | "cloudflare-ai-gateway-api-key"
  | "moonshot-api-key"
  | "moonshot-api-key-cn"
  | "kimi-code-api-key"
  | "synthetic-api-key"
  | "venice-api-key"
  | "together-api-key"
  | "huggingface-api-key"
  | "apiKey"
  | "gemini-api-key"
  | "google-antigravity"
  | "google-gemini-cli"
  | "zai-api-key"
  | "zai-coding-global"
  | "zai-coding-cn"
  | "zai-global"
  | "zai-cn"
  | "xiaomi-api-key"
  | "minimax"
  | "minimax-api"
  | "minimax-api-key-cn"
  | "minimax-api-lightning"
  | "minimax-portal"
  | "opencode-zen"
  | "github-copilot"
  | "copilot-proxy"
  | "qwen-portal"
  | "xai-api-key"
  | "qianfan-api-key"
  | "custom-api-key"
  | "skip";
export type AuthChoiceGroupId =
  | "openai"
  | "anthropic"
  | "chutes"
  | "ollama"
  | "vllm"
  | "google"
  | "copilot"
  | "openrouter"
  | "litellm"
  | "ai-gateway"
  | "cloudflare-ai-gateway"
  | "moonshot"
  | "zai"
  | "xiaomi"
  | "opencode-zen"
  | "minimax"
  | "synthetic"
  | "venice"
  | "qwen"
  | "together"
  | "huggingface"
  | "qianfan"
  | "xai"
  | "custom";
export type GatewayAuthChoice = "token" | "password";
export type ResetScope = "config" | "config+creds+sessions" | "full";
export type GatewayBind = "loopback" | "lan" | "auto" | "custom" | "tailnet";
export type TailscaleMode = "off" | "serve" | "funnel";
export type NodeManagerChoice = "npm" | "pnpm" | "bun";
export type ChannelChoice = ChannelId;

export type OnboardOptions = {
  mode?: OnboardMode;
  flow?: "quickstart" | "advanced";
  workspace?: string;
  p0Soul?: string;
  p0Identity?: string;
  p0User?: string;
  nonInteractive?: boolean;
  /** Required for non-interactive onboarding; skips the interactive risk prompt when true. */
  acceptRisk?: boolean;
  reset?: boolean;
  authChoice?: AuthChoice;
  /** Used when `authChoice=token` in non-interactive mode. */
  tokenProvider?: string;
  /** Used when `authChoice=token` in non-interactive mode. */
  token?: string;
  /** Used when `authChoice=token` in non-interactive mode. */
  tokenProfileId?: string;
  /** Used when `authChoice=token` in non-interactive mode. */
  tokenExpiresIn?: string;
  anthropicApiKey?: string;
  openaiApiKey?: string;
  openrouterApiKey?: string;
  litellmApiKey?: string;
  aiGatewayApiKey?: string;
  cloudflareAiGatewayAccountId?: string;
  cloudflareAiGatewayGatewayId?: string;
  cloudflareAiGatewayApiKey?: string;
  moonshotApiKey?: string;
  kimiCodeApiKey?: string;
  geminiApiKey?: string;
  zaiApiKey?: string;
  xiaomiApiKey?: string;
  minimaxApiKey?: string;
  syntheticApiKey?: string;
  veniceApiKey?: string;
  togetherApiKey?: string;
  huggingfaceApiKey?: string;
  opencodeZenApiKey?: string;
  xaiApiKey?: string;
  qianfanApiKey?: string;
  customBaseUrl?: string;
  customApiKey?: string;
  customModelId?: string;
  customProviderId?: string;
  customCompatibility?: "openai" | "anthropic";
  gatewayPort?: number;
  gatewayBind?: GatewayBind;
  gatewayAuth?: GatewayAuthChoice;
  gatewayToken?: string;
  gatewayPassword?: string;
  tailscale?: TailscaleMode;
  tailscaleResetOnExit?: boolean;
  installDaemon?: boolean;
  daemonRuntime?: GatewayDaemonRuntime;
  skipChannels?: boolean;
  skipSkills?: boolean;
  skipHealth?: boolean;
  skipUi?: boolean;
  /** Enable memory search in non-interactive onboarding (default: false). */
  enableMemorySearch?: boolean;
  /** Memory search provider for non-interactive onboarding (e.g. "local", "openai", "gemini"). */
  memorySearchProvider?: string;
  /** Enable subagent auto-routing in non-interactive onboarding (default: false). */
  enableAutoRouting?: boolean;
  nodeManager?: NodeManagerChoice;
  remoteUrl?: string;
  remoteToken?: string;
  json?: boolean;
};
