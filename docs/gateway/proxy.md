---
summary: "Outbound proxy configuration: per-channel proxy routing, AI model proxies, and multi-account setups"
read_when:
  - Configuring outbound HTTP/SOCKS5 proxies for channels or model providers
  - Routing Telegram, Discord, or TTS traffic through a proxy
  - Debugging proxy connectivity issues
  - Setting up different proxies for different channels or accounts
title: "Proxy Configuration"
---

# Proxy Configuration

Marv supports per-channel outbound proxy configuration so that each channel
can route API traffic through a different HTTP, HTTPS, or SOCKS5 proxy.
Channels without a proxy configured use a direct connection.

## Channel Proxies

### Telegram

```bash
marv config set channels.telegram.proxy "http://127.0.0.1:7890"
```

Or in `marv.json`:

```json5
{
  channels: {
    telegram: {
      proxy: "http://127.0.0.1:7890",
    },
  },
}
```

Supports `http://`, `https://`, and `socks5://` URLs.
The proxy is used for all Telegram Bot API calls (polling, webhooks, media downloads).

### Discord

```bash
marv config set channels.discord.proxy "socks5://127.0.0.1:1080"
```

```json5
{
  channels: {
    discord: {
      proxy: "socks5://127.0.0.1:1080",
    },
  },
}
```

Covers Discord gateway WebSocket connections and REST API calls.

### TTS (Edge)

```bash
marv config set tts.edge.proxy "http://127.0.0.1:7890"
```

```json5
{
  tts: {
    edge: {
      proxy: "http://127.0.0.1:7890",
    },
  },
}
```

## Multi-Account Proxies

Each account within a channel can use a different proxy:

```json5
{
  channels: {
    telegram: {
      accounts: {
        personal: { proxy: "http://proxy-a:7890" },
        work: { proxy: "http://proxy-b:8080" },
      },
    },
    discord: {
      accounts: {
        main: { proxy: "socks5://proxy-c:1080" },
      },
    },
  },
}
```

Account-level `proxy` overrides the top-level channel `proxy` for that account.

## AI Model Request Proxies

Model provider API requests (OpenAI, Anthropic, etc.) do not have a dedicated
per-provider `proxy` config field. Two approaches:

### Environment Variable

Set `HTTPS_PROXY` or `ALL_PROXY` before starting the gateway.
This affects **all** outbound requests that do not have an explicit channel-level proxy:

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
marv gateway run
```

### Local Proxy Client with Split Routing (Recommended)

Use a local proxy client (Clash, V2Ray, Surge, etc.) with rule-based routing.
Point Marv at the local proxy port and let the proxy client decide which
requests go through which upstream, based on domain or IP rules.

This keeps Marv configuration simple while giving you full control over
routing logic outside the application.

## Common Scenarios

### Telegram and Discord via proxy, AI API direct

```bash
marv config set channels.telegram.proxy "http://127.0.0.1:7890"
marv config set channels.discord.proxy "http://127.0.0.1:7890"
# AI model requests use direct connection (no proxy set)
```

### Everything through one proxy

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
marv gateway run
```

### Different proxies for different channels

```bash
marv config set channels.telegram.proxy "http://proxy-a:7890"
marv config set channels.discord.proxy "socks5://proxy-b:1080"
# Other channels and model APIs go direct
```

## Verifying Connectivity

After configuring proxies, verify that channels can reach their APIs:

```bash
marv channels status --probe
```

The `--probe` flag performs live API checks for each configured channel
and reports connectivity status.

## Proxy Speed Testing

The repo includes a speed test script to compare direct vs proxied throughput:

```bash
scripts/proxy-speed-test.sh --proxy http://127.0.0.1:7890
scripts/proxy-speed-test.sh --socks5 127.0.0.1:1080 --rounds 5
```

## Technical Details

Channel proxies use [undici ProxyAgent](https://undici.nodejs.org/#/docs/api/ProxyAgent)
under the hood. Each channel creates an isolated `fetch` instance bound to its
proxy, so proxy failures in one channel do not affect others.
