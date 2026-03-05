# Proxy Configuration Reference

## Per-channel proxy

Each channel supports an independent `proxy` field (HTTP/HTTPS/SOCKS5 URL).
Channels without `proxy` use direct connections.

### Telegram

```bash
marv config set channels.telegram.proxy "http://127.0.0.1:7890"
```

JSON5:

```json5
{ channels: { telegram: { proxy: "http://127.0.0.1:7890" } } }
```

Covers: Bot API polling, webhooks, media downloads.

### Discord

```bash
marv config set channels.discord.proxy "socks5://127.0.0.1:1080"
```

Covers: gateway WebSocket + REST API.

### TTS (Edge)

```bash
marv config set tts.edge.proxy "http://127.0.0.1:7890"
```

## Multi-account proxies

Each account within a channel can override the top-level proxy:

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

Account-level `proxy` overrides channel-level `proxy` for that account.

## AI model request proxies

Model providers (OpenAI, Anthropic, etc.) have no per-provider `proxy` config field.

### Option 1: environment variable

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
marv gateway run
```

Affects all outbound requests without an explicit channel-level proxy.

### Option 2: local proxy client split routing (recommended)

Use Clash/V2Ray/Surge with domain-based rules. Marv points at local proxy port; routing logic is external.

## Common scenarios

| Scenario                                | Config                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------ |
| Telegram + Discord via proxy, AI direct | Set `channels.telegram.proxy` and `channels.discord.proxy`; leave model config alone |
| Everything through one proxy            | `export HTTPS_PROXY=...` before `marv gateway run`                                   |
| Different proxies per channel           | Set each channel's `proxy` independently                                             |

## Verification

```bash
marv channels status --probe    # live API checks per channel
```

## Speed testing

```bash
scripts/proxy-speed-test.sh --proxy http://127.0.0.1:7890
scripts/proxy-speed-test.sh --socks5 127.0.0.1:1080 --rounds 5
```

## Implementation detail

Channel proxies use undici `ProxyAgent`. Each channel creates an isolated `fetch` bound to its proxy; failures in one channel do not affect others.
