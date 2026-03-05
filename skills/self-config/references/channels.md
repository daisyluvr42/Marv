# Channels Configuration Reference

## Supported channels

Core (built-in): Telegram, Discord, Slack, Signal, iMessage, WhatsApp (web), WebChat
Extensions: MS Teams, Matrix, Zalo, Nextcloud Talk, Twitch, Mattermost, LINE, and more under `extensions/`

## Adding a channel

```bash
marv channels login --channel <name>    # interactive login
marv channels list                       # list configured channels
marv channels status --probe             # connectivity check
```

## Common config keys

### Telegram

```bash
marv config set channels.telegram.token "BOT_TOKEN"
marv config set channels.telegram.allowFrom '["*"]'        # open to all
marv config set channels.telegram.dmPolicy "open"
marv config set channels.telegram.proxy "http://..."        # optional proxy
```

### Discord

```bash
marv config set channels.discord.token "BOT_TOKEN"
marv config set channels.discord.proxy "socks5://..."       # optional proxy
marv config set channels.discord.groupPolicy "allowlist"    # default
```

### WhatsApp

```bash
marv channels login --channel whatsapp     # scan QR code
marv config set channels.whatsapp.allowFrom '["+15555550123"]'
```

## Multi-account

Channels support multiple accounts via `channels.<channel>.accounts.<id>`:

```json5
{
  channels: {
    telegram: {
      accounts: {
        personal: { token: "...", allowFrom: ["*"] },
        work: { token: "...", allowFrom: ["+1555..."] },
      },
    },
  },
}
```

## DM policy and allowlists

- `dmPolicy: "open"` requires `allowFrom: ["*"]`
- `dmPolicy: "closed"` (default) requires explicit allowFrom entries
- Allowlist entries: phone numbers, usernames, or user IDs depending on channel

## Diagnostics

```bash
marv channels status --probe   # live connectivity check
marv doctor                    # general health check
marv logs --follow             # live gateway logs
```
