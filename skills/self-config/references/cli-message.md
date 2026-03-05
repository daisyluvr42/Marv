# Message CLI Reference

## marv message send

Send a message to a channel/user.

```bash
marv message send --target <target> --message <text> [options]
```

| Flag                | Description                                 |
| ------------------- | ------------------------------------------- |
| `--target <target>` | Recipient (phone, channel ID, user ID)      |
| `--message <text>`  | Message body                                |
| `--channel <name>`  | Channel (telegram, discord, whatsapp, etc.) |
| `--account <id>`    | Account ID (multi-account)                  |
| `--media <path>`    | Attach media file                           |

Examples:

```bash
marv message send --target +15555550123 --message "Hello from Marv"
marv message send --target +15555550123 --message "See this" --media photo.jpg
marv message send --channel discord --target channel:123 --message "Hello"
```

## marv message broadcast

Send a message to multiple targets:

```bash
marv message broadcast --target +1555... --target +1666... --message "Announcement"
```

## marv message read

Read messages from a channel/user:

```bash
marv message read --channel discord --target channel:123 --limit 20
```

## marv message edit

Edit a sent message:

```bash
marv message edit --channel discord --channel-id 123 --message-id 456 --message "fixed typo"
```

## marv message delete

Delete a message:

```bash
marv message delete --channel discord --channel-id 123 --message-id 456
```

## marv message poll

Create a poll:

```bash
marv message poll --channel discord --target channel:123 \
  --poll-question "Lunch?" \
  --poll-option Pizza --poll-option Sushi --poll-option Salad \
  --poll-duration-hours 24
```

## marv message react / unreact

React to a message:

```bash
marv message react --channel discord --target 123 --message-id 456 --emoji "checkmark"
marv message unreact --channel discord --target 123 --message-id 456 --emoji "checkmark"
```

## marv message pin / unpin

```bash
marv message pin --channel discord --channel-id 123 --message-id 456
marv message unpin --channel discord --channel-id 123 --message-id 456
```

## marv message search

Search messages:

```bash
marv message search --channel discord --guild-id 999 --query "release notes" --limit 10
```

## marv message thread

Manage threads:

```bash
marv message thread create --channel discord --channel-id 123 --message-id 456 --thread-name "bug triage"
```

## marv message permissions

Show channel permissions:

```bash
marv message permissions --channel discord
```

## marv message emoji / sticker

Manage emoji and stickers (channel-specific).

## marv message discord-admin

Discord-specific admin operations (roles, moderation, etc.).
