# Channels CLI Reference

## marv channels list

List configured channels and auth profiles:

```bash
marv channels list
```

## marv channels status

Show gateway channel status with optional live probes:

```bash
marv channels status
marv channels status --probe         # live API connectivity check
```

## marv channels capabilities

Show provider capabilities (intents/scopes + features):

```bash
marv channels capabilities --channel discord
```

## marv channels resolve

Resolve channel/user names to IDs:

```bash
marv channels resolve --channel discord --name "general"
```

## marv channels logs

Show recent channel logs from the gateway log file:

```bash
marv channels logs
marv channels logs --channel telegram
```

## marv channels add

Add or update a channel account:

```bash
marv channels add --channel telegram --token "BOT_TOKEN"
```

## marv channels remove

Disable or delete a channel account:

```bash
marv channels remove --channel telegram --account work
```

## marv channels login

Link a channel account (interactive login flow):

```bash
marv channels login --channel whatsapp     # scan QR
marv channels login --channel telegram     # enter token
```

## marv channels logout

Log out of a channel session:

```bash
marv channels logout --channel whatsapp
```

## marv pairing

Secure DM pairing (approve inbound requests):

```bash
marv pairing list                    # list pending requests
marv pairing approve                 # approve a request
```

## marv directory

Lookup contact and group IDs:

```bash
marv directory self                  # show connected account identity
marv directory peers list            # search contacts by name
marv directory groups list           # list available groups/channels
marv directory groups members        # list members of a group
```
