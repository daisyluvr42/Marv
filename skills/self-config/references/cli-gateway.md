# Gateway CLI Reference

## marv gateway run

Run the WebSocket Gateway (foreground):

```bash
marv gateway run
marv gateway run --bind loopback --port 4242
marv gateway run --bind lan --port 4242
marv gateway run --verbose
marv gateway run --force             # override existing gateway lock
```

Default address: `ws://127.0.0.1:4242`

### Bind modes

- `loopback` (default): localhost only, most secure
- `lan`: accessible from local network
- `custom`: bind to a specific address

## marv gateway status

Show gateway service status and probe the Gateway:

```bash
marv gateway status
```

## marv gateway start / stop / restart

Manage the gateway background service (systemd/launchd):

```bash
marv gateway start
marv gateway stop
marv gateway restart
```

macOS: prefer the Marv menu bar app or `scripts/restart-mac.sh`.

Manual headless restart:

```bash
pkill -9 -f marv-gateway || true
nohup marv gateway run --bind loopback --port 4242 --force > /tmp/marv-gateway.log 2>&1 &
```

Verify:

```bash
marv channels status --probe
```

## marv gateway call

Call a Gateway method directly:

```bash
marv gateway call <method> [args...]
```

## marv gateway usage-cost

Fetch usage cost summary from session logs:

```bash
marv gateway usage-cost
```

## marv gateway discover

Find local and wide-area gateway beacons:

```bash
marv gateway discover
```

## Authentication

```bash
marv config set gateway.auth.mode "token"
marv config set gateway.auth.token "your-secret"
```

Modes: `none` (loopback only), `token`, `password`, `trusted-proxy`.

## marv system

System events, heartbeat, and presence:

```bash
marv system event                    # enqueue a system event
marv system heartbeat last           # show last heartbeat
marv system heartbeat enable         # enable heartbeats
marv system heartbeat disable        # disable heartbeats
marv system presence                 # list presence entries
```

## marv tui

Open a terminal UI connected to the Gateway:

```bash
marv tui
marv tui --session <id>
marv tui --message "initial message"
```

## marv acp

Agent Control Protocol bridge:

```bash
marv acp                             # run ACP bridge
marv acp client                      # interactive ACP client
```

## marv dns

DNS helpers for wide-area discovery (Tailscale + CoreDNS):

```bash
marv dns bootstrap                   # bootstrap CoreDNS config
marv dns status                      # show DNS status
marv dns resolve <hostname>          # resolve via CoreDNS
```

## marv browser

Manage the dedicated browser (Chrome/Chromium):

```bash
marv browser status                  # check browser status
marv browser launch                  # launch browser
marv browser close                   # close browser
marv browser navigate <url>          # navigate to URL
marv browser screenshot              # take screenshot
marv browser inspect                 # inspect page elements
marv browser find <query>            # find elements
marv browser click <selector>        # click an element
marv browser fill <selector> <value> # fill a form field
marv browser eval <js>               # execute JavaScript
```
