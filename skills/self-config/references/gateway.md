# Gateway Configuration Reference

## Starting the gateway

```bash
marv gateway run --bind loopback --port 4242          # foreground (dev)
marv onboard --install-daemon                          # background daemon
marv gateway status                                    # check status
```

Default address: `ws://127.0.0.1:4242`

## Bind modes

- `loopback` (default): localhost only, most secure
- `lan`: accessible from local network
- `custom`: bind to a specific address

```bash
marv config set gateway.bind "loopback"
marv config set gateway.port 4242
```

## Authentication modes

- `none`: no auth (loopback only)
- `token`: shared secret token
- `password`: password-based
- `trusted-proxy`: delegate auth to a reverse proxy (Pomerium, Caddy, nginx)

```bash
marv config set gateway.auth.mode "token"
marv config set gateway.auth.token "your-secret"
```

## Health and diagnostics

```bash
marv health                 # health check
marv health --json          # structured output
marv status --all           # full status (read-only, pasteable)
marv status --deep          # status with probes
marv doctor                 # diagnose common issues
marv logs --follow          # tail gateway logs
```

## Restart

macOS app: use the Marv menu bar app or `scripts/restart-mac.sh`

Manual (headless):

```bash
pkill -9 -f marv-gateway || true
nohup marv gateway run --bind loopback --port 4242 --force > /tmp/marv-gateway.log 2>&1 &
```

Verify:

```bash
marv channels status --probe
ss -ltnp | rg 4242       # or: lsof -nP -iTCP:4242
tail -n 120 /tmp/marv-gateway.log
```

## Tailscale integration

```bash
marv config set gateway.tailscale.enabled true
```

See `/gateway/tailscale` docs for Tailscale Serve and Funnel setup.

## Configuration file

Location: `~/.marv/marv.json` (JSON5, supports comments and trailing commas)

```bash
marv config set <key> <value>
marv config get <key>
marv configure                  # interactive wizard
```

Full reference: `/gateway/configuration-reference` in docs.
