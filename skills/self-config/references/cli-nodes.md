# Nodes & Devices CLI Reference

## marv nodes

Manage gateway-connected nodes (remote devices).

### Status

```bash
marv nodes status                    # list nodes with live status
```

### Pairing

```bash
marv nodes pairing pending           # show pending pairing requests
marv nodes pairing approve           # approve a pairing request
```

### Commands

```bash
marv nodes run <node> -- <command>   # run a shell command on a node
marv nodes notify <node> <message>   # send a notification
marv nodes push <node> <message>     # send a push notification
```

### Media

```bash
marv nodes camera snap <node>        # capture a photo
marv nodes camera video <node>       # record video
marv nodes screen shot <node>        # take a screenshot
```

### Other

```bash
marv nodes location get <node>       # get node location
marv nodes canvas <node>             # manage canvas on nodes
```

## marv node

Run and manage the headless node host service (on the node device itself):

```bash
marv node run                        # run node host (foreground)
marv node status                     # show node host status
marv node install                    # install as system service
marv node stop                       # stop the service
marv node restart                    # restart the service
marv node uninstall                  # uninstall the service
```

## marv devices

Device pairing and token management:

```bash
marv devices list                    # list paired devices + pending requests
marv devices approve                 # approve a pending pairing request
marv devices revoke                  # revoke a device token
marv devices rename                  # rename a paired device
```

## marv qr

Generate iOS pairing QR code and setup code:

```bash
marv qr                             # generate QR code (ASCII)
marv qr --remote                    # for remote access
marv qr --json                      # JSON output
marv qr --setup-code-only           # only the setup code
```
