# Operations & Diagnostics CLI Reference

## marv status

Show channel health and recent session recipients:

```bash
marv status                          # basic status
marv status --all                    # full status (read-only, pasteable)
marv status --deep                   # status with live probes
marv status --usage                  # include usage stats
marv status --json                   # JSON output
```

## marv health

Fetch health from the running gateway:

```bash
marv health
marv health --json
```

## marv doctor

Health checks + quick fixes for the gateway and channels:

```bash
marv doctor
```

Diagnoses common issues: config errors, channel connectivity, permissions, legacy config.

## marv logs

Tail gateway file logs via RPC:

```bash
marv logs                            # recent logs
marv logs --follow                   # live tail
marv logs --limit 200                # last N lines
marv logs --json                     # JSON output
marv logs --plain                    # no ANSI colors
marv logs --local-time               # show local timestamps
```

## marv security

Security tools and config audits:

```bash
marv security audit                  # audit config + local state
marv security audit --deep           # deep audit with probes
marv security audit --fix            # auto-fix safe defaults
marv security audit --json           # JSON output
```

## marv sandbox

Manage sandbox containers for agent isolation:

```bash
marv sandbox list                    # list containers and status
marv sandbox recreate                # recreate containers
marv sandbox explain                 # show effective sandbox config
```

## marv approvals

Manage exec approvals (command allowlists):

```bash
marv approvals list                  # list approvals
marv approvals add                   # add an approval
marv approvals remove                # remove an approval
marv approvals check                 # check if a command is approved
```

## marv cron

Manage scheduled jobs via the Gateway scheduler:

```bash
marv cron status                     # show cron status
marv cron list                       # list configured jobs
marv cron add                        # add a new job
marv cron edit                       # edit a job
marv cron start                      # start a job
marv cron stop                       # stop a job
marv cron remove                     # remove a job
```

Example:

```bash
marv cron add --name "daily-audit" --schedule "0 9 * * *" --command "security audit --deep"
```

## marv dashboard

Open the Control UI in browser:

```bash
marv dashboard
```

## marv docs

Search live Marv docs:

```bash
marv docs "proxy configuration"
marv docs "telegram setup"
```
