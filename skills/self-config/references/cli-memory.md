# Memory CLI Reference

## marv memory search

Search through memory files:

```bash
marv memory search --query "deployment notes"
marv memory search --query "proxy config" --agent ops
```

## marv memory reindex

Reindex memory for full-text search:

```bash
marv memory reindex
```

## marv memory list

List memory files for an agent:

```bash
marv memory list
marv memory list --agent ops
```

## marv memory status

Show memory indexing status:

```bash
marv memory status
```

## Memory file locations

- Agent memory: `~/.marv/agents/<agentId>/memory/`
- Daily logs: `memory/YYYY-MM-DD.md`
- Long-term: `MEMORY.md` (optional, in workspace)
- Session logs: `~/.marv/agents/<agentId>/sessions/*.jsonl`
