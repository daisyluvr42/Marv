# Plugins, Skills & Hooks CLI Reference

## marv plugins

Manage plugins and extensions.

```bash
marv plugins list                    # list installed plugins
marv plugins info <name>             # show plugin details
marv plugins install <source>        # install from npm or file path
marv plugins enable <name>           # enable a plugin
marv plugins disable <name>          # disable a plugin
marv plugins uninstall <name>        # uninstall a plugin
marv plugins update                  # update installed plugins
marv plugins check                   # check plugin installations
```

Install examples:

```bash
marv plugins install @marv/msteams
marv plugins install ./extensions/matrix
marv plugins install marv-plugin-example
```

## marv skills

List and inspect available skills.

```bash
marv skills list                     # list all available skills
marv skills info <name>              # show skill details
marv skills check                    # check which skills are ready vs missing
```

Skills are loaded from (in precedence order):

1. Extra dirs (from `skills.load.extraDirs` config)
2. Bundled skills (shipped with Marv)
3. Managed skills (`~/.marv/skills`)
4. Personal agent skills (`~/.agents/skills`)
5. Project agent skills (`.agents/skills` in workspace)
6. Workspace skills (`workspace/skills`)

## marv hooks

Manage internal agent hooks.

```bash
marv hooks list                      # list available hooks
marv hooks info <name>               # show hook details
marv hooks check                     # check hook installations
marv hooks install <name>            # install a hook
marv hooks enable <name>             # enable a hook
marv hooks disable <name>            # disable a hook
marv hooks uninstall <name>          # uninstall a hook
marv hooks update                    # update installed hooks
```
