---
name: skill-extractor
description: "Extract reusable skills from repeated cross-workspace workflows found in soul memory. Use during heartbeat, memory compaction, memory review, soul memory maintenance, promotion cycles, or when the agent notices a workflow pattern appearing across multiple tasks or scopes. Triggers on skill extraction candidates, cross-scope workflow patterns, repeated multi-step procedures, distilled skills from task context, and memory promotion events."
---

# Skill Extractor

Extract cross-workspace workflow patterns from soul memory into standalone, reusable skills.

## When this runs

This skill activates during memory review cycles (heartbeat, compaction, manual review) when the system surfaces `skillExtractionCandidates` from the soul memory promotion cycle, or when you notice repeated multi-step patterns across different tasks or scopes.

## Workflow

Follow steps 1-5 in order. Do not skip the evaluation gate (step 2).

### 1) Gather candidates

Collect workflow patterns from these sources:

- **Promotion cycle output**: `skillExtractionCandidates` from `promoteSoulMemories()` (items promoted P2 to P1 that met cross-scope thresholds).
- **Task distillation**: `DistilledKnowledge.skills` from archived task contexts (workflows with 3+ action-verb steps).
- **Manual observation**: Patterns you notice repeating across conversations in different workspaces.

For each candidate, retrieve the full memory item:

```bash
marv memory search "<candidate-content-snippet>" --json
```

Record for each: `id`, `content`, `reinforcementCount`, `distinctScopeHits`, `kind`, `tier`.

### 2) Evaluate: is this worth extracting?

A candidate must pass ALL four gates to proceed.

**Gate A - Cross-scope**: `distinctScopeHits >= 2`
The pattern appeared in at least 2 different tasks or scopes. Single-project patterns are not skills.

**Gate B - Reinforcement**: `reinforcementCount >= 2`
The pattern was retrieved and used at least twice. One-off workflows are not skills.

**Gate C - Actionable steps**: The workflow contains >= 3 distinct actionable steps (steps with action verbs: run, create, update, test, deploy, verify, open, edit, fix, check, install, configure, build, push, merge).

**Gate D - Generalizability**: The workflow is NOT specific to one project. Check:

- Does it reference hardcoded paths, repo names, or project-specific config? If yes and those cannot be parameterized, reject.
- Would another project benefit from this workflow? If no, reject.

**Decision**:

- All four gates pass: proceed to step 3.
- Any gate fails: skip. Log reason to daily memory (`memory/YYYY-MM-DD.md`).

### 3) Abstract the workflow

Transform the concrete workflow into a generalized skill definition.

**Abstraction rules** (apply in order):

1. **Strip project-specific details**: Replace repo names, file paths, package names, URLs, and usernames with parameter placeholders: `<repo-name>`, `<file-path>`, `<package>`, `<url>`, `<username>`.

2. **Identify parameters**: List all values that change between projects. These become the skill's implicit parameters documented as placeholders the agent fills at runtime.

3. **Generalize tool references**: Replace specific CLI tools with categories when interchangeable (e.g., "package manager" instead of "pnpm" unless the skill is specifically about pnpm).

4. **Preserve action verbs**: Keep imperative structure. Each step must start with an action verb.

5. **Merge redundant steps**: If multiple steps do the same thing with minor variation, combine into one step with a note about variations.

6. **Add decision points**: Where the original workflow had implicit choices, make them explicit with "If X, then Y; otherwise Z" structure.

7. **Cap at 10 steps**: If the abstracted workflow exceeds 10 steps, split into phases or extract sub-workflows.

**Output of this step**: A structured outline:

- Skill name (hyphen-case, verb-led, under 64 chars)
- One-line description (what it does + when to use it)
- Ordered steps (3-10, imperative form)
- Parameters (placeholders the agent fills at runtime)
- Prerequisites (tools, permissions, environment)

### 4) Create the skill

Use the existing skill-creator infrastructure. Do NOT manually create the directory.

```bash
python3 skills/skill-creator/scripts/init_skill.py <skill-name> --path ~/.marv/skills
```

Then edit the generated `~/.marv/skills/<skill-name>/SKILL.md`:

**Frontmatter**:

- `name`: hyphen-case skill name from step 3.
- `description`: combine the one-line description with "Use when..." trigger scenarios. Keep under 1024 chars.

**Body** (keep under 200 lines):

- `## Overview`: 1-2 sentences.
- `## Prerequisites`: tools, permissions (if any).
- `## Workflow`: numbered steps from step 3, with parameter placeholders.
- `## Parameters`: table of placeholders and what they represent.

Do NOT include:

- "When to use" sections in the body (belongs in frontmatter description).
- README, changelog, or meta-documentation.
- References to the extraction process itself.

### 5) Validate and report

Run validation:

```bash
python3 skills/skill-creator/scripts/quick_validate.py ~/.marv/skills/<skill-name>
```

Fix any validation errors before proceeding.

**Report** (output to user or log to daily memory):

```
Extracted skill: <skill-name>
Source: <memory-item-ids>
Reason: appeared in <N> scopes, reinforced <M> times, <K> actionable steps
Location: ~/.marv/skills/<skill-name>/SKILL.md
```

## Anti-patterns (do not extract)

- Single-project build/deploy scripts (not cross-workspace).
- User preferences or configuration choices (soul memory facts, not skills).
- Debugging sessions for a specific bug (lessons, not skills).
- Workflows with fewer than 3 steps (too simple to justify a skill).
- Workflows that duplicate an existing skill.

## Duplicate detection

Before creating, check for existing skills with overlapping purpose:

```bash
marv skills list 2>/dev/null
ls ~/.marv/skills/ skills/ 2>/dev/null
```

If a similar skill exists, update it instead of creating a new one. Use the skill-creator skill for advanced editing patterns.
