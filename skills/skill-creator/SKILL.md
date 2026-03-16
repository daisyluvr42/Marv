---
name: skill-creator
description: Create new skills, edit or improve existing skills, and optimize skill descriptions for better triggering accuracy. Use when: designing a new skill from scratch; modifying, refactoring, or iterating on an existing skill; improving a skill's description so it triggers more reliably; evaluating or benchmarking skill performance against real usage examples; or packaging reusable scripts and references into a persistent skill.
---

# Skill Creator

Skills are modular, self-contained packages that extend the agent's capabilities with specialized knowledge, workflows, and reusable tools. Think of them as "onboarding guides" for specific domains—they provide procedural knowledge and bundled assets that make repeated tasks reliable and token-efficient.

## Core Principles

### Concise is Key

The context window is a shared resource. Skills compete with conversation history, system prompt, and other skill metadata for space. Only add context the agent doesn't already have. Challenge every paragraph: "Does this justify its token cost?"

Prefer concise examples over verbose explanations.

### Set Appropriate Degrees of Freedom

Match specificity to task fragility:

- **High freedom (text instructions)** — multiple valid approaches, context-driven decisions
- **Medium freedom (pseudocode/parameterized scripts)** — preferred pattern, some variation acceptable
- **Low freedom (specific scripts, few params)** — fragile operations, consistency critical

### Progressive Disclosure

Skills load in three levels to manage context efficiently:

1. **Metadata** (name + description) — always in context, ~100 words
2. **SKILL.md body** — loaded when skill triggers; keep under 500 lines
3. **Bundled resources** — loaded only when needed (scripts, references, assets)

The description is the **only** field the agent reads to decide whether to invoke the skill. All "when to use" context belongs in the description, not the body.

## Anatomy of a Skill

```
skill-name/
├── SKILL.md                  (required)
│   ├── YAML frontmatter: name, description
│   └── Markdown instructions
└── Bundled Resources         (optional)
    ├── scripts/              - Executable code (Python/Bash/etc.)
    ├── references/           - Documentation loaded into context as needed
    └── assets/               - Output files: templates, icons, fonts, boilerplate
```

**Do not include** README.md, INSTALLATION_GUIDE.md, CHANGELOG.md, or other auxiliary docs. Skills contain only what the agent needs to execute the task.

### scripts/

For code that would otherwise be rewritten from scratch on every invocation:

- Provide deterministic reliability for fragile operations
- Can be executed without loading into context
- Must be tested by actually running them

### references/

For domain knowledge the agent needs while working—schemas, API docs, policies, workflow guides:

- Kept out of SKILL.md to avoid bloat
- Loaded only when relevant
- For files over ~100 lines, add a table of contents at the top

### assets/

For files used in output rather than loaded into context—templates, images, boilerplate code.

## Skill Creation Process

### Step 1: Understand the Use Cases

Before writing anything, gather concrete examples. Ask:

- "What tasks should this skill handle?"
- "What would a user say that should trigger this skill?"
- "Are there variations or edge cases to support?"

### Step 2: Plan Reusable Contents

For each example, identify what reusable resources would help execute it reliably:

- Code rewritten repeatedly → `scripts/rotate_pdf.py`
- Domain knowledge needed each time → `references/schema.md`
- Boilerplate output → `assets/template/`

### Step 3: Initialize the Skill

Use `init_skill.py` (in this skill's `scripts/` directory) to scaffold a new skill. Resolve its path from this SKILL.md's `<location>`:

```bash
# Managed (persists globally — recommended for agent-created tools):
python3 <skill-creator-dir>/scripts/init_skill.py <skill-name> --path ~/.marv/skills

# Workspace-scoped (checked into the project):
python3 <skill-creator-dir>/scripts/init_skill.py <skill-name> --path <workspace>/skills
```

Optional flags: `--resources scripts,references,assets` `--examples`

Skip this step only when iterating on an existing skill.

### Step 4: Implement Resources and Write SKILL.md

Start with scripts, references, and assets identified in Step 2. Test all scripts by running them.

Then write or update SKILL.md:

**Frontmatter** (only `name` and `description`; no other fields):

- `name`: hyphen-case, under 64 chars; verb-led when possible (e.g. `gh-address-comments`)
- `description`: the primary triggering signal — include what the skill does AND when to use it. Put all triggering context here; the body loads only after triggering.

**Body guidelines:**

- Use imperative/infinitive form
- Keep under 500 lines; split detailed content into `references/` files
- Reference bundled resources explicitly: say what each file contains and when to read it
- Cross-reference from SKILL.md → keep references one level deep

**Example of a strong description** (for a `docx` skill):

> "Create, edit, and analyze Word documents (.docx). Use when working with: creating new docs, modifying content, working with tracked changes, adding comments, extracting text, or any other .docx task."

### Step 5: Optimize the Description

The description is the sole triggering mechanism. A weak description causes missed triggers or false positives.

**Test triggering accuracy:**

1. Write 5–10 example queries that should trigger this skill
2. Write 3–5 queries that should NOT trigger it (negative cases)
3. Run `marv skills check` to verify the skill appears in the loaded index
4. If triggering is inconsistent, revise the description for specificity

**Description quality checklist:**

- Does it state what the skill does AND when to use it?
- Does it name specific file types, tools, or task patterns that trigger it?
- Is it free of vague phrases like "various tasks" or "handles many things"?
- Is it distinct from similar skills' descriptions to avoid ambiguity?

### Step 6: Eval and Iterate

After real-world usage, identify gaps and update:

1. **Description** — sharpen if the skill is missed or over-triggered
2. **Body** — add missing steps, fix incorrect procedures, trim unused sections
3. **Scripts** — fix bugs, handle edge cases, improve output format
4. **References** — add domain knowledge that keeps being needed

**Benchmarking variance**: run the same task multiple times and observe whether the skill triggers consistently. If variance is high, the description likely overlaps with another skill or is under-specified.
