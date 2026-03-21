---
name: experiment
description: "Run a measurable optimization experiment. Iteratively mutate, measure, and keep improvements or roll back regressions. Works for code (test coverage, performance, bundle size), content (copywriting, prompts), and any task with a measurable outcome. Usage: /experiment [objective]"
user-invocable: true
metadata: { "marv": { "emoji": "🧪" } }
---

# experiment — Autoresearch-Style Optimization Loop

You are an experiment orchestrator. Your job is to set up and run a measurable optimization loop: define evaluators, run iterative mutations, keep improvements, and roll back regressions.

## Phase 1 — Understand the Objective

Parse the user's request and determine:

1. **Objective**: What should be optimized? (e.g., "improve test coverage", "make this copy more persuasive", "reduce bundle size")
2. **Domain**: What type of task is this? Use the templates below.
3. **Target files**: What files will be modified and evaluated?
4. **Success criteria**: What metric value means "good enough"? (threshold for early stop)

If the objective is unclear, ask the user. Do not guess.

## Phase 2 — Configure the Evaluator

Choose or build an evaluator based on the domain. An evaluator = a way to get a numeric score.

### Built-in Templates

**Code: Test Coverage**

```json
{
  "id": "coverage",
  "name": "Test Coverage",
  "measureCommand": "pnpm test:coverage 2>&1 | grep 'All files'",
  "metricParser": "(\\d+\\.?\\d*)\\s*\\|\\s*\\d+",
  "direction": "higher_is_better",
  "timeoutSeconds": 120
}
```

**Code: Build Time**

```json
{
  "id": "build-time",
  "name": "Build Time",
  "measureCommand": "{ time pnpm build 2>&1; } 2>&1 | grep real | awk '{print $2}'",
  "metricParser": "first_number",
  "direction": "lower_is_better",
  "timeoutSeconds": 300
}
```

**Code: Bundle Size**

```json
{
  "id": "bundle-size",
  "name": "Bundle Size (KB)",
  "measureCommand": "du -sk dist/ | awk '{print $1}'",
  "metricParser": "first_number",
  "direction": "lower_is_better"
}
```

**Code: Line Count**

```json
{
  "id": "line-count",
  "name": "Line Count",
  "measureCommand": "wc -l src/**/*.ts | tail -1",
  "metricParser": "first_number",
  "direction": "lower_is_better"
}
```

**Content: LLM-as-Judge (Copywriting, Prompts, etc.)**

For non-code content, use the LLM-as-Judge pattern. Set `judgePrompt` and `judgeFile` — the evaluator will read the file and pipe it to `measureCommand` for scoring.

```json
{
  "id": "copy-quality",
  "name": "Copy Quality",
  "measureCommand": "marv message send --stdin",
  "metricParser": "last_number",
  "direction": "higher_is_better",
  "threshold": 8,
  "judgeFile": "draft.txt",
  "judgePrompt": "Score this marketing copy 1-10 on: clarity (is the message clear?), persuasiveness (does it compel action?), and conciseness (no wasted words). Weight clarity highest."
}
```

**Content: Image Quality (with Vision)**

```json
{
  "id": "image-quality",
  "name": "Image Quality",
  "measureCommand": "marv message send --attach output.png --stdin",
  "metricParser": "last_number",
  "direction": "higher_is_better",
  "threshold": 8,
  "judgePrompt": "Score this image 1-10 on composition, visual appeal, and relevance to the brief."
}
```

**Benchmark: API Latency**

```json
{
  "id": "latency",
  "name": "API Latency (ms)",
  "measureCommand": "curl -s -w '%{time_total}' -o /dev/null http://localhost:3000/api/health | awk '{printf \"%.0f\", $1*1000}'",
  "metricParser": "first_number",
  "direction": "lower_is_better"
}
```

**Custom**: If none of the templates fit, construct a `measureCommand` that outputs a number. Any shell command works.

### Multi-Evaluator

You can combine evaluators — all must pass for a change to be kept. Example: "improve coverage BUT don't increase build time":

```json
"evaluators": [
  { "id": "coverage", "direction": "higher_is_better", ... },
  { "id": "build-time", "direction": "lower_is_better", ... }
]
```

## Phase 3 — Dry-Run the Evaluator

Before starting the experiment loop, **always** test the evaluator:

1. Run the `measureCommand` once manually
2. Check that it produces output
3. Check that `metricParser` correctly extracts a number from that output
4. Report the baseline value to the user

If the evaluator fails or the metric can't be parsed, fix the command before proceeding. Do NOT start the loop with a broken evaluator.

## Phase 4 — Configure the Experiment

Build the full `ExperimentSpec`:

```json
{
  "id": "exp_<short-id>",
  "name": "<descriptive name>",
  "evaluators": [<evaluator specs from Phase 2>],
  "objective": "<what the mutation agent should try to improve>",
  "constraints": {
    "allowedFiles": ["<glob patterns for files to modify>"],
    "deniedFiles": ["<glob patterns to never touch>"]
  },
  "maxIterations": 10,
  "checkpoint": { "strategy": "git" }
}
```

**Checkpoint strategy selection:**

- `"git"` — for code changes in a repo (default)
- `{ "strategy": "file-copy", "paths": ["path/to/file"] }` — for config files outside the repo
- `{ "strategy": "none" }` — when rollback isn't needed (just log changes)

**Constraints:**

- Set `allowedFiles` to limit the blast radius
- Set `deniedFiles` to protect critical files
- `maxIterations`: 5-10 for quick experiments, up to 20 for thorough optimization
- `tokenBudget` and `timeBudgetSeconds` are optional safety caps

## Phase 5 — Run the Experiment

Use the experiment tools to execute the loop. The system will:

1. Measure baseline metrics
2. For each iteration:
   - Save a checkpoint
   - Run a mutation agent turn (with the objective + prior iteration history)
   - Measure new metrics
   - If improved: keep changes, update best
   - If regressed: restore checkpoint
   - Log the iteration
3. On completion: write experiment log, persist to memory

Report results to the user after each iteration and at completion.

## Phase 6 — Report Results

Present a summary:

- Baseline vs. best metrics achieved
- Number of iterations (kept vs. rolled back)
- What changes were most effective
- The experiment log location for full details

## Guidelines

- **Dry-run first**: Always validate the evaluator before starting the loop
- **Small changes**: Each iteration should make one focused change
- **Learn from history**: The mutation prompt includes prior iteration results — avoid repeating rolled-back approaches
- **Multi-evaluator**: When optimizing one metric, add a guard evaluator for metrics that shouldn't regress
- **Local models**: Experiments default to local compute for unlimited iterations
