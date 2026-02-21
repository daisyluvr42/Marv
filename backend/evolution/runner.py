"""Main EA runner that integrates evaluator, ledger, and storage."""

from __future__ import annotations

import asyncio
import copy
import json
import random
from typing import Any
from uuid import uuid4

from backend.agent.state import now_ts
from backend.ledger.events import EvolutionEvent
from backend.ledger.store import append_event
from backend.patch.state import _merge_patch, read_seed
from backend.permissions.exec_approvals import load_exec_approvals
from backend.approvals.policy import load_approval_policy
from backend.sandbox.runtime import load_execution_config
from backend.storage.db import init_db
from backend.tools.registry import list_tools, scan_tools

from .evaluators.offline_suite import OfflineSuiteEvaluator
from .operators import WHITELIST_PATHS, crossover_patch, mutate_patch, sanitize_patch_doc, tournament_selection
from .schemas import EvolutionRunConfig, GenerationResult, Genome, Individual
from .storage import (
    create_evolution_individual,
    create_evolution_run,
    get_evolution_run,
    list_top_individuals,
    update_evolution_individual,
    update_evolution_run,
)


class EvolutionRunner:
    """Patch-native EA runner with ledger + DB persistence."""

    def __init__(self, evaluator: OfflineSuiteEvaluator | None = None) -> None:
        self._evaluator = evaluator
        self.last_summary: dict[str, Any] = {}

    def run(self, run_config: EvolutionRunConfig | dict[str, Any]) -> str:
        config = run_config if isinstance(run_config, EvolutionRunConfig) else EvolutionRunConfig.model_validate(run_config)
        init_db()
        scan_tools()
        return asyncio.run(self._run_async(config))

    async def _run_async(self, config: EvolutionRunConfig) -> str:
        evaluator = self._evaluator or OfflineSuiteEvaluator(config.suite_path)
        rng = random.Random(config.seed)

        safe_tools = sorted(set(config.safe_tools or self._detect_safe_tools()))
        base_config = self._build_base_config(safe_tools=safe_tools)

        base_patch = sanitize_patch_doc(copy.deepcopy(config.base_patch_doc), self._whitelist_schema(base_config, config, safe_tools))
        run_row = create_evolution_run(
            config=config.model_dump(mode="json"),
            suite_path=str(evaluator.suite_path),
            seed=config.seed,
            status="running",
        )
        run_id = run_row.run_id

        self._append_ledger_event(
            run_id=run_id,
            stage="run_start",
            details={
                "seed": config.seed,
                "population_size": config.population_size,
                "generations": config.generations,
                "suite_path": str(evaluator.suite_path),
            },
        )

        whitelist_schema = self._whitelist_schema(base_config, config, safe_tools)
        population = self._initialize_population(
            population_size=config.population_size,
            base_patch=base_patch,
            rng=rng,
            whitelist_schema=whitelist_schema,
        )

        if config.dry_run:
            dry_payload = self._persist_dry_run_population(run_id=run_id, population=population)
            summary = {
                "run_id": run_id,
                "mode": "dry_run",
                "population_size": len(population),
                "individuals": dry_payload,
            }
            update_evolution_run(run_id, status="completed", summary=summary)
            self._append_ledger_event(run_id=run_id, stage="run_end", details=summary)
            self.last_summary = summary
            return run_id

        executed_generations: list[GenerationResult] = []
        all_evaluated: list[Individual] = []
        early_stopped = False

        for generation_index in range(config.generations):
            individuals = self._materialize_population(generation_index=generation_index, population=population)
            for individual in individuals:
                create_evolution_individual(
                    run_id=run_id,
                    individual_id=individual.id,
                    generation=generation_index,
                    patch_doc=individual.genome.patch_doc,
                    metadata=individual.genome.metadata,
                    parent_ids=individual.parent_ids,
                    status="pending",
                )

            evaluated = await self._evaluate_generation(
                run_id=run_id,
                generation_index=generation_index,
                individuals=individuals,
                base_config=base_config,
                evaluator=evaluator,
                config=config,
                semaphore=asyncio.Semaphore(config.concurrency),
                whitelist_schema=whitelist_schema,
            )
            evaluated.sort(key=lambda item: float(item.fitness if item.fitness is not None else -999999.0), reverse=True)
            all_evaluated.extend(evaluated)

            generation_summary = GenerationResult(
                generation_index=generation_index,
                individuals=[self._individual_summary(item) for item in evaluated],
                best=self._individual_summary(evaluated[0]) if evaluated else None,
            )
            executed_generations.append(generation_summary)

            if evaluated:
                best = evaluated[0]
                self._append_ledger_event(
                    run_id=run_id,
                    stage="generation_best",
                    details={
                        "generation": generation_index,
                        "individual_id": best.id,
                        "fitness": best.fitness,
                        "patch_doc": best.genome.patch_doc,
                    },
                )
                if config.early_stop_fitness is not None and best.fitness is not None and best.fitness >= config.early_stop_fitness:
                    early_stopped = True
                    break

            if generation_index >= config.generations - 1:
                break

            population = self._next_generation_population(
                evaluated=evaluated,
                population_size=config.population_size,
                elitism=config.elitism,
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate,
                tournament_k=config.tournament_k,
                rng=rng,
                whitelist_schema=whitelist_schema,
            )

        top_rows = list_top_individuals(run_id=run_id, limit=config.top_k)
        top_payload = [self._row_to_output_payload(item, base_config=base_config) for item in top_rows]

        best_id = None
        best_fitness = None
        if top_rows:
            best_id = top_rows[0].individual_id
            best_fitness = top_rows[0].fitness

        summary = {
            "run_id": run_id,
            "status": "completed",
            "early_stopped": early_stopped,
            "generations_executed": len(executed_generations),
            "best_individual_id": best_id,
            "best_fitness": best_fitness,
            "top_individuals": top_payload,
        }
        update_evolution_run(
            run_id,
            status="completed",
            best_individual_id=best_id,
            best_fitness=best_fitness,
            summary=summary,
        )
        self._append_ledger_event(run_id=run_id, stage="run_end", details=summary)
        self.last_summary = summary
        return run_id

    def get_best(self, run_id: str, *, top_k: int = 5) -> dict[str, Any]:
        run_row = get_evolution_run(run_id)
        if run_row is None:
            raise ValueError(f"evolution run not found: {run_id}")
        summary = json.loads(run_row.summary_json) if run_row.summary_json else {}
        top_rows = list_top_individuals(run_id=run_id, limit=top_k)
        base_config = self._build_base_config(safe_tools=self._detect_safe_tools())
        top_payload = [self._row_to_output_payload(item, base_config=base_config) for item in top_rows]
        if not top_payload and str(summary.get("mode", "")) == "dry_run":
            individuals = summary.get("individuals", [])
            if isinstance(individuals, list):
                for item in individuals[: max(1, top_k)]:
                    if not isinstance(item, dict):
                        continue
                    top_payload.append(
                        {
                            "individual_id": item.get("individual_id"),
                            "generation": 0,
                            "status": "dry_run",
                            "fitness": None,
                            "error": None,
                            "patch_doc": item.get("patch_doc", {}),
                            "patch_diff": self._build_patch_diff(
                                base_config=base_config,
                                patch_doc=item.get("patch_doc", {}) if isinstance(item.get("patch_doc"), dict) else {},
                            ),
                            "metrics": {},
                        }
                    )
        return {
            "run_id": run_id,
            "status": run_row.status,
            "best_individual_id": run_row.best_individual_id,
            "best_fitness": run_row.best_fitness,
            "top_individuals": top_payload,
            "summary": summary,
        }

    async def _evaluate_generation(
        self,
        *,
        run_id: str,
        generation_index: int,
        individuals: list[Individual],
        base_config: dict[str, Any],
        evaluator: OfflineSuiteEvaluator,
        config: EvolutionRunConfig,
        semaphore: asyncio.Semaphore,
        whitelist_schema: dict[str, Any],
    ) -> list[Individual]:
        async def _evaluate_one(individual: Individual) -> Individual:
            async with semaphore:
                update_evolution_individual(
                    individual.id,
                    status="running",
                    fitness=None,
                    metrics={"generation": generation_index},
                    error=None,
                )

                patch_doc = sanitize_patch_doc(individual.genome.patch_doc, whitelist_schema)
                applied_config = self._apply_patch(base_config, patch_doc)
                fitness, metrics, error = await evaluator.evaluate_individual(
                    run_id=run_id,
                    individual_id=individual.id,
                    candidate_config=applied_config,
                    actor_id=config.actor_id,
                    cost_norm_cap=config.cost_norm_cap,
                    risk_norm_cap=config.risk_norm_cap,
                )

                patch_diff = self._build_patch_diff(base_config=base_config, patch_doc=patch_doc)
                metrics = {
                    **metrics,
                    "patch_doc": patch_doc,
                    "patch_diff": patch_diff,
                }

                individual.genome.patch_doc = patch_doc
                individual.fitness = fitness
                individual.metrics = metrics
                individual.error = error
                individual.status = "failed" if error else "completed"

                update_evolution_individual(
                    individual.id,
                    status=individual.status,
                    fitness=individual.fitness,
                    metrics=metrics,
                    error=error,
                )
                self._append_ledger_event(
                    run_id=run_id,
                    stage="individual_result",
                    details={
                        "generation": generation_index,
                        "individual_id": individual.id,
                        "status": individual.status,
                        "fitness": individual.fitness,
                        "error": error,
                        "metrics": metrics,
                    },
                    task_id=individual.id,
                )
                return individual

        return list(await asyncio.gather(*[_evaluate_one(item) for item in individuals]))

    def _initialize_population(
        self,
        *,
        population_size: int,
        base_patch: dict[str, Any],
        rng: random.Random,
        whitelist_schema: dict[str, Any],
    ) -> list[Genome]:
        population: list[Genome] = [Genome(patch_doc=copy.deepcopy(base_patch), metadata={"origin": "seed"})]
        while len(population) < population_size:
            patch = mutate_patch(copy.deepcopy(base_patch), rng, whitelist_schema)
            population.append(Genome(patch_doc=patch, metadata={"origin": "mutated_seed"}))
        return population

    def _next_generation_population(
        self,
        *,
        evaluated: list[Individual],
        population_size: int,
        elitism: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_k: int,
        rng: random.Random,
        whitelist_schema: dict[str, Any],
    ) -> list[Genome]:
        if not evaluated:
            return []

        elites = max(0, min(population_size, elitism))
        next_population: list[Genome] = []
        for elite in evaluated[:elites]:
            next_population.append(
                Genome(
                    patch_doc=copy.deepcopy(elite.genome.patch_doc),
                    metadata={"origin": "elite", "source_individual": elite.id},
                )
            )

        while len(next_population) < population_size:
            parent_a, parent_b = tournament_selection(evaluated, max(2, tournament_k), rng)
            parent_a_patch = copy.deepcopy(parent_a.genome.patch_doc)
            parent_b_patch = copy.deepcopy(parent_b.genome.patch_doc)

            if rng.random() < crossover_rate:
                child_patch = crossover_patch(parent_a_patch, parent_b_patch, rng)
            else:
                child_patch = parent_a_patch

            if rng.random() < mutation_rate:
                child_patch = mutate_patch(child_patch, rng, whitelist_schema)

            child_patch = sanitize_patch_doc(child_patch, whitelist_schema)
            next_population.append(
                Genome(
                    patch_doc=child_patch,
                    metadata={"origin": "offspring", "parents": [parent_a.id, parent_b.id]},
                )
            )

        return next_population

    def _materialize_population(self, *, generation_index: int, population: list[Genome]) -> list[Individual]:
        individuals: list[Individual] = []
        for genome in population:
            metadata = genome.metadata if isinstance(genome.metadata, dict) else {}
            parent_ids = metadata.get("parents", []) if isinstance(metadata.get("parents", []), list) else []
            individuals.append(
                Individual(
                    id=f"evi_{uuid4().hex}",
                    generation=generation_index,
                    genome=Genome(patch_doc=copy.deepcopy(genome.patch_doc), metadata=copy.deepcopy(metadata)),
                    status="pending",
                    parent_ids=[str(item) for item in parent_ids],
                )
            )
        return individuals

    def _persist_dry_run_population(self, *, run_id: str, population: list[Genome]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for genome in population:
            individual_id = f"evi_{uuid4().hex}"
            create_evolution_individual(
                run_id=run_id,
                individual_id=individual_id,
                generation=0,
                patch_doc=genome.patch_doc,
                metadata=genome.metadata,
                parent_ids=[],
                status="dry_run",
            )
            payload.append(
                {
                    "individual_id": individual_id,
                    "patch_doc": genome.patch_doc,
                    "metadata": genome.metadata,
                }
            )
        return payload

    def _individual_summary(self, individual: Individual) -> dict[str, Any]:
        return {
            "individual_id": individual.id,
            "generation": individual.generation,
            "status": individual.status,
            "fitness": individual.fitness,
            "error": individual.error,
            "patch_doc": individual.genome.patch_doc,
        }

    def _row_to_output_payload(self, item: Any, *, base_config: dict[str, Any]) -> dict[str, Any]:
        patch_doc = json.loads(item.patch_json)
        metrics = json.loads(item.metrics_json) if item.metrics_json else {}
        patch_diff = metrics.get("patch_diff")
        if not isinstance(patch_diff, list):
            patch_diff = self._build_patch_diff(base_config=base_config, patch_doc=patch_doc)
        return {
            "individual_id": item.individual_id,
            "generation": item.generation,
            "status": item.status,
            "fitness": item.fitness,
            "error": item.error,
            "patch_doc": patch_doc,
            "patch_diff": patch_diff,
            "metrics": metrics,
        }

    def _build_base_config(self, *, safe_tools: list[str]) -> dict[str, Any]:
        base = copy.deepcopy(read_seed())

        memory = base.get("memory")
        if not isinstance(memory, dict):
            memory = {}
            base["memory"] = memory
        memory.setdefault("top_k", 5)
        if "score_threshold" not in memory:
            min_score = memory.get("min_score", 0.27)
            if isinstance(min_score, (int, float)):
                memory["score_threshold"] = float(min_score)
            else:
                memory["score_threshold"] = 0.27

        base.setdefault("model", "mock")
        base.setdefault("temperature", 0.2)
        base.setdefault("top_p", 1.0)

        approvals_cfg = load_approval_policy()
        base.setdefault("approvals", {})
        if isinstance(base["approvals"], dict):
            mode = str(approvals_cfg.get("mode", "risky")).strip().lower()
            base["approvals"]["mode"] = mode if mode in {"risky", "policy", "all"} else "risky"

        exec_mode_cfg = load_execution_config()
        base.setdefault("sandbox", {})
        if isinstance(base["sandbox"], dict):
            mode = str(exec_mode_cfg.get("mode", "auto")).strip().lower()
            base["sandbox"]["mode"] = mode if mode in {"local", "auto", "sandbox"} else "auto"

        exec_approvals = load_exec_approvals()
        default_allowlist = []
        defaults = exec_approvals.get("defaults", {}) if isinstance(exec_approvals, dict) else {}
        if isinstance(defaults, dict):
            allowlist = defaults.get("allowlist")
            if isinstance(allowlist, list):
                default_allowlist = [str(item).strip() for item in allowlist if str(item).strip()]
        safe_allowlist = sorted(set(tool for tool in default_allowlist if tool in safe_tools))
        base.setdefault("tools", {})
        if isinstance(base["tools"], dict):
            base["tools"]["allowlist"] = safe_allowlist

        return base

    def _apply_patch(self, base_config: dict[str, Any], patch_doc: dict[str, Any]) -> dict[str, Any]:
        # Reuse existing patch merge semantics from backend.patch.state.
        target = copy.deepcopy(base_config)
        _merge_patch(target, patch_doc)
        return target

    def _build_patch_diff(self, *, base_config: dict[str, Any], patch_doc: dict[str, Any]) -> list[dict[str, Any]]:
        patched = self._apply_patch(base_config, patch_doc)
        output: list[dict[str, Any]] = []
        for path in WHITELIST_PATHS:
            before = self._resolve_path(base_config, path)
            after = self._resolve_path(patched, path)
            if before == after:
                continue
            output.append(
                {
                    "path": ".".join(path),
                    "before": before,
                    "after": after,
                }
            )
        return output

    def _resolve_path(self, payload: dict[str, Any], path: tuple[str, ...]) -> Any:
        cursor: Any = payload
        for part in path:
            if not isinstance(cursor, dict) or part not in cursor:
                return None
            cursor = cursor[part]
        return cursor

    def _detect_safe_tools(self) -> list[str]:
        safe_tools: list[str] = []
        for tool in list_tools():
            if str(tool.get("risk", "")).strip().lower() == "read_only":
                safe_tools.append(str(tool.get("name", "")).strip())
        return sorted(set(item for item in safe_tools if item))

    def _whitelist_schema(
        self,
        base_config: dict[str, Any],
        config: EvolutionRunConfig,
        safe_tools: list[str],
    ) -> dict[str, Any]:
        return {
            "base": copy.deepcopy(base_config),
            "safe_tools": safe_tools,
            "model_choices": config.model_choices,
        }

    def _append_ledger_event(
        self,
        *,
        run_id: str,
        stage: str,
        details: dict[str, Any],
        task_id: str | None = None,
    ) -> None:
        append_event(
            EvolutionEvent(
                conversation_id=f"evolution:{run_id}",
                task_id=task_id,
                ts=now_ts(),
                actor_id="evolution",
                stage=stage,
                details=details,
            )
        )
