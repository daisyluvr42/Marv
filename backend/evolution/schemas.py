"""Schema models for patch-native evolution runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EvolutionRunConfig(BaseModel):
    """Configuration for one local offline EA run."""

    suite_path: str
    seed: int = 7
    population_size: int = 6
    generations: int = 3
    elitism: int = 1
    tournament_k: int = 3
    mutation_rate: float = 0.65
    crossover_rate: float = 0.7
    concurrency: int = 2
    top_k: int = 5
    early_stop_fitness: float | None = None
    dry_run: bool = False
    actor_id: str = "evolution"
    cost_norm_cap: float = 100.0
    risk_norm_cap: float = 10.0
    base_patch_doc: dict[str, Any] = Field(default_factory=dict)
    model_choices: list[str] = Field(default_factory=lambda: ["mock", "gpt-4.1", "gpt-4o-mini"])
    safe_tools: list[str] | None = None

    @field_validator("population_size")
    @classmethod
    def _validate_population_size(cls, value: int) -> int:
        return max(2, min(256, value))

    @field_validator("generations")
    @classmethod
    def _validate_generations(cls, value: int) -> int:
        return max(1, min(256, value))

    @field_validator("elitism")
    @classmethod
    def _validate_elitism(cls, value: int) -> int:
        return max(0, min(64, value))

    @field_validator("tournament_k")
    @classmethod
    def _validate_tournament_k(cls, value: int) -> int:
        return max(2, min(32, value))

    @field_validator("concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        return max(1, min(64, value))

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        return max(1, min(50, value))

    @field_validator("mutation_rate", "crossover_rate")
    @classmethod
    def _validate_ratio(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value


class Genome(BaseModel):
    """Patch-native genome: only patch document + metadata."""

    patch_doc: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Individual(BaseModel):
    """One individual evaluated in a generation."""

    id: str
    generation: int
    genome: Genome
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    fitness: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    parent_ids: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """Generation-level summary."""

    generation_index: int
    individuals: list[dict[str, Any]] = Field(default_factory=list)
    best: dict[str, Any] | None = None
