"""Storage helpers for evolution runs and individuals."""

from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import select

from backend.agent.state import now_ts
from backend.storage.db import get_session
from backend.storage.models import EvolutionIndividual, EvolutionRun


def create_evolution_run(*, config: dict[str, object], suite_path: str, seed: int, status: str = "running") -> EvolutionRun:
    ts = now_ts()
    item = EvolutionRun(
        run_id=f"evr_{uuid4().hex}",
        created_at=ts,
        updated_at=ts,
        status=status,
        suite_path=suite_path,
        seed=seed,
        config_json=json.dumps(config, ensure_ascii=True),
        summary_json="{}",
    )
    with get_session() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def update_evolution_run(
    run_id: str,
    *,
    status: str | None = None,
    best_individual_id: str | None = None,
    best_fitness: float | None = None,
    summary: dict[str, object] | None = None,
) -> EvolutionRun | None:
    with get_session() as session:
        item = session.exec(select(EvolutionRun).where(EvolutionRun.run_id == run_id)).first()
        if item is None:
            return None
        if status is not None:
            item.status = status
        if best_individual_id is not None:
            item.best_individual_id = best_individual_id
        if best_fitness is not None:
            item.best_fitness = best_fitness
        if summary is not None:
            item.summary_json = json.dumps(summary, ensure_ascii=True)
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def get_evolution_run(run_id: str) -> EvolutionRun | None:
    with get_session() as session:
        return session.exec(select(EvolutionRun).where(EvolutionRun.run_id == run_id)).first()


def create_evolution_individual(
    *,
    run_id: str,
    individual_id: str,
    generation: int,
    patch_doc: dict[str, object],
    metadata: dict[str, object],
    parent_ids: list[str],
    status: str = "pending",
) -> EvolutionIndividual:
    item = EvolutionIndividual(
        individual_id=individual_id,
        run_id=run_id,
        generation=generation,
        status=status,
        patch_json=json.dumps(patch_doc, ensure_ascii=True),
        metadata_json=json.dumps(metadata, ensure_ascii=True),
        parent_ids_json=json.dumps(parent_ids, ensure_ascii=True),
        metrics_json="{}",
        created_at=now_ts(),
        updated_at=now_ts(),
    )
    with get_session() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def update_evolution_individual(
    individual_id: str,
    *,
    status: str,
    fitness: float | None,
    metrics: dict[str, object],
    error: str | None = None,
) -> EvolutionIndividual | None:
    with get_session() as session:
        item = session.exec(select(EvolutionIndividual).where(EvolutionIndividual.individual_id == individual_id)).first()
        if item is None:
            return None
        item.status = status
        item.fitness = fitness
        item.metrics_json = json.dumps(metrics, ensure_ascii=True)
        item.error = error
        item.updated_at = now_ts()
        session.add(item)
        session.commit()
        session.refresh(item)
        return item


def list_evolution_individuals(
    *,
    run_id: str,
    generation: int | None = None,
    limit: int = 500,
) -> list[EvolutionIndividual]:
    with get_session() as session:
        stmt = select(EvolutionIndividual).where(EvolutionIndividual.run_id == run_id)
        if generation is not None:
            stmt = stmt.where(EvolutionIndividual.generation == generation)
        stmt = stmt.order_by(EvolutionIndividual.generation.asc(), EvolutionIndividual.created_at.asc())
        stmt = stmt.limit(max(1, min(limit, 5000)))
        return list(session.exec(stmt))


def list_top_individuals(*, run_id: str, limit: int = 5) -> list[EvolutionIndividual]:
    with get_session() as session:
        stmt = (
            select(EvolutionIndividual)
            .where(EvolutionIndividual.run_id == run_id)
            .where(EvolutionIndividual.fitness.is_not(None))
            .order_by(EvolutionIndividual.fitness.desc(), EvolutionIndividual.updated_at.asc())
            .limit(max(1, min(limit, 100)))
        )
        return list(session.exec(stmt))
