"""CLI helpers for local evolution run/best commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import EvolutionRunner
from .schemas import EvolutionRunConfig


def load_run_config(path: str | Path) -> EvolutionRunConfig:
    file_path = Path(path).expanduser().resolve()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evolution config must be a JSON object")
    return EvolutionRunConfig.model_validate(payload)


def run_evolution(*, config_path: str | Path, out_path: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    config = load_run_config(config_path)
    if dry_run:
        config = config.model_copy(update={"dry_run": True})

    runner = EvolutionRunner()
    run_id = runner.run(config)
    result = runner.get_best(run_id, top_k=config.top_k)

    payload = {
        "run_id": run_id,
        "status": result.get("status"),
        "best_individual_id": result.get("best_individual_id"),
        "best_fitness": result.get("best_fitness"),
        "top_individuals": result.get("top_individuals", []),
    }

    if out_path is not None:
        output_file = Path(out_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    return payload


def best_evolution(*, run_id: str, top_k: int = 5) -> dict[str, Any]:
    runner = EvolutionRunner()
    return runner.get_best(run_id, top_k=top_k)
