from __future__ import annotations

import json
import random

from backend.evolution.operators import WHITELIST_PATHS, mutate_patch, sanitize_patch_doc
from backend.evolution.runner import EvolutionRunner
from backend.evolution.schemas import EvolutionRunConfig
from backend.evolution.storage import get_evolution_run, list_evolution_individuals
from backend.ledger.store import query_events


def _leaf_paths(payload: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(payload, dict):
        output: set[tuple[str, ...]] = set()
        for key, value in payload.items():
            output.update(_leaf_paths(value, prefix + (str(key),)))
        return output
    return {prefix}


def test_evolution_mutation_whitelist() -> None:
    whitelist_schema = {
        "base": {
            "model": "mock",
            "temperature": 0.2,
            "top_p": 1.0,
            "memory": {"top_k": 5, "score_threshold": 0.27},
            "tools": {"allowlist": ["mock_web_search"]},
            "approvals": {"mode": "risky"},
            "sandbox": {"mode": "auto"},
            "forbidden": {"value": "x"},
        },
        "safe_tools": ["mock_web_search"],
        "model_choices": ["mock", "gpt-4.1"],
    }
    allowed = set(WHITELIST_PATHS)

    patch_doc: dict[str, object] = {
        "forbidden": {"value": "should-be-removed"},
        "memory": {"top_k": 7},
        "tools": {"allowlist": ["mock_external_write"]},
    }
    rng = random.Random(123)
    for _ in range(20):
        patch_doc = mutate_patch(patch_doc, rng, whitelist_schema)
        patch_doc = sanitize_patch_doc(patch_doc, whitelist_schema)

        paths = _leaf_paths(patch_doc)
        assert paths.issubset(allowed)
        assert "forbidden" not in patch_doc

        allowlist = patch_doc.get("tools", {}).get("allowlist", []) if isinstance(patch_doc.get("tools"), dict) else []
        assert all(tool == "mock_web_search" for tool in allowlist)

        approvals = patch_doc.get("approvals", {}) if isinstance(patch_doc.get("approvals"), dict) else {}
        if "mode" in approvals:
            assert approvals["mode"] in {"risky", "policy", "all"}

        sandbox = patch_doc.get("sandbox", {}) if isinstance(patch_doc.get("sandbox"), dict) else {}
        if "mode" in sandbox:
            assert sandbox["mode"] in {"auto", "sandbox"}


def test_evolution_runner_smoke() -> None:
    config = EvolutionRunConfig(
        suite_path="backend/evolution/evaluators/data/sample_suite.json",
        seed=11,
        population_size=4,
        generations=1,
        top_k=3,
        concurrency=2,
    )
    runner = EvolutionRunner()
    run_id = runner.run(config)

    result = runner.get_best(run_id, top_k=3)
    assert run_id.startswith("evr_")
    assert result["status"] == "completed"
    assert result["top_individuals"]
    assert len(result["top_individuals"]) <= 3

    run_row = get_evolution_run(run_id)
    assert run_row is not None

    individuals = list_evolution_individuals(run_id=run_id, limit=20)
    assert len(individuals) >= 4
    assert any(item.fitness is not None for item in individuals)

    events = query_events(conversation_id=f"evolution:{run_id}")
    evo_events = [item for item in events if item.type == "EvolutionEvent"]
    assert evo_events
    stages = [json.loads(item.payload_json).get("stage") for item in evo_events]
    assert "run_start" in stages
    assert "run_end" in stages
