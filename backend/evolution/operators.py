"""Evolution operators constrained by patch whitelist rules."""

from __future__ import annotations

import copy
import random
from typing import Any

SAFE_APPROVAL_MODES = {"risky", "policy", "all"}
APPROVAL_MODE_RANK = {"risky": 1, "policy": 2, "all": 3}
SANDBOX_MODE_RANK = {"local": 0, "auto": 1, "sandbox": 2}

WHITELIST_PATHS: tuple[tuple[str, ...], ...] = (
    ("model",),
    ("temperature",),
    ("top_p",),
    ("memory", "top_k"),
    ("memory", "score_threshold"),
    ("tools", "allowlist"),
    ("approvals", "mode"),
    ("sandbox", "mode"),
)


def mutate_patch(
    patch_doc: dict[str, Any],
    rng: random.Random,
    whitelist_schema: dict[str, Any],
) -> dict[str, Any]:
    """Mutate one patch document under strict field whitelist and safety bounds."""

    mutated = copy.deepcopy(patch_doc)
    base = _as_dict(whitelist_schema.get("base"))
    safe_tools = set(_string_list(whitelist_schema.get("safe_tools")))
    model_choices = _string_list(whitelist_schema.get("model_choices")) or ["mock"]

    selected_path = rng.choice(list(WHITELIST_PATHS))
    current_value = _resolve_value(mutated, selected_path)
    if current_value is None:
        current_value = _resolve_value(base, selected_path)

    if selected_path == ("model",):
        _set_value(mutated, selected_path, rng.choice(model_choices))
    elif selected_path == ("temperature",):
        previous = float(current_value) if isinstance(current_value, (int, float)) else 0.2
        step = rng.uniform(-0.25, 0.25)
        _set_value(mutated, selected_path, _clamp(round(previous + step, 2), 0.0, 1.5))
    elif selected_path == ("top_p",):
        previous = float(current_value) if isinstance(current_value, (int, float)) else 1.0
        step = rng.uniform(-0.2, 0.2)
        _set_value(mutated, selected_path, _clamp(round(previous + step, 2), 0.05, 1.0))
    elif selected_path == ("memory", "top_k"):
        previous = int(current_value) if isinstance(current_value, int) else 5
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        _set_value(mutated, selected_path, int(_clamp(previous + delta, 1, 50)))
    elif selected_path == ("memory", "score_threshold"):
        previous = float(current_value) if isinstance(current_value, (int, float)) else 0.27
        step = rng.uniform(-0.12, 0.12)
        _set_value(mutated, selected_path, _clamp(round(previous + step, 3), 0.0, 1.0))
    elif selected_path == ("tools", "allowlist"):
        current_allowlist = set(_string_list(current_value))
        if not current_allowlist:
            current_allowlist = set(_string_list(_resolve_value(base, selected_path)))
        current_allowlist &= safe_tools
        if safe_tools:
            if (not current_allowlist) or rng.random() < 0.6:
                remaining = sorted(safe_tools - current_allowlist)
                if remaining:
                    current_allowlist.add(rng.choice(remaining))
            elif current_allowlist:
                current_allowlist.remove(rng.choice(sorted(current_allowlist)))
        _set_value(mutated, selected_path, sorted(current_allowlist))
    elif selected_path == ("approvals", "mode"):
        base_mode = str(_resolve_value(base, selected_path) or "risky").strip().lower()
        min_rank = APPROVAL_MODE_RANK.get(base_mode, APPROVAL_MODE_RANK["risky"])
        candidates = [mode for mode, rank in APPROVAL_MODE_RANK.items() if rank >= min_rank]
        _set_value(mutated, selected_path, rng.choice(candidates or ["risky"]))
    elif selected_path == ("sandbox", "mode"):
        base_mode = str(_resolve_value(base, selected_path) or "auto").strip().lower()
        min_rank = SANDBOX_MODE_RANK.get(base_mode, SANDBOX_MODE_RANK["auto"])
        candidates = [mode for mode, rank in SANDBOX_MODE_RANK.items() if rank >= min_rank]
        _set_value(mutated, selected_path, rng.choice(candidates or ["sandbox"]))

    return sanitize_patch_doc(mutated, whitelist_schema)


def crossover_patch(
    patch_a: dict[str, Any],
    patch_b: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Field-level crossover over whitelisted paths."""

    child: dict[str, Any] = {}
    for path in WHITELIST_PATHS:
        value_a = _resolve_value(patch_a, path)
        value_b = _resolve_value(patch_b, path)
        if value_a is None and value_b is None:
            continue
        if value_a is None:
            selected = copy.deepcopy(value_b)
        elif value_b is None:
            selected = copy.deepcopy(value_a)
        else:
            selected = copy.deepcopy(value_a if rng.random() < 0.5 else value_b)
        _set_value(child, path, selected)
    return child


def tournament_selection(pop: list[Any], k: int, rng: random.Random) -> tuple[Any, Any]:
    """Pick two parents by tournament selection on fitness."""

    if not pop:
        raise ValueError("population must not be empty")
    size = max(2, min(len(pop), k))

    def _pick_one() -> Any:
        candidates = rng.sample(pop, size)
        return max(candidates, key=_fitness_value)

    return _pick_one(), _pick_one()


def sanitize_patch_doc(patch_doc: dict[str, Any], whitelist_schema: dict[str, Any]) -> dict[str, Any]:
    """Drop non-whitelisted fields and enforce safe bounds for sensitive fields."""

    sanitized: dict[str, Any] = {}
    base = _as_dict(whitelist_schema.get("base"))
    safe_tools = set(_string_list(whitelist_schema.get("safe_tools")))
    model_choices = _string_list(whitelist_schema.get("model_choices")) or ["mock"]

    for path in WHITELIST_PATHS:
        value = _resolve_value(patch_doc, path)
        if value is None:
            continue

        if path == ("model",):
            if str(value) not in model_choices:
                continue
            _set_value(sanitized, path, str(value))
            continue

        if path == ("temperature",):
            if not isinstance(value, (int, float)):
                continue
            _set_value(sanitized, path, _clamp(float(value), 0.0, 1.5))
            continue

        if path == ("top_p",):
            if not isinstance(value, (int, float)):
                continue
            _set_value(sanitized, path, _clamp(float(value), 0.05, 1.0))
            continue

        if path == ("memory", "top_k"):
            if not isinstance(value, int):
                continue
            _set_value(sanitized, path, int(_clamp(value, 1, 50)))
            continue

        if path == ("memory", "score_threshold"):
            if not isinstance(value, (int, float)):
                continue
            _set_value(sanitized, path, _clamp(float(value), 0.0, 1.0))
            continue

        if path == ("tools", "allowlist"):
            tools = [item for item in _string_list(value) if item in safe_tools]
            _set_value(sanitized, path, sorted(set(tools)))
            continue

        if path == ("approvals", "mode"):
            mode = str(value).strip().lower()
            base_mode = str(_resolve_value(base, path) or "risky").strip().lower()
            min_rank = APPROVAL_MODE_RANK.get(base_mode, APPROVAL_MODE_RANK["risky"])
            if mode not in SAFE_APPROVAL_MODES:
                continue
            if APPROVAL_MODE_RANK.get(mode, 0) < min_rank:
                continue
            _set_value(sanitized, path, mode)
            continue

        if path == ("sandbox", "mode"):
            mode = str(value).strip().lower()
            base_mode = str(_resolve_value(base, path) or "auto").strip().lower()
            min_rank = SANDBOX_MODE_RANK.get(base_mode, SANDBOX_MODE_RANK["auto"])
            if mode not in SANDBOX_MODE_RANK:
                continue
            if SANDBOX_MODE_RANK[mode] < min_rank:
                continue
            _set_value(sanitized, path, mode)
            continue

    return sanitized


def _fitness_value(item: Any) -> float:
    value = getattr(item, "fitness", None)
    if value is None and isinstance(item, dict):
        value = item.get("fitness")
    if not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _resolve_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = payload
    for part in path[:-1]:
        next_item = cursor.get(part)
        if not isinstance(next_item, dict):
            next_item = {}
            cursor[part] = next_item
        cursor = next_item
    cursor[path[-1]] = value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
