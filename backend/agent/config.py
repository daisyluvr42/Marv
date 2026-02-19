from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    data_dir: Path
    edge_db_path: Path
    core_base_url: str
    request_timeout_seconds: float
    max_retries: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("EDGE_DATA_DIR", "./data")).resolve()
    edge_db_path = Path(os.getenv("EDGE_DB_PATH", str(data_dir / "edge.db"))).resolve()
    core_base_url = os.getenv("CORE_BASE_URL", "http://localhost:9000").rstrip("/")
    request_timeout_seconds = float(os.getenv("CORE_REQUEST_TIMEOUT_SECONDS", "5"))
    max_retries = int(os.getenv("CORE_MAX_RETRIES", "1"))

    return Settings(
        data_dir=data_dir,
        edge_db_path=edge_db_path,
        core_base_url=core_base_url,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
    )
