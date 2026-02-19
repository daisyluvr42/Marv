from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from backend.agent.config import get_settings
from backend.storage import models  # noqa: F401


settings = get_settings()
engine = create_engine(
    f"sqlite:///{settings.edge_db_path}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    settings.edge_db_path.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
