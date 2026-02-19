from __future__ import annotations

from typing import Optional

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: str = Field(primary_key=True)
    channel: str
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    created_at: int
    updated_at: int


class Task(SQLModel, table=True):
    __tablename__ = "tasks"

    id: str = Field(primary_key=True)
    conversation_id: str = Field(index=True)
    status: str = Field(index=True)
    created_at: int
    updated_at: int
    last_error: Optional[str] = None
    current_stage: Optional[str] = None


class LedgerEvent(SQLModel, table=True):
    __tablename__ = "ledger_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(unique=True, index=True)
    task_id: Optional[str] = Field(default=None, index=True)
    conversation_id: str = Field(index=True)
    type: str = Field(index=True)
    ts: int = Field(index=True)
    actor_id: Optional[str] = None
    payload_json: str
    hash: Optional[str] = None
    prev_hash: Optional[str] = None


class ToolRegistry(SQLModel, table=True):
    __tablename__ = "tools_registry"

    name: str = Field(primary_key=True)
    version: str = "0.1.0"
    risk: str
    requires_approval: bool
    schema_payload: str = Field(sa_column=Column("schema_json", Text, nullable=False))
    enabled: bool = True
    updated_at: int


class ToolCall(SQLModel, table=True):
    __tablename__ = "tool_calls"

    tool_call_id: str = Field(primary_key=True)
    task_id: Optional[str] = Field(default=None, index=True)
    tool: str
    args_json: str
    status: str = Field(index=True)
    approval_id: Optional[str] = None
    created_at: int
    updated_at: int
    result_json: Optional[str] = None
    error: Optional[str] = None


class Approval(SQLModel, table=True):
    __tablename__ = "approvals"

    approval_id: str = Field(primary_key=True)
    type: str
    status: str = Field(index=True)
    summary: str
    constraints_json: str
    created_at: int = Field(index=True)
    updated_at: int
    actor_id: Optional[str] = None
    decided_by: Optional[str] = None


class ConfigSeed(SQLModel, table=True):
    __tablename__ = "config_seed"

    id: str = Field(primary_key=True)
    seed_json: str
    created_at: int


class ConfigRevision(SQLModel, table=True):
    __tablename__ = "config_revisions"

    revision: str = Field(primary_key=True)
    scope_type: str = Field(index=True)
    scope_id: str = Field(index=True)
    created_at: int = Field(index=True)
    actor_id: str
    patch_json: str
    explanation: str
    risk_level: str
    status: str = Field(index=True)


class PatchProposal(SQLModel, table=True):
    __tablename__ = "patch_proposals"

    proposal_id: str = Field(primary_key=True)
    scope_type: str = Field(index=True)
    scope_id: str = Field(index=True)
    natural_language: str
    patch_json: str
    risk_level: str
    explanation: str
    needs_approval: bool
    created_at: int = Field(index=True)
    actor_id: str
    status: str = Field(index=True)


class MemoryItem(SQLModel, table=True):
    __tablename__ = "memory_items"

    id: str = Field(primary_key=True)
    scope_type: str = Field(index=True)
    scope_id: str = Field(index=True)
    kind: str
    content: str
    embedding_json: str
    confidence: float
    created_at: int = Field(index=True)
    source_event_id: Optional[str] = None


class MemoryCandidate(SQLModel, table=True):
    __tablename__ = "memory_candidates"

    id: str = Field(primary_key=True)
    scope_type: str = Field(index=True)
    scope_id: str = Field(index=True)
    kind: str
    content: str
    embedding_json: str
    confidence: float
    status: str = Field(index=True)
    created_at: int = Field(index=True)
    decided_at: Optional[int] = None
    decided_by: Optional[str] = None
