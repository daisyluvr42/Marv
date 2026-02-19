from __future__ import annotations

import os

from pydantic import BaseModel, Field
from fastapi import FastAPI

from core.backends import get_backend


app = FastAPI(title="Blackbox Core Compute", version="0.1.0")
backend = get_backend(os.getenv("CORE_BACKEND", "mock"))


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "mock"
    messages: list[Message] = Field(default_factory=list)
    stream: bool = False


class EmbeddingsRequest(BaseModel):
    model: str = "mock-embedding"
    input: str | list[str]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionsRequest) -> dict[str, object]:
    payload_messages = [message.model_dump() for message in body.messages]
    return await backend.chat_completions(model=body.model, messages=payload_messages, stream=body.stream)


@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingsRequest) -> dict[str, object]:
    input_values = body.input if isinstance(body.input, list) else [body.input]
    return await backend.embeddings(model=body.model, inputs=input_values)
