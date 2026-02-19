from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from backend.agent.state import update_task_status

TaskProcessor = Callable[[str], Awaitable[None]]


class TaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._processor: TaskProcessor = self._default_processor

    def set_processor(self, processor: TaskProcessor) -> None:
        self._processor = processor

    async def enqueue_task(self, task_id: str) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        await self._queue.put(task_id)

    async def start(self) -> None:
        if self._worker_task is None:
            if self._queue is None:
                self._queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._worker_loop(), name="edge-task-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None
        self._queue = None

    async def _worker_loop(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        while True:
            task_id = await self._queue.get()
            try:
                await self._processor(task_id)
            finally:
                self._queue.task_done()

    async def _default_processor(self, task_id: str) -> None:
        update_task_status(task_id, status="running", stage="plan")
        await asyncio.sleep(0.1)
        update_task_status(task_id, status="completed", stage="answer")


task_queue = TaskQueue()
