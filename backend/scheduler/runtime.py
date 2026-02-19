from __future__ import annotations

from datetime import timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.agent.queue import task_queue
from backend.agent.session_runtime import ensure_session_workspace
from backend.agent.state import create_task, now_ts, upsert_conversation
from backend.ledger.events import InputEvent
from backend.ledger.store import append_event
from backend.scheduler.store import (
    get_scheduled_task,
    list_scheduled_tasks,
    set_scheduled_task_next_run,
    touch_scheduled_task_run,
)


class ScheduledTaskRuntime:
    def __init__(self) -> None:
        self._scheduler: AsyncIOScheduler | None = None
        self._started = False

    async def start(self) -> None:
        self._started = True
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._scheduler.start()
        await self.reload()

    async def stop(self) -> None:
        scheduler = self._scheduler
        self._scheduler = None
        self._started = False
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    async def reload(self) -> dict[str, Any]:
        scheduler = self._scheduler
        if scheduler is None:
            return self.status()
        for job in scheduler.get_jobs():
            if job.id.startswith("scheduled:"):
                scheduler.remove_job(job.id)

        active_tasks = list_scheduled_tasks(status="active", limit=1000)
        loaded = 0
        for item in active_tasks:
            try:
                trigger = CronTrigger.from_crontab(item.cron, timezone=timezone.utc)
            except ValueError:
                touch_scheduled_task_run(
                    schedule_id=item.schedule_id,
                    task_id=None,
                    error=f"invalid cron: {item.cron}",
                    next_run_at=None,
                )
                continue
            job = scheduler.add_job(
                self._run_schedule_job,
                trigger=trigger,
                args=[item.schedule_id],
                id=f"scheduled:{item.schedule_id}",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            next_run = int(job.next_run_time.timestamp() * 1000) if job.next_run_time else None
            set_scheduled_task_next_run(item.schedule_id, next_run)
            loaded += 1
        return {
            **self.status(),
            "loaded": loaded,
        }

    def status(self) -> dict[str, Any]:
        scheduler_running = bool(self._scheduler and self._scheduler.running)
        jobs = [job.id for job in self._scheduler.get_jobs()] if self._scheduler else []
        return {
            "started": self._started,
            "scheduler_running": scheduler_running,
            "job_count": len(jobs),
            "jobs": jobs,
        }

    async def run_once(self, schedule_id: str) -> dict[str, Any]:
        return await self._run_schedule_job(schedule_id)

    async def _run_schedule_job(self, schedule_id: str) -> dict[str, Any]:
        item = get_scheduled_task(schedule_id)
        if item is None:
            return {"schedule_id": schedule_id, "status": "not_found"}
        if item.status != "active":
            return {"schedule_id": schedule_id, "status": f"skipped:{item.status}"}

        conversation_id = item.conversation_id or f"scheduled:{schedule_id}"
        try:
            upsert_conversation(
                conversation_id=conversation_id,
                channel=item.channel,
                channel_id=item.channel_id,
                user_id=item.user_id,
                thread_id=item.thread_id,
            )
            ensure_session_workspace(conversation_id=conversation_id, actor_id=f"scheduler:{schedule_id}")
            task = create_task(conversation_id=conversation_id, status="queued", stage="plan")
            append_event(
                InputEvent(
                    conversation_id=conversation_id,
                    task_id=task.id,
                    ts=now_ts(),
                    actor_id=f"scheduler:{schedule_id}",
                    message=item.prompt,
                )
            )
            await task_queue.enqueue_task(task.id)
            job = self._scheduler.get_job(f"scheduled:{schedule_id}") if self._scheduler else None
            next_run = int(job.next_run_time.timestamp() * 1000) if job and job.next_run_time else None
            touch_scheduled_task_run(
                schedule_id=schedule_id,
                task_id=task.id,
                error=None,
                next_run_at=next_run,
            )
            return {
                "schedule_id": schedule_id,
                "status": "queued",
                "conversation_id": conversation_id,
                "task_id": task.id,
                "next_run_at": next_run,
            }
        except Exception as exc:
            touch_scheduled_task_run(schedule_id=schedule_id, task_id=None, error=str(exc))
            return {
                "schedule_id": schedule_id,
                "status": "error",
                "error": str(exc),
            }


scheduled_task_runtime = ScheduledTaskRuntime()
