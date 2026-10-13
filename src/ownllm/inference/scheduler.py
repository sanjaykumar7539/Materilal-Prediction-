"""Priority queue inference scheduler for async request batching/backpressure."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass(order=True)
class Job:
    priority: int
    seq: int
    payload: object = field(compare=False)
    future: asyncio.Future = field(compare=False)


class InferenceScheduler:
    def __init__(
        self,
        handler: Callable[[Any], Awaitable[Any]],
        workers: int = 1,
        max_queue: int = 128,
    ):
        self.handler = handler
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue)
        self.seq = 0
        self.tasks: list[asyncio.Task] = []
        self.workers = workers
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        self._started = True

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        # Fail any jobs still queued so submitters unblock immediately
        # instead of hanging until their timeout.
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.set_exception(RuntimeError("inference scheduler stopped"))
            self.queue.task_done()
        self._started = False

    async def submit(self, payload: Any, priority: int = 10, timeout: float = 120.0) -> Any:
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self.seq += 1
        try:
            self.queue.put_nowait(Job(priority, self.seq, payload, fut))
        except asyncio.QueueFull as exc:
            raise RuntimeError("inference queue full") from exc
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _worker(self) -> None:
        while True:
            job: Job = await self.queue.get()
            try:
                result = await self.handler(job.payload)
                if not job.future.done():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                # Scheduler shutdown mid-job: unblock the submitter, then re-raise.
                if not job.future.done():
                    job.future.set_exception(RuntimeError("inference scheduler stopped"))
                raise
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.queue.task_done()
