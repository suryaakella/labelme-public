import asyncio
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)


async def _cleanup_thread_engine():
    from config.database import _dispose_thread_engine
    await _dispose_thread_engine()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskQueue:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def submit_sync(self, fn: Callable, *args, task_id: str = None, **kwargs) -> str:
        task_id = task_id or str(uuid.uuid4())
        future = self.executor.submit(fn, *args, **kwargs)
        self._tasks[task_id] = {
            "future": future,
            "status": TaskStatus.RUNNING,
            "result": None,
            "error": None,
        }
        future.add_done_callback(lambda f: self._on_done(task_id, f))
        logger.info(f"Submitted sync task {task_id}")
        return task_id

    def submit_async(self, coro_fn: Callable, *args, task_id: str = None, **kwargs) -> str:
        task_id = task_id or str(uuid.uuid4())

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create a fresh DB engine for this thread's event loop
                from config.database import _create_thread_engine
                _create_thread_engine()
                return loop.run_until_complete(coro_fn(*args, **kwargs))
            finally:
                loop.run_until_complete(_cleanup_thread_engine())
                loop.close()

        future = self.executor.submit(_run)
        self._tasks[task_id] = {
            "future": future,
            "status": TaskStatus.RUNNING,
            "result": None,
            "error": None,
        }
        future.add_done_callback(lambda f: self._on_done(task_id, f))
        logger.info(f"Submitted async task {task_id}")
        return task_id

    def _on_done(self, task_id: str, future: Future):
        task = self._tasks.get(task_id)
        if not task:
            return
        try:
            task["result"] = future.result()
            task["status"] = TaskStatus.COMPLETED
        except Exception as e:
            task["error"] = str(e)
            task["status"] = TaskStatus.FAILED
            logger.error(f"Task {task_id} failed: {e}")

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id": task_id,
            "status": task["status"],
            "result": task["result"],
            "error": task["error"],
        }

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)
        logger.info("Task queue shut down")


task_queue = TaskQueue()
