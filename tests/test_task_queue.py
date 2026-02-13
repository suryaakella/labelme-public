import time

from config.task_queue import TaskQueue, TaskStatus


def test_submit_sync():
    q = TaskQueue(max_workers=2)
    try:
        tid = q.submit_sync(lambda: 42)
        time.sleep(0.5)
        status = q.get_status(tid)
        assert status["status"] == TaskStatus.COMPLETED
        assert status["result"] == 42
    finally:
        q.shutdown()


def test_submit_sync_failure():
    q = TaskQueue(max_workers=2)
    try:
        def fail():
            raise ValueError("boom")
        tid = q.submit_sync(fail)
        time.sleep(0.5)
        status = q.get_status(tid)
        assert status["status"] == TaskStatus.FAILED
        assert "boom" in status["error"]
    finally:
        q.shutdown()


def test_submit_async():
    import asyncio
    q = TaskQueue(max_workers=2)
    try:
        async def coro():
            await asyncio.sleep(0.1)
            return "async_result"

        tid = q.submit_async(coro)
        time.sleep(1)
        status = q.get_status(tid)
        assert status["status"] == TaskStatus.COMPLETED
        assert status["result"] == "async_result"
    finally:
        q.shutdown()


def test_get_status_unknown():
    q = TaskQueue(max_workers=1)
    try:
        assert q.get_status("nonexistent") is None
    finally:
        q.shutdown()
