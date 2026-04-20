from __future__ import annotations

import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def as_dict(self, *, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }
        if include_result:
            payload["result"] = self.result
        return payload


class JobManager:
    def __init__(self, max_workers: int | None = None) -> None:
        workers = max_workers or max(2, min(4, os.cpu_count() or 2))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="smartgrid-api")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def submit(self, job_type: str, fn: Callable[..., dict[str, Any]], *args, **kwargs) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status="queued",
            created_at=_utc_now_iso(),
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run_job, job_id, fn, args, kwargs)
        return record

    def _run_job(
        self,
        job_id: str,
        fn: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = _utc_now_iso()

        try:
            result = fn(*args, **kwargs)
        except Exception:
            error = traceback.format_exc()
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.completed_at = _utc_now_iso()
                record.error = error
            return

        with self._lock:
            record = self._jobs[job_id]
            record.status = "succeeded"
            record.completed_at = _utc_now_iso()
            record.result = result

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda record: record.created_at, reverse=True)
        return [record.as_dict(include_result=False) for record in jobs]


job_manager = JobManager()
