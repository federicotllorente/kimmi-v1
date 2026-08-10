"""Background job manager for long-running Kimmi tasks."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from tools.common import ensure_cache_dir

try:
    from mqtt_bus import bus as mqtt_bus
except ImportError:  # pragma: no cover
    mqtt_bus = None  # type: ignore


JobFn = Callable[["JobContext"], Dict[str, Any]]


@dataclass
class JobAnnouncement:
    job_id: str
    kind: str
    title: str
    ok: bool
    error: str = ""


@dataclass
class JobRecord:
    id: str
    kind: str
    title: str
    status: str  # queued | running | done | failed | cancelled
    created_at: float
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    summary_path: str = ""
    result_path: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobContext:
    job_id: str
    kind: str
    title: str
    job_dir: str
    stopping: Callable[[], bool]

    def write_summary(self, text: str) -> str:
        path = os.path.join(self.job_dir, "summary.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text or "")
        return path

    def write_result(self, data: Dict[str, Any]) -> str:
        path = os.path.join(self.job_dir, "result.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return path


_announcements: Deque[JobAnnouncement] = deque()
_announcements_lock = threading.Lock()


def enqueue_announcement(item: JobAnnouncement) -> None:
    with _announcements_lock:
        _announcements.append(item)


def pop_announcement() -> Optional[JobAnnouncement]:
    with _announcements_lock:
        if not _announcements:
            return None
        return _announcements.popleft()


def peek_announcement_count() -> int:
    with _announcements_lock:
        return len(_announcements)


class JobManager:
    def __init__(self, allowed_root: str, max_workers: int = 2) -> None:
        self.allowed_root = os.path.realpath(allowed_root)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kimmi-job",
        )
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._stopping = False

    def start(
        self,
        kind: str,
        title: str,
        fn: JobFn,
        **meta: Any,
    ) -> str:
        if self._stopping:
            raise RuntimeError("JobManager is shutting down")

        job_id = uuid.uuid4().hex[:12]
        job_dir = ensure_cache_dir(self.allowed_root, "jobs", job_id)
        record = JobRecord(
            id=job_id,
            kind=kind,
            title=title or kind,
            status="queued",
            created_at=time.time(),
            meta=dict(meta),
        )
        with self._lock:
            self._jobs[job_id] = record

        self._write_meta(record, job_dir)
        if mqtt_bus is not None:
            try:
                mqtt_bus.publish_job_queued(job_id, kind, record.title)
            except Exception:  # noqa: BLE001
                pass

        def _runner() -> None:
            self._run_job(record, job_dir, fn)

        fut = self._executor.submit(_runner)
        with self._lock:
            self._futures[job_id] = fut
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return asdict(rec)

    def list_active(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                asdict(r)
                for r in self._jobs.values()
                if r.status in ("queued", "running")
            ]

    def list_recent(self, limit: int = 8) -> List[Dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._jobs.values(),
                key=lambda r: r.created_at,
                reverse=True,
            )
            return [asdict(r) for r in items[: max(1, min(20, int(limit)))]]

    def read_result(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            # Try disk
            job_dir = os.path.join(
                self.allowed_root, ".kimmi_cache", "jobs", job_id
            )
            summary_path = os.path.join(job_dir, "summary.md")
            result_path = os.path.join(job_dir, "result.json")
            if not os.path.isdir(job_dir):
                return {"error": "not_found", "message": f"Unknown job_id {job_id}"}
            summary = ""
            result: Dict[str, Any] = {}
            if os.path.isfile(summary_path):
                with open(summary_path, "r", encoding="utf-8") as fh:
                    summary = fh.read()
            if os.path.isfile(result_path):
                with open(result_path, "r", encoding="utf-8") as fh:
                    result = json.load(fh)
            return {
                "job_id": job_id,
                "status": "done",
                "summary": summary,
                "result": result,
            }

        summary = ""
        result: Dict[str, Any] = {}
        if rec.summary_path and os.path.isfile(rec.summary_path):
            with open(rec.summary_path, "r", encoding="utf-8") as fh:
                summary = fh.read()
        if rec.result_path and os.path.isfile(rec.result_path):
            with open(rec.result_path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        return {
            "job_id": rec.id,
            "kind": rec.kind,
            "title": rec.title,
            "status": rec.status,
            "error": rec.error,
            "summary": summary,
            "result": result,
        }

    def shutdown(self, wait: bool = False) -> None:
        self._stopping = True
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            for rec in self._jobs.values():
                if rec.status in ("queued", "running"):
                    rec.status = "cancelled"
                    rec.finished_at = time.time()

    def _run_job(self, record: JobRecord, job_dir: str, fn: JobFn) -> None:
        if self._stopping:
            record.status = "cancelled"
            record.finished_at = time.time()
            self._write_meta(record, job_dir)
            return

        record.status = "running"
        record.started_at = time.time()
        self._write_meta(record, job_dir)

        ctx = JobContext(
            job_id=record.id,
            kind=record.kind,
            title=record.title,
            job_dir=job_dir,
            stopping=lambda: self._stopping,
        )
        ok = False
        error = ""
        try:
            data = fn(ctx) or {}
            summary = str(data.get("summary") or "")
            if summary and not os.path.isfile(os.path.join(job_dir, "summary.md")):
                record.summary_path = ctx.write_summary(summary)
            else:
                record.summary_path = os.path.join(job_dir, "summary.md")
            result_body = {k: v for k, v in data.items() if k != "summary"}
            record.result_path = ctx.write_result(result_body)
            if self._stopping:
                record.status = "cancelled"
            else:
                record.status = "done"
                ok = True
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            record.status = "failed"
            record.error = error
            try:
                record.summary_path = ctx.write_summary(f"Job failed: {error}")
                record.result_path = ctx.write_result({"error": error})
            except OSError:
                pass

        record.finished_at = time.time()
        self._write_meta(record, job_dir)

        if mqtt_bus is not None and not self._stopping:
            try:
                mqtt_bus.publish_job_done(
                    record.id,
                    record.kind,
                    record.title,
                    ok=ok,
                    summary_path=record.summary_path or None,
                    error=error or None,
                )
                body = (
                    f"Finished: {record.title}"
                    if ok
                    else f"Failed: {record.title}"
                )
                mqtt_bus.publish_notify(
                    "Kimmi",
                    body,
                    level="info" if ok else "error",
                    job_id=record.id,
                )
            except Exception:  # noqa: BLE001
                pass

        if not self._stopping:
            enqueue_announcement(
                JobAnnouncement(
                    job_id=record.id,
                    kind=record.kind,
                    title=record.title,
                    ok=ok,
                    error=error,
                )
            )

    def _write_meta(self, record: JobRecord, job_dir: str) -> None:
        path = os.path.join(job_dir, "meta.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(asdict(record), fh, ensure_ascii=False, indent=2)
        except OSError:
            pass


# Set by assistant after ALLOWED_DATA_ROOT is known.
job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global job_manager
    if job_manager is None:
        root = os.path.realpath(os.environ.get("KIMMI_ALLOWED_ROOT") or ".")
        job_manager = JobManager(root)
    return job_manager


def demo_long_job_worker(ctx: JobContext, seconds: int = 30) -> Dict[str, Any]:
    """Sleep then write a short demo summary (Phase B validation)."""
    seconds = max(5, min(120, int(seconds)))
    steps = max(1, seconds)
    for i in range(steps):
        if ctx.stopping():
            raise RuntimeError("cancelled")
        time.sleep(1.0)
        if mqtt_bus is not None and i % 5 == 0:
            try:
                mqtt_bus.publish_job_progress(
                    ctx.job_id,
                    ctx.kind,
                    ctx.title,
                    stage=f"working {i + 1}/{steps}",
                    pct=round(100.0 * (i + 1) / steps, 1),
                )
            except Exception:  # noqa: BLE001
                pass

    summary = (
        f"Demo job '{ctx.title}' finished after about {seconds} seconds. "
        "This is a placeholder summary for Phase B validation. "
        "No real research was performed."
    )
    ctx.write_summary(summary)
    return {
        "summary": summary,
        "seconds": seconds,
        "demo": True,
    }
