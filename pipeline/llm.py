"""DeepSeek batch client: resumable JSONL outputs, spend ledger, off-peak scheduling.

Every labeling task in the pipeline goes through `run_batch`. Results are keyed by a
stable job id, so a restarted run skips finished jobs. Token usage from each response is
priced and appended to a shared ledger; the ledger total is checked before each request
so the per-task and overall caps hold across restarts.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx

from . import config

log = logging.getLogger(__name__)

LEDGER_PATH = config.out("llm", "ledger.jsonl")


class BudgetExceeded(RuntimeError):
    pass


def is_peak(now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.weekday() >= 5:
        return False
    return any(start <= now.hour < end for start, end in config.PEAK_WINDOWS_UTC)


def seconds_until_offpeak(now: dt.datetime | None = None) -> float:
    now = now or dt.datetime.now(dt.timezone.utc)
    t = now
    while is_peak(t):
        t = (t + dt.timedelta(minutes=5)).replace(second=0, microsecond=0)
    return max(0.0, (t - now).total_seconds())


def price_usage(usage: dict, peak: bool) -> float:
    rate = 1.0 if peak else 0.5
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = (usage.get("prompt_tokens", 0) or 0) - hit
    completion = usage.get("completion_tokens", 0) or 0
    p = config.PRICE_PEAK
    return rate * (hit * p["input_hit"] + miss * p["input_miss"] + completion * p["output"]) / 1e6


def ledger_totals() -> tuple[float, dict[str, float]]:
    total, by_task = 0.0, {}
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += r["cost_usd"]
                by_task[r["task"]] = by_task.get(r["task"], 0.0) + r["cost_usd"]
    return total, by_task


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    done[r["id"]] = r
    return done


@dataclass
class Job:
    id: str
    system: str
    user: str
    meta: dict | None = None


@dataclass
class BatchStats:
    submitted: int = 0
    ok: int = 0
    parse_errors: int = 0
    http_errors: int = 0
    cost_usd: float = 0.0
    prompt_hit: int = 0
    prompt_miss: int = 0
    completion: int = 0

    def summary(self) -> dict:
        n = max(self.ok + self.parse_errors, 1)
        return {
            **self.__dict__,
            "avg_prompt_miss": round(self.prompt_miss / n, 1),
            "avg_prompt_hit": round(self.prompt_hit / n, 1),
            "avg_completion": round(self.completion / n, 1),
            "cost_per_1k_jobs": round(1000 * self.cost_usd / n, 4),
        }


class _Writer:
    """Buffered appends; Drive-backed files are slow to open for every line."""

    def __init__(self, path: Path, flush_every: int = 50, flush_secs: float = 30.0):
        self.path = path
        self.buf: list[str] = []
        self.flush_every = flush_every
        self.flush_secs = flush_secs
        self.last = time.time()

    def add(self, record: dict) -> None:
        self.buf.append(json.dumps(record, ensure_ascii=False))
        if len(self.buf) >= self.flush_every or time.time() - self.last > self.flush_secs:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write("\n".join(self.buf) + "\n")
            self.buf.clear()
        self.last = time.time()


async def _call(client: httpx.AsyncClient, job: Job, max_tokens: int) -> tuple[dict | None, dict, str]:
    body = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": job.system},
            {"role": "user", "content": job.user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    delay = 2.0
    for attempt in range(7):
        try:
            r = await client.post("/chat/completions", json=body)
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                try:
                    return json.loads(content), usage, content
                except json.JSONDecodeError:
                    return None, usage, content
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except (httpx.TimeoutException, httpx.TransportError):
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 60)
    raise RuntimeError("retries exhausted")


async def _run(
    task: str,
    jobs: list[Job],
    out_path: Path,
    validate: Callable[[dict], dict],
    max_tokens: int,
    concurrency: int,
    allow_peak: bool,
) -> BatchStats:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    cap_task = config.BUDGET_TASK_USD[task]
    spent_total, by_task = ledger_totals()
    spent_task = by_task.get(task, 0.0)

    results = _Writer(out_path)
    ledger = _Writer(LEDGER_PATH)
    stats = BatchStats()
    queue: asyncio.Queue[Job] = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stop = asyncio.Event()

    async def worker(client: httpx.AsyncClient, wid: int) -> None:
        nonlocal spent_total, spent_task
        while not queue.empty() and not stop.is_set():
            if not allow_peak and is_peak():
                wait = seconds_until_offpeak()
                if wid == 0:
                    log.info("peak window, sleeping %.0f s", wait)
                results.flush(); ledger.flush()
                await asyncio.sleep(wait + 5)
                continue
            if spent_total >= config.BUDGET_TOTAL_USD or spent_task >= cap_task:
                stop.set()
                break
            job = queue.get_nowait()
            peak = is_peak()
            stats.submitted += 1
            try:
                parsed, usage, raw = await _call(client, job, max_tokens)
            except Exception as e:  # noqa: BLE001
                stats.http_errors += 1
                log.warning("job %s failed: %s", job.id, e)
                continue
            cost = price_usage(usage, peak)
            spent_total += cost
            spent_task += cost
            stats.cost_usd += cost
            stats.prompt_hit += usage.get("prompt_cache_hit_tokens", 0) or 0
            stats.prompt_miss += usage.get("prompt_cache_miss_tokens", 0) or 0
            stats.completion += usage.get("completion_tokens", 0) or 0
            ledger.add({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "task": task, "id": job.id, "peak": peak, "usage": usage, "cost_usd": round(cost, 8),
            })
            rec = {"id": job.id, "meta": job.meta}
            try:
                if parsed is None:
                    raise ValueError("invalid json")
                rec.update(status="ok", label=validate(parsed))
                stats.ok += 1
            except Exception as e:  # noqa: BLE001
                rec.update(status="parse_error", error=str(e)[:200], raw=raw[:1000])
                stats.parse_errors += 1
            results.add(rec)
            if stats.submitted % 500 == 0:
                log.info("%s: %d done, $%.3f task / $%.3f total", task, stats.submitted, spent_task, spent_total)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=config.DEEPSEEK_BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(120.0, connect=20.0),
        limits=limits,
    ) as client:
        await asyncio.gather(*(worker(client, i) for i in range(concurrency)))
    results.flush()
    ledger.flush()
    if stop.is_set():
        log.warning("%s stopped at budget cap: task $%.3f, total $%.3f", task, spent_task, spent_total)
    return stats


def run_batch(
    task: str,
    jobs: Iterable[Job],
    out_path: Path,
    validate: Callable[[dict], dict],
    max_tokens: int = 200,
    concurrency: int = 400,
    limit: int | None = None,
    allow_peak: bool | None = None,
) -> BatchStats:
    """Run jobs not already present (status ok) in `out_path`; parse errors are retried on rerun."""
    if allow_peak is None:
        allow_peak = os.environ.get("LLM_ALLOW_PEAK") == "1"
    done = load_done(out_path)
    todo = [j for j in jobs if j.id not in done]
    if limit is not None:
        todo = todo[:limit]
    log.info("%s: %d done, %d to run", task, len(done), len(todo))
    if not todo:
        return BatchStats()
    stats = asyncio.run(_run(task, todo, out_path, validate, max_tokens, concurrency, allow_peak))
    log.info("%s summary: %s", task, stats.summary())
    return stats


def read_results(path: Path) -> list[dict]:
    return list(load_done(path).values())
