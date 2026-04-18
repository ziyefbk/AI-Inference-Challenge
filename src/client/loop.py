"""
主循环模块。

实现客户端主事件循环，协调任务获取、推理和提交。
"""

import os
import sys
import asyncio
import time
import subprocess
import httpx
from typing import Optional

from src.config import config
from src.client.platform import (
    RateLimiter,
    BackoffState,
    get_circuit_breakers,
    query_task,
    accept_task,
    reject_task,
    register,
    process_task,
)
from src.client.task_holder import PriorityTaskHolder, estimate_task_feasible
from src.client.monitor import init_monitor, log_queried, log_submitted
from src.inference.vllm import interrupt_vllm_requests
from src.utils.logger import setup_logger, get_logger
from src.utils.graceful import shutdown_manager
from src.utils.metrics import metrics

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("client.loop")

MAX_QUERY_RATE = 32
MAX_HELD_TASKS = int(os.environ.get("MAX_HELD_TASKS", "64"))
TASK_ACCEPT_TIMEOUT = int(os.environ.get("TASK_ACCEPT_TIMEOUT", "300"))
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "/tmp/client_checkpoint.json")
CLIENT_LIMITS = httpx.Limits(
    max_connections=int(os.environ.get("CLIENT_MAX_CONNECTIONS", "256")),
    max_keepalive_connections=int(os.environ.get("CLIENT_MAX_KEEPALIVE", "128")),
)


def save_checkpoint(stats: dict, holder_stats: Optional[dict] = None):
    data = {"timestamp": time.time(), "stats": stats, "holder_stats": holder_stats}
    with open(CHECKPOINT_FILE, "w") as f:
        import json
        json.dump(data, f)
    logger.debug("checkpoint_saved", file=CHECKPOINT_FILE)


def load_checkpoint() -> Optional[dict]:
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE) as f:
        import json
        data = json.load(f)
    logger.info("checkpoint_loaded", timestamp=data.get("timestamp"))
    return data


async def main_loop() -> None:
    """主客户端循环。"""
    shutdown_manager.register_handler()
    shutdown_manager.on_shutdown_begin(interrupt_vllm_requests)

    init_monitor()
    logger.info("monitor_initialized")

    checkpoint = load_checkpoint()
    if checkpoint:
        logger.info("resuming_from_checkpoint", **checkpoint.get("stats", {}))

    stats = {
        "tasks_completed": 0,
        "tasks_failed": 0,
        "tasks_expired": 0,
        "queries_made": 0,
        "total_inference_time": 0.0,
    }

    backoff = BackoffState()
    rate_limiter = RateLimiter(max_rate=MAX_QUERY_RATE, burst=MAX_QUERY_RATE)

    # 动态配置 Worker
    base_workers = int(os.environ.get("NUM_WORKERS", "3"))
    base_prefetch = int(os.environ.get("PREFETCH_SIZE", "8"))

    num_gpus = 1
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split("\n"))
    except Exception:
        pass

    if num_gpus > 1:
        num_workers = min(base_workers + num_gpus * 3, 32)
        prefetch_size = base_prefetch * 3
        os.environ["MAX_CONCURRENT_MESSAGES"] = str(min(32, 16 + num_gpus * 2))
        logger.info("multi_gpu_config", num_gpus=num_gpus, num_workers=num_workers, prefetch_size=prefetch_size)
    else:
        num_workers = base_workers
        prefetch_size = base_prefetch
        logger.info("single_gpu_config", num_workers=num_workers, prefetch_size=prefetch_size)

    # holder 容量应与 worker 数量匹配，避免 prefetch 超过处理能力导致任务堆积
    task_holder = PriorityTaskHolder(
        max_held=max(int(os.environ.get("MAX_HELD_TASKS", str(num_workers * 4))), num_workers * 4),
        timeout_s=TASK_ACCEPT_TIMEOUT
    )

    async with httpx.AsyncClient(timeout=60, limits=CLIENT_LIMITS) as client:
        if not register():
            logger.error("registration_failed")
            return

        async def inference_worker(worker_id: int):
            last_checkpoint = time.time()

            while True:
                if shutdown_manager.is_shutting_down:
                    logger.info("worker_stopping", worker_id=worker_id, reason="shutdown")
                    break

                item = await task_holder.pop_task()
                if item is None:
                    task_holder._check_counter += 1
                    if await task_holder.should_check_expired():
                        expired = await task_holder.get_expired()
                        for task, _ in expired:
                            tid = task.get("overview", {}).get("task_id")
                            logger.warning("task_expired", task_id=tid)
                            stats["tasks_expired"] += 1
                            metrics.inc_counter("client.tasks.expired")
                        task_holder._check_counter = 0
                    await asyncio.sleep(0.05)
                    continue

                task, _ = item
                overview = task.get("overview", {})
                task_id = overview.get("task_id")
                sla_level = overview.get("target_sla")

                start_time = time.monotonic()
                success = await process_task(client, task, backoff, sla_level)
                elapsed = time.monotonic() - start_time

                if success:
                    stats["tasks_completed"] += 1
                    stats["total_inference_time"] += elapsed
                    metrics.inc_counter("client.tasks.completed")
                    metrics.observe_histogram("client.inference_time", elapsed)
                    log_submitted(
                        task_id=task_id,
                        result_msg_count=len(task.get("messages", [])),
                        sla=sla_level,
                        success=True,
                    )
                    logger.info("task_completed", task_id=task_id, elapsed=elapsed, worker_id=worker_id)
                else:
                    stats["tasks_failed"] += 1
                    metrics.inc_counter("client.tasks.failed")
                    log_submitted(
                        task_id=task_id,
                        result_msg_count=len(task.get("messages", [])),
                        sla=sla_level,
                        success=False,
                    )
                    logger.error("task_failed", task_id=task_id, elapsed=elapsed)

                if time.time() - last_checkpoint > 60:
                    holder_stats = await task_holder.get_stats()
                    save_checkpoint(stats, holder_stats)
                    last_checkpoint = time.time()

                completed = stats["tasks_completed"] + stats["tasks_failed"] + stats["tasks_expired"]
                if completed % 10 == 0 and worker_id == 0:
                    avg_time = stats["total_inference_time"] / max(stats["tasks_completed"], 1)
                    holder_stats = await task_holder.get_stats()
                    logger.info(
                        "stats_update",
                        completed=stats["tasks_completed"],
                        failed=stats["tasks_failed"],
                        expired=stats["tasks_expired"],
                        held=holder_stats["held"],
                        avg_inference_time=avg_time,
                        circuit_breaker_states={
                            name: cb.state
                            for name, cb in get_circuit_breakers().items()
                        }
                    )

        workers = [asyncio.create_task(inference_worker(i)) for i in range(num_workers)]
        logger.info("workers_started", num_workers=num_workers)

        async def prefetch_tasks():
            holder_stats = await task_holder.get_stats()
            fill_ratio = holder_stats["held"] / max(holder_stats["max"], 1)

            if fill_ratio < 0.5:
                dynamic_size = prefetch_size * 2
            elif fill_ratio > 0.8:
                dynamic_size = 1
            else:
                dynamic_size = prefetch_size

            no_task_count = 0
            prefetched = 0
            for _ in range(dynamic_size):
                if not await task_holder.has_capacity():
                    break
                if shutdown_manager.is_shutting_down:
                    break

                task_overview = await query_task(client, backoff)
                stats["queries_made"] += 1

                if task_overview is None:
                    no_task_count += 1
                    if no_task_count >= 3:
                        await asyncio.sleep(0.5)
                        no_task_count = 0
                    continue

                task_id = task_overview.get("task_id")
                target_sla = task_overview.get("target_sla", "standard")
                target_reward = task_overview.get("target_reward", 1.0)
                deadline_ms = task_overview.get("deadline_ms")

                log_queried(
                    task_id=task_id,
                    sla=target_sla,
                    reward=target_reward,
                    msg_count=len(task_overview.get("messages", [])),
                    deadline_ms=deadline_ms,
                )

                temp_task = {"overview": task_overview, "messages": task_overview.get("messages", [])}
                if not estimate_task_feasible(temp_task, target_sla):
                    remaining = (deadline_ms / 1000.0) if deadline_ms else float("inf")
                    est_duration = estimate_task_feasible.__wrapped__(temp_task, target_sla, 1.0) if hasattr(estimate_task_feasible, "__wrapped__") else 0.5
                    logger.info(
                        "task_rejected_predicted_timeout",
                        task_id=task_id,
                        remaining=remaining,
                        reward=target_reward,
                    )
                    metrics.inc_counter("client.tasks.rejected_timeout_predicted")
                    await reject_task(client, task_id, backoff, reason="timeout_predicted")
                    continue

                task = await accept_task(client, task_id, target_sla, backoff)
                if task is None:
                    # 竞争失败：任务从 queried 集合移除，需要显式 reject 让 matcher 释放
                    await reject_task(client, task_id, backoff, reason="accept_failed")
                    continue

                if await task_holder.add_task(task):
                    prefetched += 1
                    logger.debug("task_prefetched", task_id=task_id)
                else:
                    # 任务已被接受(inflight)，但 holder 满了，需要拒绝它让 matcher 重新分配
                    await reject_task(client, task_id, backoff, reason="holder_full")
                    break

            return prefetched

        while True:
            wait_time = await rate_limiter.acquire(1.0)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            if shutdown_manager.is_shutting_down:
                logger.info("main_loop_stopping", reason="shutdown")
                break

            prefetched = await prefetch_tasks()
            if prefetched > 0:
                logger.debug("prefetch_batch", count=prefetched)

            holder_stats = await task_holder.get_stats()
            if holder_stats["capacity"] < 4:
                await asyncio.sleep(0.1)

        logger.info("waiting_for_workers")
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info("all_workers_stopped")
