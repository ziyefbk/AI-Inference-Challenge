"""
平台交互模块。

负责与评测平台的通信:
- 注册
- 查询/领取/拒绝任务
- 提交结果
- 速率限制、熔断器、退避策略
"""

import os
import sys
import asyncio
import time
import random
import argparse
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx
import requests

from src.config import config
from src.utils.circuit import CircuitBreaker
from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("client.platform")

PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://127.0.0.1:8003")
TOKEN = os.environ.get("TEAM_TOKEN", "")
TEAM_NAME = os.environ.get("TEAM_NAME", "contestant")

# ── 速率限制 ─────────────────────────────────────────────────────────────

class RateLimiter:
    """令牌桶速率限制器。"""

    def __init__(self, max_rate: float, burst: Optional[float] = None):
        self.max_rate = max_rate
        self.burst = burst or max_rate
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.burst, self.tokens + elapsed * self.max_rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            wait_time = (tokens - self.tokens) / self.max_rate
            self.tokens = 0.0
            return wait_time

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.burst, self.tokens + elapsed * self.max_rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# ── 指数退避 ─────────────────────────────────────────────────────────────

@dataclass
class BackoffState:
    """追踪限流退避状态。"""
    base_delay: float = 1.0
    max_delay: float = 30.0
    current_delay: float = 1.0
    consecutive_failures: int = 0

    def record_success(self):
        self.consecutive_failures = 0
        self.current_delay = self.base_delay

    def record_failure(self) -> float:
        self.consecutive_failures += 1
        delay = min(self.current_delay * 2, self.max_delay)
        delay *= (0.5 + random.random() * 0.5)
        self.current_delay = delay
        return delay


# ── 熔断器 ─────────────────────────────────────────────────────────────

_query_breaker: Optional[CircuitBreaker] = None
_ask_breaker: Optional[CircuitBreaker] = None
_submit_breaker: Optional[CircuitBreaker] = None


def _init_circuit_breakers():
    """从配置初始化熔断器（延迟导入避免循环依赖）。"""
    global _query_breaker, _ask_breaker, _submit_breaker
    _query_breaker = CircuitBreaker(
        name="query",
        failure_threshold=config.cb_query_failure_threshold,
        timeout=config.cb_query_timeout,
    )
    _ask_breaker = CircuitBreaker(
        name="ask",
        failure_threshold=config.cb_ask_failure_threshold,
        timeout=config.cb_ask_timeout,
    )
    _submit_breaker = CircuitBreaker(
        name="submit",
        failure_threshold=config.cb_submit_failure_threshold,
        timeout=config.cb_submit_timeout,
    )


def get_circuit_breakers() -> Dict[str, CircuitBreaker]:
    if _query_breaker is None:
        _init_circuit_breakers()
    return {"query": _query_breaker, "ask": _ask_breaker, "submit": _submit_breaker}


_init_circuit_breakers()


# ── 注册 ─────────────────────────────────────────────────────────────

def register(max_retries: int = 30, retry_interval: float = 2.0) -> bool:
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{PLATFORM_URL}/register",
                json={"name": TEAM_NAME, "token": TOKEN},
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("registration_success", team=TEAM_NAME)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning("register_retry", attempt=attempt + 1, max_retries=max_retries, error=str(e)[:100])
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
    logger.error("registration_exhausted", max_retries=max_retries)
    return False


# ── 任务查询/领取/拒绝 ─────────────────────────────────────────────────────

async def query_task(client: httpx.AsyncClient, backoff: Optional[BackoffState] = None) -> Optional[Dict[str, Any]]:
    if not await _query_breaker.can_attempt():
        now = time.monotonic()
        if now - _query_breaker._last_warn_time >= 10.0:
            logger.warning("query_circuit_open", retry_after=_query_breaker._retry_after())
            _query_breaker._last_warn_time = now
        return None

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = await client.post(f"{PLATFORM_URL}/query", json={"token": TOKEN}, timeout=30.0)
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            _query_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            logger.warning("query_connection_error", attempt=attempt + 1, error=str(e)[:100])
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code == 200:
            _query_breaker.record_success_sync()
            if backoff:
                backoff.record_success()
            return resp.json()
        elif resp.status_code == 404:
            if backoff:
                backoff.record_success()
            return None
        elif resp.status_code == 429:
            _query_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))
            continue
        elif resp.status_code >= 500:
            _query_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))
            continue
        else:
            logger.error("query_failed", status=resp.status_code, response=resp.text[:200])
            _query_breaker.record_failure_sync()
            return None

    logger.warning("query_exhausted_retries")
    return None


async def accept_task(
    client: httpx.AsyncClient,
    task_id: int,
    target_sla: str,
    backoff: Optional[BackoffState] = None,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    if not await _ask_breaker.can_attempt():
        now = time.monotonic()
        if now - _ask_breaker._last_warn_time >= 10.0:
            logger.warning("ask_circuit_open", task_id=task_id, retry_after=_ask_breaker._retry_after())
            _ask_breaker._last_warn_time = now
        return None

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{PLATFORM_URL}/ask",
                json={"token": TOKEN, "task_id": task_id, "sla": target_sla},
                timeout=30.0,
            )
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            _ask_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            logger.warning("ask_connection_error", task_id=task_id, attempt=attempt + 1, error=str(e)[:100])
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code == 200:
            result = resp.json()
            if result.get("status") == "accepted":
                _ask_breaker.record_success_sync()
                if backoff:
                    backoff.record_success()
                return result.get("task")
            elif result.get("status") == "rejected":
                logger.warning("task_rejected", task_id=task_id, reason=result.get("reason"))
            elif result.get("status") == "closed":
                logger.info("task_closed", task_id=task_id)
            return None
        elif resp.status_code == 429 or resp.status_code >= 500:
            _ask_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            logger.warning("ask_server_error", status=resp.status_code, task_id=task_id, attempt=attempt + 1)
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))
            continue
        else:
            _ask_breaker.record_failure_sync()
            return None

    return None


async def reject_task(
    client: httpx.AsyncClient,
    task_id: int,
    backoff: Optional[BackoffState] = None,
    reason: str = "timeout_predicted",
) -> bool:
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/reject",
            json={"token": TOKEN, "task_id": task_id, "reason": reason},
            timeout=10.0,
        )
        if resp.status_code == 200:
            logger.info("task_rejected_predicted_timeout", task_id=task_id, reason=reason)
            return True
        else:
            logger.warning("task_reject_failed", task_id=task_id, status=resp.status_code)
            return False
    except Exception as e:
        logger.error("task_reject_error", task_id=task_id, error=str(e))
        return False


async def submit_results(
    client: httpx.AsyncClient,
    task_data: Dict[str, Any],
    backoff: Optional[BackoffState] = None,
    max_retries: int = 5,
) -> bool:
    if not await _submit_breaker.can_attempt():
        now = time.monotonic()
        if now - _submit_breaker._last_warn_time >= 10.0:
            logger.warning("submit_circuit_open", retry_after=_submit_breaker._retry_after())
            _submit_breaker._last_warn_time = now
        return False

    task_id = task_data.get("overview", {}).get("task_id", "unknown")

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{PLATFORM_URL}/submit",
                json={"user": {"name": TEAM_NAME, "token": TOKEN}, "msg": task_data},
                timeout=60.0,
            )
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            _submit_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            logger.warning("submit_connection_error", task_id=task_id, attempt=attempt + 1, error=str(e)[:100])
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 200:
            _submit_breaker.record_success_sync()
            if backoff:
                backoff.record_success()
            return True
        elif resp.status_code == 429 or resp.status_code >= 500:
            _submit_breaker.record_failure_sync()
            if backoff:
                backoff.record_failure()
            logger.warning("submit_error", task_id=task_id, status=resp.status_code, attempt=attempt + 1)
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
            continue
        else:
            _submit_breaker.record_failure_sync()
            logger.error("submit_failed", task_id=task_id, status=resp.status_code, response=resp.text[:200])
            return False

    return False


# ── 任务处理 ─────────────────────────────────────────────────────────────

async def process_task(
    client: httpx.AsyncClient,
    task: Dict[str, Any],
    backoff: Optional[BackoffState] = None,
    sla_level: Optional[str] = None,
) -> bool:
    """处理单个任务: 运行推理并提交结果。"""
    from src.inference import run_inference_async
    from src.client.monitor import log_inference

    messages = task.get("messages", [])
    if not messages:
        logger.warning("process_task_empty_messages")
        return False

    deadline_ms = None
    overview = task.get("overview", {})
    if isinstance(overview, dict):
        deadline_ms = overview.get("deadline_ms")
    task_id = overview.get("task_id")

    logger.info("inference_start", task_id=task_id, msg_count=len(messages), sla=sla_level or "auto")
    results = await run_inference_async(messages, sla_level=sla_level, deadline_ms=deadline_ms)

    log_inference(task_id, messages, results, sla_level or "auto")

    task_data = {
        "overview": overview,
        "messages": results,
        "sla_level": sla_level,
    }

    return await submit_results(client, task_data, backoff)
