"""
平台交互客户端。
实现参赛端主循环: register -> query -> ask -> inference -> submit。

Q&A 关键约束:
- /query 频率限制: 32/s
- 最大并发持有任务: 64
- 任务领取超时: 300s
- 多 message 正确性取平均
"""

import os
import sys
import asyncio
import httpx
import argparse
import random
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入工具模块
from .utils.circuit import CircuitBreaker, CircuitBreakerOpen
from .utils.metrics import metrics
from .utils.logger import setup_logger, get_logger
from .utils.graceful import shutdown_manager

from inference import run_inference


PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://10.0.0.1:8003")
TOKEN = os.environ.get("CONTESTANT_TOKEN", "your_secret_token")
TEAM_NAME = os.environ.get("CONTESTANT_NAME", "team_alpha")
CONTESTANT_PORT = int(os.environ.get("CONTESTANT_PORT", "9000"))

# Q&A 约束常量
MAX_QUERY_RATE = 32  # /query 每秒最大请求数 (Q56-Q7)
MAX_HELD_TASKS = 64  # 最大并发持有任务数 (Q56-Q4)
TASK_ACCEPT_TIMEOUT = 300  # 任务领取超时 300s (Q55)

# Checkpoint 文件路径
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "/tmp/client_checkpoint.json")

# 连接池配置
CLIENT_LIMITS = httpx.Limits(
    max_connections=int(os.environ.get("CLIENT_MAX_CONNECTIONS", "128")),
    max_keepalive_connections=int(os.environ.get("CLIENT_MAX_KEEPALIVE", "64")),
)

# 初始化日志和指标
setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"), json_output=False)
logger = get_logger("client")
metrics.inc_counter("client.startup")


# ── 速率限制器 ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    基于令牌桶的速率限制器。
    限制 /query 等请求的频率。
    """

    def __init__(self, max_rate: float, burst: Optional[float] = None):
        """
        Args:
            max_rate: 最大速率 (次/秒)
            burst: 突发容量,默认为 max_rate
        """
        self.max_rate = max_rate
        self.burst = burst or max_rate
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """
        获取令牌,等待直到可用。
        Returns: 实际等待时间(秒)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # 补充令牌
            self.tokens = min(self.burst, self.tokens + elapsed * self.max_rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # 需要等待
            wait_time = (tokens - self.tokens) / self.max_rate
            self.tokens = 0.0
            return wait_time

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌,不阻塞。Returns: 是否成功。"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            self.tokens = min(self.burst, self.tokens + elapsed * self.max_rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# ── 指数退避与重试 ─────────────────────────────────────────────────────────

@dataclass
class BackoffState:
    """追踪限流退避状态。"""
    base_delay: float = 1.0       # 初始延迟 1 秒
    max_delay: float = 30.0      # 最大延迟 30 秒
    current_delay: float = 1.0    # 当前延迟
    consecutive_failures: int = 0  # 连续失败次数

    def record_success(self):
        """成功后重置状态。"""
        self.consecutive_failures = 0
        self.current_delay = self.base_delay

    def record_failure(self) -> float:
        """记录失败并返回下一次重试的延迟。"""
        self.consecutive_failures += 1
        # 指数退避 + 抖动
        delay = min(self.current_delay * 2, self.max_delay)
        delay *= (0.5 + random.random() * 0.5)  # 添加抖动 [0.5x, 1.0x]
        self.current_delay = delay
        return delay


# ── 任务持有管理器 ──────────────────────────────────────────────────────────

class TaskHolder:
    """
    管理已接受但未提交的任务。
    支持:
    - 最多持有 64 个任务 (Q56-Q4)
    - 任务超时追踪 (300s 领取超时)
    - 按截止时间排序处理
    """

    def __init__(self, max_held: int = MAX_HELD_TASKS, timeout_s: int = TASK_ACCEPT_TIMEOUT):
        self.max_held = max_held
        self.timeout_s = timeout_s
        self.tasks: deque = deque()  # (task, accept_time)
        self._lock = asyncio.Lock()

    async def add_task(self, task: Dict[str, Any]) -> bool:
        """添加任务,超容返回 False。"""
        async with self._lock:
            if len(self.tasks) >= self.max_held:
                return False
            accept_time = time.monotonic()
            self.tasks.append((task, accept_time))
            self._sort_by_deadline()
            return True

    def _sort_by_deadline(self):
        """按截止时间排序,最紧急的在前。"""
        def get_deadline(item):
            task, _ = item
            overview = task.get("overview", {})
            if isinstance(overview, dict):
                deadline_ms = overview.get("deadline_ms")
                if deadline_ms:
                    return deadline_ms
            return float("inf")

        self.tasks = deque(sorted(self.tasks, key=get_deadline))

    async def get_task(self) -> Optional[tuple]:
        """获取下一个待处理任务(不移除)。"""
        async with self._lock:
            if self.tasks:
                return self.tasks[0]
            return None

    async def pop_task(self) -> Optional[tuple]:
        """取出并移除最早的任务。"""
        async with self._lock:
            if self.tasks:
                return self.tasks.popleft()
            return None

    async def mark_done(self, task_id: int) -> bool:
        """标记任务完成,从持有列表中移除。"""
        async with self._lock:
            for i, (task, _) in enumerate(self.tasks):
                task_overview = task.get("overview", {})
                if task_overview.get("task_id") == task_id:
                    del self.tasks[i]
                    return True
            return False

    async def get_expired(self) -> List[tuple]:
        """获取已超时的任务。"""
        now = time.monotonic()
        expired = []
        valid = deque()

        async with self._lock:
            for item in self.tasks:
                task, accept_time = item
                if now - accept_time > self.timeout_s:
                    expired.append(item)
                else:
                    valid.append(item)
            self.tasks = valid

        return expired

    async def size(self) -> int:
        """获取当前持有任务数。"""
        async with self._lock:
            return len(self.tasks)

    async def has_capacity(self) -> bool:
        """检查是否还有容量接受新任务。"""
        async with self._lock:
            return len(self.tasks) < self.max_held

    async def get_stats(self) -> Dict[str, Any]:
        """获取持有器统计。"""
        async with self._lock:
            now = time.monotonic()
            expired_count = sum(
                1 for _, accept_time in self.tasks
                if now - accept_time > self.timeout_s
            )
            return {
                "held": len(self.tasks),
                "max": self.max_held,
                "expired": expired_count,
                "capacity": self.max_held - len(self.tasks),
            }


# ── 熔断器组 ─────────────────────────────────────────────────────────────

# 为不同 API 维护独立的熔断器
_query_breaker = CircuitBreaker(
    name="query",
    failure_threshold=10,
    timeout=60.0,
)
_ask_breaker = CircuitBreaker(
    name="ask",
    failure_threshold=5,
    timeout=120.0,
)
_submit_breaker = CircuitBreaker(
    name="submit",
    failure_threshold=5,
    timeout=120.0,
)


def get_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """获取所有熔断器。"""
    return {
        "query": _query_breaker,
        "ask": _ask_breaker,
        "submit": _submit_breaker,
    }


# ── Checkpoint ──────────────────────────────────────────────────────────

def save_checkpoint(
    stats: Dict[str, Any],
    holder_stats: Dict[str, Any] = None,
):
    """保存 checkpoint 到文件。"""
    data = {
        "timestamp": time.time(),
        "stats": stats,
        "holder_stats": holder_stats,
    }
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f)
        logger.debug("checkpoint_saved", file=CHECKPOINT_FILE)
    except Exception as e:
        logger.error("checkpoint_save_failed", error=str(e))


def load_checkpoint() -> Optional[Dict[str, Any]]:
    """从文件加载 checkpoint。"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        logger.info("checkpoint_loaded", timestamp=data.get("timestamp"))
        return data
    except Exception as e:
        logger.error("checkpoint_load_failed", error=str(e))
        return None


async def register(client: httpx.AsyncClient) -> bool:
    """向平台注册。"""
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/register",
            json={"name": TEAM_NAME, "token": TOKEN},
        )
        if resp.status_code == 200:
            logger.info("registration_success", team=TEAM_NAME)
            metrics.inc_counter("client.registration.success")
            return True
        else:
            logger.error("registration_failed", status=resp.status_code, response=resp.text)
            metrics.inc_counter("client.registration.failure")
            return False
    except Exception as e:
        logger.exception("registration_error", error=str(e))
        metrics.inc_counter("client.registration.error")
        return False


async def query_task(client: httpx.AsyncClient, backoff: Optional[BackoffState] = None) -> Optional[Dict[str, Any]]:
    """查询可用任务,包含重试逻辑和熔断器保护。"""
    # 检查熔断器
    if not await _query_breaker.can_attempt():
        retry_after = _query_breaker._retry_after()
        logger.warning("query_circuit_open", retry_after=retry_after)
        return None

    try:
        resp = await client.post(
            f"{PLATFORM_URL}/query",
            json={"token": TOKEN},
            timeout=30.0,
        )

        if resp.status_code == 200:
            _query_breaker.record_success_sync()
            if backoff:
                backoff.record_success()
            return resp.json()
        elif resp.status_code == 404:
            return None  # 没有可用任务
        elif resp.status_code == 429:
            _query_breaker.record_failure_sync()
            logger.warning("query_rate_limited")
            return None
        else:
            logger.error("query_failed", status=resp.status_code, response=resp.text)
            _query_breaker.record_failure_sync()
            return None
    except httpx.TimeoutException:
        _query_breaker.record_failure_sync()
        logger.warning("query_timeout")
        return None
    except Exception as e:
        _query_breaker.record_failure_sync()
        logger.exception("query_error", error=str(e))
        return None


async def accept_task(client: httpx.AsyncClient, task_id: int, target_sla: str, backoff: Optional[BackoffState] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """接受任务,对瞬时失败进行重试,带熔断器保护。"""
    # 检查熔断器
    if not await _ask_breaker.can_attempt():
        retry_after = _ask_breaker._retry_after()
        logger.warning("ask_circuit_open", task_id=task_id, retry_after=retry_after)
        return None

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{PLATFORM_URL}/ask",
                json={
                    "token": TOKEN,
                    "task_id": task_id,
                    "sla": target_sla,
                },
                timeout=30.0,
            )

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
                    delay = backoff.record_failure()
                logger.warning("ask_server_error", status=resp.status_code, attempt=attempt+1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay if backoff else 1 * (attempt + 1))
                continue
            else:
                _ask_breaker.record_failure_sync()
                return None
        except httpx.TimeoutException:
            logger.warning("ask_timeout", task_id=task_id, attempt=attempt+1)
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            logger.exception("ask_error", task_id=task_id, attempt=attempt+1, error=str(e))
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

    return None


async def submit_results(client: httpx.AsyncClient, task_data: Dict[str, Any], backoff: Optional[BackoffState] = None, max_retries: int = 3) -> bool:
    """提交推理结果,包含重试逻辑,带熔断器保护。"""
    # 检查熔断器
    if not await _submit_breaker.can_attempt():
        retry_after = _submit_breaker._retry_after()
        logger.warning("submit_circuit_open", retry_after=retry_after)
        metrics.inc_counter("client.submit.circuit_open")
        return False

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{PLATFORM_URL}/submit",
                json={
                    "user": {"name": TEAM_NAME, "token": TOKEN},
                    "msg": task_data,
                },
                timeout=60.0,
            )

            if resp.status_code == 200:
                _submit_breaker.record_success_sync()
                if backoff:
                    backoff.record_success()
                metrics.inc_counter("client.submit.success")
                return True
            elif resp.status_code == 429 or resp.status_code >= 500:
                _submit_breaker.record_failure_sync()
                if backoff:
                    delay = backoff.record_failure()
                logger.warning("submit_error", status=resp.status_code, attempt=attempt+1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay if backoff else 2 * (attempt + 1))
                continue
            else:
                _submit_breaker.record_failure_sync()
                logger.error("submit_failed", status=resp.status_code, response=resp.text)
                metrics.inc_counter("client.submit.error")
                return False
        except httpx.TimeoutException:
            _submit_breaker.record_failure_sync()
            logger.warning("submit_timeout", attempt=attempt+1)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
        except Exception as e:
            _submit_breaker.record_failure_sync()
            logger.exception("submit_error", attempt=attempt+1, error=str(e))
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

    metrics.inc_counter("client.submit.exhausted_retries")
    return False


async def process_task(
    client: httpx.AsyncClient,
    task: Dict[str, Any],
    backoff: Optional[BackoffState] = None,
    sla_level: Optional[str] = None,
) -> bool:
    """
    处理单个任务: 运行推理并提交结果。
    使用 SLA 自适应推理策略。
    """
    messages = task.get("messages", [])
    if not messages:
        logger.warning("process_task_empty_messages")
        return False

    # 从任务概览中提取截止时间
    deadline_ms = None
    overview = task.get("overview", {})
    if isinstance(overview, dict):
        deadline_ms = overview.get("deadline_ms")

    task_id = overview.get("task_id")

    # 使用 SLA 策略运行推理
    logger.info("inference_start", task_id=task_id, msg_count=len(messages), sla=sla_level or "auto")
    results = run_inference(messages, sla_level=sla_level, deadline_ms=deadline_ms)

    # 构建提交数据
    task_data = {
        "overview": overview,
        "messages": results,
        "sla_level": sla_level,
    }

    # 提交
    return await submit_results(client, task_data, backoff)


async def main_loop():
    """
    主客户端循环,支持:
    - 速率限制 (/query ≤32/s)
    - 最多持有 64 个任务
    - 任务超时追踪 (300s)
    - 优先级调度
    - 熔断器保护
    - 优雅关闭
    """
    # 注册优雅关闭处理器
    shutdown_manager.register_handler()
    shutdown_manager.set_task_checker(lambda: task_holder.max_held if False else 0)

    # 加载 checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        saved_stats = checkpoint.get("stats", {})
        logger.info("resuming_from_checkpoint", **saved_stats)

    # 性能统计
    stats = {
        "tasks_completed": 0,
        "tasks_failed": 0,
        "tasks_expired": 0,
        "queries_made": 0,
        "total_inference_time": 0.0,
    }

    backoff = BackoffState()
    rate_limiter = RateLimiter(max_rate=MAX_QUERY_RATE, burst=MAX_QUERY_RATE)
    task_holder = TaskHolder(max_held=MAX_HELD_TASKS, timeout_s=TASK_ACCEPT_TIMEOUT)

    async with httpx.AsyncClient(timeout=60, limits=CLIENT_LIMITS) as client:
        # 注册
        if not await register(client):
            logger.error("registration_failed")
            return

        # 创建并发推理 worker
        async def inference_worker():
            """从持有器中处理任务的 worker。"""
            last_checkpoint = time.time()

            while True:
                # 检查优雅关闭
                if shutdown_manager.is_shutting_down:
                    logger.info("worker_stopping", reason="shutdown")
                    break

                # 检查超时任务
                expired = await task_holder.get_expired()
                for task, _ in expired:
                    task_id = task.get("overview", {}).get("task_id")
                    logger.warning("task_expired", task_id=task_id)
                    stats["tasks_expired"] += 1
                    metrics.inc_counter("client.tasks.expired")
                    await task_holder.mark_done(task_id)

                # 获取下一个任务
                item = await task_holder.get_task()
                if item is None:
                    await asyncio.sleep(0.05)
                    continue

                task, _ = item
                overview = task.get("overview", {})
                task_id = overview.get("task_id")
                sla_level = overview.get("target_sla")

                # 估算剩余时间
                deadline_ms = overview.get("deadline_ms")
                if deadline_ms:
                    remaining = deadline_ms / 1000.0
                    if remaining < 1:
                        logger.warning("task_about_to_expire", task_id=task_id)
                        await task_holder.pop_task()
                        await task_holder.mark_done(task_id)
                        stats["tasks_expired"] += 1
                        continue

                start_time = time.monotonic()
                success = await process_task(client, task, backoff, sla_level)
                elapsed = time.monotonic() - start_time

                # 标记完成
                await task_holder.mark_done(task_id)

                async with asyncio.Lock():
                    if success:
                        stats["tasks_completed"] += 1
                        stats["total_inference_time"] += elapsed
                        metrics.inc_counter("client.tasks.completed")
                        metrics.observe_histogram("client.inference_time", elapsed)
                        logger.info("task_completed", task_id=task_id, elapsed=elapsed)
                    else:
                        stats["tasks_failed"] += 1
                        metrics.inc_counter("client.tasks.failed")
                        logger.error("task_failed", task_id=task_id)

                # 定期保存 checkpoint
                if time.time() - last_checkpoint > 60:
                    holder_stats = await task_holder.get_stats()
                    save_checkpoint(stats, holder_stats)
                    last_checkpoint = time.time()

                # 定期打印统计
                completed = stats["tasks_completed"] + stats["tasks_failed"] + stats["tasks_expired"]
                if completed % 10 == 0:
                    avg_time = stats["total_inference_time"] / max(stats["tasks_completed"], 1)
                    holder_stats = await task_holder.get_stats()
                    circuit_breakers = get_circuit_breakers()
                    logger.info(
                        "stats_update",
                        completed=stats["tasks_completed"],
                        failed=stats["tasks_failed"],
                        expired=stats["tasks_expired"],
                        held=holder_stats["held"],
                        avg_inference_time=avg_time,
                        circuit_breaker_states={
                            name: cb.state
                            for name, cb in circuit_breakers.items()
                        }
                    )

        # 启动 worker
        worker_task = asyncio.create_task(inference_worker())

        # 主循环 - 获取并持有任务
        while True:
            try:
                # 速率限制: /query ≤ 32/s
                wait_time = await rate_limiter.acquire(1.0)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # 检查持有容量
                if not await task_holder.has_capacity():
                    print(f"[Client] 已达到最大持有任务数 ({MAX_HELD_TASKS}),等待...")
                    await asyncio.sleep(1)
                    continue

                # 查询任务
                task_overview = await query_task(client, backoff)
                stats["queries_made"] += 1

                if task_overview is None:
                    # 没有可用任务,快速轮询
                    await asyncio.sleep(0.1)
                    continue

                backoff.record_success()
                task_id = task_overview.get("task_id")
                target_sla = task_overview.get("target_sla")
                target_reward = task_overview.get("target_reward", 1.0)
                deadline_ms = task_overview.get("deadline_ms")

                print(f"[Client] 发现任务 {task_id} (SLA: {target_sla}, "
                      f"奖励: {target_reward}, 截止: {deadline_ms}ms)")

                # 快速接受任务
                task = await accept_task(client, task_id, target_sla, backoff)

                if task is None:
                    await asyncio.sleep(0.1)
                    continue

                # 添加到持有器
                if await task_holder.add_task(task):
                    print(f"[Client] 任务 {task_id} 已持有 (持有数: {await task_holder.size()})")
                else:
                    # 容量满,同步处理
                    print(f"[Client] 持有器满,同步处理任务 {task_id}")
                    start_time = time.monotonic()
                    success = await process_task(client, task, backoff, target_sla)
                    elapsed = time.monotonic() - start_time

                    async with asyncio.Lock():
                        if success:
                            stats["tasks_completed"] += 1
                            stats["total_inference_time"] += elapsed
                        else:
                            stats["tasks_failed"] += 1

            except KeyboardInterrupt:
                print("\n[Client] 收到中断信号,正在退出...")
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
                break
            except Exception as e:
                print(f"[Client] 循环错误: {e}")
                await asyncio.sleep(1)


def main():
    """入口函数。"""
    parser = argparse.ArgumentParser(description="参赛客户端")
    parser.add_argument("--token", default=None, help="参赛 token")
    parser.add_argument("--name", default=None, help="队伍名称")
    parser.add_argument("--platform-url", default=None, help="平台 URL")
    args = parser.parse_args()

    # 命令行参数覆盖全局变量
    global TOKEN, TEAM_NAME, PLATFORM_URL
    if args.token:
        TOKEN = args.token
    if args.name:
        TEAM_NAME = args.name
    if args.platform_url:
        PLATFORM_URL = args.platform_url

    print(f"[Client] 启动, token={TOKEN[:8]}..., name={TEAM_NAME}")
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
