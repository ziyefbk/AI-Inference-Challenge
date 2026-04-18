"""
任务持有器模块。

基于优先队列的任务持有管理，支持:
- 优先级调度
- 过期检测
- 容量管理
"""

import os
import heapq
import time
import random
import asyncio
from typing import Dict, Any, Optional, List

from src.utils.logger import get_logger

logger = get_logger("client.task_holder")

MAX_HELD_TASKS = int(os.environ.get("MAX_HELD_TASKS", "64"))
TASK_ACCEPT_TIMEOUT = int(os.environ.get("TASK_ACCEPT_TIMEOUT", "300"))
# 长短请求分桶阈值（字符数）：prompt + eval_continuation 的总长度
PROMPT_LENGTH_SHORT_THRESHOLD = int(os.environ.get("PROMPT_LENGTH_SHORT_THRESHOLD", "512"))


def classify_task_length(task: Dict[str, Any]) -> int:
    """
    返回任务的长度分类。

    0 = short 任务（可与长任务混合批处理，延迟敏感）
    1 = long  任务（需要更大 batch / 更多显存，吞吐优先）

    长度取任务中所有消息的 prompt + eval_continuation 字符数之和，
    用最大值而不是平均值 — 一条长消息就能破坏 vLLM 的 batching 效率。
    """
    messages = task.get("messages", [])
    if not messages:
        return 0
    max_len = 0
    for msg in messages:
        length = len(msg.get("prompt", ""))
        cont = msg.get("eval_continuation") or ""
        length += len(cont)
        if length > max_len:
            max_len = length
    return 0 if max_len <= PROMPT_LENGTH_SHORT_THRESHOLD else 1


class PriorityTaskHolder:
    """基于 heapq 优先队列的任务持有器，按 prompt 长度分桶。

    short_queue: 短 prompt 任务（< PROMPT_LENGTH_SHORT_THRESHOLD 字符）
    long_queue:  长 prompt 任务

    pop_task() 采用短任务优先策略：优先取 short_queue，
    仅在 short_queue 为空时才取 long_queue，避免长任务饿死。
    """

    def __init__(self, max_held: int = MAX_HELD_TASKS, timeout_s: int = TASK_ACCEPT_TIMEOUT):
        self.max_held = max_held
        self.timeout_s = timeout_s
        self._short_heap: List[tuple] = []
        self._long_heap: List[tuple] = []
        # task_id -> (priority, task, accept_time, bucket)  用于快速查找
        self._task_map: Dict[int, tuple] = {}
        self._lock = asyncio.Lock()
        self._counter = 0
        self._check_counter = 0

    def _heap_for_bucket(self, bucket: int) -> List[tuple]:
        return self._short_heap if bucket == 0 else self._long_heap

    def _compute_priority(self, task: Dict[str, Any], accept_time: float) -> float:
        """计算任务优先级，越小越优先。"""
        overview = task.get("overview", {})
        if not isinstance(overview, dict):
            return float("inf")

        deadline_ms = overview.get("deadline_ms", float("inf"))
        msg_count = len(task.get("messages", []))
        reward = overview.get("target_reward", 1.0)
        sla = overview.get("target_sla", "standard")

        remaining = deadline_ms / 1000.0 - time.monotonic()

        if remaining < 0:
            return float("-inf")

        urgency_weight = 1.0
        complexity_penalty = msg_count * 0.3
        reward_factor = reward * 20.0 / max(remaining, 0.5)
        sla_adjustment = {"express": -50, "fast": -20, "standard": 0, "high_quality": 10}.get(sla, 0)

        priority = remaining * urgency_weight + complexity_penalty - reward_factor + sla_adjustment
        priority += random.uniform(-0.2, 0.2)
        return priority

    async def add_task(self, task: Dict[str, Any]) -> bool:
        async with self._lock:
            if len(self._short_heap) + len(self._long_heap) >= self.max_held:
                return False
            accept_time = time.monotonic()
            priority = self._compute_priority(task, accept_time)
            self._counter += 1
            task_id = task.get("overview", {}).get("task_id", self._counter)
            bucket = classify_task_length(task)
            heap = self._heap_for_bucket(bucket)
            heapq.heappush(heap, (priority, self._counter, accept_time, task, bucket))
            self._task_map[task_id] = (priority, task, accept_time, bucket)
            return True

    async def pop_task(self) -> Optional[tuple]:
        """短任务优先弹出；仅在短队列空时才取长队列。"""
        async with self._lock:
            # 优先取 short
            while self._short_heap:
                priority, counter, accept_time, task, bucket = heapq.heappop(self._short_heap)
                task_id = task.get("overview", {}).get("task_id", counter)
                if task_id in self._task_map:
                    del self._task_map[task_id]
                    return (task, accept_time, bucket)
            # fallback: long
            while self._long_heap:
                priority, counter, accept_time, task, bucket = heapq.heappop(self._long_heap)
                task_id = task.get("overview", {}).get("task_id", counter)
                if task_id in self._task_map:
                    del self._task_map[task_id]
                    return (task, accept_time, bucket)
            return None

    async def mark_done(self, task_id: int) -> bool:
        async with self._lock:
            if task_id in self._task_map:
                del self._task_map[task_id]
                return True
            return False

    async def cleanup_done(self):
        async with self._lock:
            self._short_heap = [(p, c, at, t, b) for p, c, at, t, b in self._short_heap
                                if t.get("overview", {}).get("task_id") in self._task_map]
            self._long_heap = [(p, c, at, t, b) for p, c, at, t, b in self._long_heap
                               if t.get("overview", {}).get("task_id") in self._task_map]
            heapq.heapify(self._short_heap)
            heapq.heapify(self._long_heap)

    async def get_expired(self) -> List[tuple]:
        now = time.monotonic()
        expired = []
        async with self._lock:
            short_valid = []
            for priority, counter, accept_time, task, bucket in self._short_heap:
                if now - accept_time > self.timeout_s:
                    expired.append((task, accept_time))
                    task_id = task.get("overview", {}).get("task_id", counter)
                    self._task_map.pop(task_id, None)
                else:
                    short_valid.append((priority, counter, accept_time, task, bucket))
            self._short_heap = short_valid

            long_valid = []
            for priority, counter, accept_time, task, bucket in self._long_heap:
                if now - accept_time > self.timeout_s:
                    expired.append((task, accept_time))
                    task_id = task.get("overview", {}).get("task_id", counter)
                    self._task_map.pop(task_id, None)
                else:
                    long_valid.append((priority, counter, accept_time, task, bucket))
            self._long_heap = long_valid

            heapq.heapify(self._short_heap)
            heapq.heapify(self._long_heap)
        return expired

    async def should_check_expired(self) -> bool:
        fill_ratio = (len(self._short_heap) + len(self._long_heap)) / max(self.max_held, 1)
        if fill_ratio > 0.7:
            return True
        elif fill_ratio > 0.4:
            return self._check_counter >= 3
        else:
            return self._check_counter >= 5

    async def size(self) -> int:
        async with self._lock:
            return len(self._short_heap) + len(self._long_heap)

    async def has_capacity(self) -> bool:
        async with self._lock:
            return len(self._short_heap) + len(self._long_heap) < self.max_held

    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            now = time.monotonic()
            short_expired = sum(1 for _, _, at, _, _ in self._short_heap
                                if now - at > self.timeout_s)
            long_expired = sum(1 for _, _, at, _, _ in self._long_heap
                               if now - at > self.timeout_s)
            total = len(self._short_heap) + len(self._long_heap)
            return {
                "held": total,
                "max": self.max_held,
                "expired": short_expired + long_expired,
                "short_held": len(self._short_heap),
                "long_held": len(self._long_heap),
                "capacity": self.max_held - total,
            }


# ── 任务可行性判断 ─────────────────────────────────────────────────────────

SLA_MESSAGE_DURATIONS: Dict[str, float] = {
    "express": 0.05,
    "fast": 0.1,
    "standard": 0.3,
    "high_quality": 0.5,
}
OVERHEAD_TIME = 0.3


def estimate_task_duration(task: Dict[str, Any], sla_level: str) -> float:
    messages = task.get("messages", [])
    msg_count = len(messages)
    base_per_msg = SLA_MESSAGE_DURATIONS.get(sla_level, 0.3)
    inference_time = base_per_msg * min(msg_count, 10)
    total_time = inference_time + OVERHEAD_TIME
    return max(total_time, 0.5)


def estimate_task_feasible(task: Dict[str, Any], sla_level: str, buffer_factor: float = 1.3) -> bool:
    """
    判断任务是否可行。

    策略：极度宽松。只要死线还没到就接受。
    理由：SLA 超时提交 = 不得分不扣分，拒绝 = 必定扣分。
    所以宁可超时完成，也不提前拒绝。
    """
    deadline_ms = task.get("overview", {}).get("deadline_ms")
    if deadline_ms is None:
        return True
    # deadline_ms 是任务分配时平台给的时间预算（相对于分配时刻）
    # 乐观估计实际推理速度：Qwen2.5-0.5B 单条消息远低于 0.3s 估算
    # 只在死线已经过了（remaining <= 0）时才拒绝
    remaining = deadline_ms / 1000.0
    # 即使死线已过，只要绝对超时没到，提交仍可避免"未完成"扣分
    # 这里保守一点：死线已过就认为不可行（因为推理本身需要时间）
    if remaining <= 0.5:
        return False
    return True
