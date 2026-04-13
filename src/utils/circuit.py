"""
熔断器实现，防止重试雪崩。

熔断器模式 (Circuit Breaker Pattern):
- 关闭状态 (closed): 正常请求，失败计数
- 打开状态 (open): 快速失败，拒绝请求
- 半开状态 (half-open): 探测服务是否恢复

Q&A Q56-Q7: /query 频率限制 32/s
"""

import time
import asyncio
from typing import Optional
from dataclasses import dataclass, field


class CircuitBreakerOpen(Exception):
    """熔断器打开时抛出的异常。"""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s")


@dataclass
class CircuitBreakerConfig:
    """熔断器配置。"""
    name: str = "default"
    failure_threshold: int = 5       # 触发熔断的连续失败次数
    success_threshold: int = 2       # 半开状态下连续成功次数
    timeout: float = 60.0            # 熔断持续时间(秒)
    half_open_max_calls: int = 3     # 半开状态下的最大探测请求数


class CircuitBreaker:
    """
    熔断器实现。

    状态转换:
    - closed -> open: 连续失败达到阈值
    - open -> half_open: 超时
    - half_open -> open: 探测失败
    - half_open -> closed: 连续成功达到阈值

    示例:
        breaker = CircuitBreaker("vllm", failure_threshold=5, timeout=60)

        if not breaker.can_attempt():
            raise CircuitBreakerOpen(breaker.name, breaker._retry_after())

        try:
            result = call_vllm()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.config = CircuitBreakerConfig(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
        )

        self._state: str = "closed"  # closed, open, half_open
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls: int = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """获取当前状态。"""
        return self._state

    @property
    def failure_count(self) -> int:
        """获取连续失败次数。"""
        return self._failure_count

    def _retry_after(self) -> float:
        """计算距离熔断结束剩余时间。"""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self.config.timeout - elapsed)

    async def can_attempt(self) -> bool:
        """
        检查是否可以发起请求。

        Returns:
            True: 可以发起请求
            False: 熔断器打开，快速失败
        """
        async with self._lock:
            if self._state == "closed":
                return True

            if self._state == "open":
                # 检查是否超时
                if self._retry_after() <= 0:
                    # 切换到半开状态
                    self._state = "half_open"
                    self._half_open_calls = 0
                    self._success_count = 0
                    self._failure_count = 0
                    return True
                return False

            if self._state == "half_open":
                # 半开状态下限制探测请求数
                if self._half_open_calls >= self.config.half_open_max_calls:
                    return False
                self._half_open_calls += 1
                return True

            return False

    def can_attempt_sync(self) -> bool:
        """同步版本: 检查是否可以发起请求。"""
        if self._state == "closed":
            return True

        if self._state == "open":
            return self._retry_after() <= 0

        if self._state == "half_open":
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    async def record_success(self):
        """记录成功。"""
        async with self._lock:
            self._failure_count = 0
            self._success_count += 1

            if self._state == "half_open":
                if self._success_count >= self.config.success_threshold:
                    # 恢复关闭
                    self._state = "closed"
                    self._success_count = 0

    def record_success_sync(self):
        """同步版本: 记录成功。"""
        self._failure_count = 0
        self._success_count += 1

        if self._state == "half_open":
            if self._success_count >= self.config.success_threshold:
                self._state = "closed"
                self._success_count = 0

    async def record_failure(self):
        """记录失败。"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._success_count = 0

            if self._state == "closed":
                if self._failure_count >= self.config.failure_threshold:
                    # 打开熔断器
                    self._state = "open"

            elif self._state == "half_open":
                # 任何失败都立即打开熔断器
                self._state = "open"

    def record_failure_sync(self):
        """同步版本: 记录失败。"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        self._success_count = 0

        if self._state == "closed":
            if self._failure_count >= self.config.failure_threshold:
                self._state = "open"

        elif self._state == "half_open":
            self._state = "open"

    def get_stats(self) -> dict:
        """获取熔断器统计信息。"""
        return {
            "name": self.config.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "retry_after": self._retry_after(),
            "half_open_calls": self._half_open_calls,
        }

    def reset(self):
        """重置熔断器到关闭状态。"""
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class CircuitBreakerGroup:
    """
    熔断器组，管理多个熔断器。

    用于为不同服务维护独立的熔断器。
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        name: str,
        **kwargs,
    ) -> CircuitBreaker:
        """获取或创建熔断器。"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self._breakers[name]

    async def can_attempt(self, name: str) -> bool:
        """检查指定熔断器是否可以发起请求。"""
        if name not in self._breakers:
            return True
        return await self._breakers[name].can_attempt()

    async def record_success(self, name: str):
        """记录指定熔断器的成功。"""
        if name in self._breakers:
            await self._breakers[name].record_success()

    async def record_failure(self, name: str):
        """记录指定熔断器的失败。"""
        if name in self._breakers:
            await self._breakers[name].record_failure()

    def get_all_stats(self) -> dict:
        """获取所有熔断器的统计信息。"""
        return {name: b.get_stats() for name, b in self._breakers.items()}
