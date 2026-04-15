"""
优雅关闭管理器。

功能:
- 信号处理 (SIGTERM, SIGINT)
- 平滑停止接收新任务
- 等待处理中的任务完成
- 超时强制退出
- Checkpoint 保存

用法:
    from utils.graceful import shutdown_manager, GracefulShutdown

    # 注册
    shutdown_manager.register_handler()

    # 检查是否正在关闭
    if shutdown_manager.is_shutting_down():
        await shutdown_manager.wait_for_completion(timeout=60)

    # 手动触发
    shutdown_manager.begin_shutdown()
"""

import os
import sys
import time
import signal
import asyncio
import threading
from typing import Callable, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum


class ShutdownState(Enum):
    """关闭状态。"""
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    WAITING_TASKS = "waiting_tasks"
    FORCE_KILLING = "force_killing"
    DONE = "done"


@dataclass
class ShutdownConfig:
    """关闭配置。"""
    timeout: float = 60.0           # 总超时时间(秒)
    task_wait_timeout: float = 30.0  # 等待任务完成超时
    force_kill_delay: float = 5.0   # 强制杀死的延迟
    checkpoint_interval: float = 10.0  # Checkpoint 保存间隔


@dataclass
class CheckpointData:
    """Checkpoint 数据。"""
    stats: dict = field(default_factory=dict)
    held_tasks: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    shutdown_state: str = "running"


class GracefulShutdown:
    """
    优雅关闭管理器。

    使用方式:
    1. 初始化时注册信号处理器
    2. 定期检查 is_shutting_down()
    3. 完成后调用 mark_complete()

    线程安全，支持多线程/多进程。
    """

    _instance: Optional["GracefulShutdown"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式。"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: ShutdownConfig = None):
        if self._initialized:
            return

        self.config = config or ShutdownConfig()
        self._state = ShutdownState.RUNNING
        self._shutdown_start_time: Optional[float] = None
        self._lock = threading.RLock()

        # 回调函数
        self._on_shutdown_begin: List[Callable] = []
        self._on_shutdown_complete: List[Callable] = []
        self._task_checker: Optional[Callable] = None
        self._task_waiter: Optional[Callable] = None

        # Checkpoint
        self._checkpoint_callback: Optional[Callable] = None
        self._last_checkpoint_time = time.time()

        # 事件
        self._shutdown_event = threading.Event()

        self._initialized = True

    @property
    def state(self) -> ShutdownState:
        """获取当前状态。"""
        return self._state

    @property
    def is_shutting_down(self) -> bool:
        """检查是否正在关闭。"""
        return self._state != ShutdownState.RUNNING

    def register_handler(self):
        """
        注册信号处理器。

        处理 SIGTERM 和 SIGINT 信号。
        """
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n[Shutdown] 收到信号: {sig_name}")
            self.begin_shutdown()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def on_shutdown_begin(self, callback: Callable):
        """注册关闭开始时的回调。"""
        self._on_shutdown_begin.append(callback)

    def on_shutdown_complete(self, callback: Callable):
        """注册关闭完成时的回调。"""
        self._on_shutdown_complete.append(callback)

    def set_task_checker(self, checker: Callable[[], int]):
        """
        设置任务检查器。

        Args:
            checker: 返回当前待处理任务数量的函数
        """
        self._task_checker = checker

    def set_task_waiter(self, waiter: Callable[[float], None]):
        """
        设置任务等待器。

        Args:
            waiter: 等待任务完成的函数，参数为超时时间
        """
        self._task_waiter = waiter

    def set_checkpoint_callback(self, callback: Callable[[CheckpointData], None]):
        """
        设置 checkpoint 保存回调。

        Args:
            callback: 保存 checkpoint 的函数
        """
        self._checkpoint_callback = callback

    def begin_shutdown(self, reason: str = "signal"):
        """
        开始关闭流程。

        Args:
            reason: 关闭原因
        """
        with self._lock:
            if self._state != ShutdownState.RUNNING:
                return

            self._state = ShutdownState.SHUTTING_DOWN
            self._shutdown_start_time = time.time()
            self._shutdown_event.set()

            print(f"[Shutdown] 开始优雅关闭 (reason: {reason})")

            # 执行回调
            for callback in self._on_shutdown_begin:
                try:
                    callback()
                except Exception as e:
                    print(f"[Shutdown] 回调执行错误: {e}")

    async def wait_for_completion(self, timeout: float = None) -> bool:
        """
        等待关闭完成。

        Args:
            timeout: 超时时间 (秒)

        Returns:
            True: 正常完成
            False: 超时
        """
        if timeout is None:
            timeout = self.config.timeout

        print(f"[Shutdown] 等待任务完成 (超时: {timeout}s)...")

        start = time.time()
        self._state = ShutdownState.WAITING_TASKS

        while time.time() - start < timeout:
            # 检查是否还有任务
            if self._task_checker:
                pending = self._task_checker()
                if pending == 0:
                    print("[Shutdown] 所有任务已完成")
                    break
                if pending > 0:
                    print(f"[Shutdown] 还有 {pending} 个任务处理中...")
            else:
                break

            # 如果设置了等待器，调用它
            if self._task_waiter:
                remaining = min(5.0, timeout - (time.time() - start))
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._task_waiter(remaining),
                        timeout=remaining
                    )
                except asyncio.TimeoutError:
                    pass

            await asyncio.sleep(1)

        elapsed = time.time() - start

        # 检查是否超时
        if time.time() - start >= timeout:
            print(f"[Shutdown] 等待超时 ({timeout}s)，准备强制关闭")
            self._state = ShutdownState.FORCE_KILLING

            # 强制等待一小段时间
            await asyncio.sleep(self.config.force_kill_delay)
            self._state = ShutdownState.DONE

            print("[Shutdown] 强制关闭")
        else:
            self._state = ShutdownState.DONE
            print(f"[Shutdown] 优雅关闭完成 (耗时: {elapsed:.1f}s)")

        # 执行完成回调
        for callback in self._on_shutdown_complete:
            try:
                callback()
            except Exception as e:
                print(f"[Shutdown] 完成回调执行错误: {e}")

        return True

    def mark_complete(self):
        """标记关闭完成。"""
        with self._lock:
            self._state = ShutdownState.DONE
            print("[Shutdown] 标记完成")

    def get_elapsed_time(self) -> float:
        """获取已关闭时间。"""
        if self._shutdown_start_time is None:
            return 0.0
        return time.time() - self._shutdown_start_time

    def save_checkpoint(self, stats: dict = None, held_tasks: list = None):
        """
        保存 checkpoint。

        Args:
            stats: 统计数据
            held_tasks: 持有的任务列表
        """
        if self._checkpoint_callback is None:
            return

        data = CheckpointData(
            stats=stats or {},
            held_tasks=held_tasks or [],
            timestamp=time.time(),
            shutdown_state=self._state.value,
        )

        try:
            self._checkpoint_callback(data)
            self._last_checkpoint_time = time.time()
        except Exception as e:
            print(f"[Shutdown] Checkpoint 保存失败: {e}")

    def should_save_checkpoint(self) -> bool:
        """检查是否应该保存 checkpoint。"""
        return (
            time.time() - self._last_checkpoint_time >= self.config.checkpoint_interval
        )

    def get_state_info(self) -> dict:
        """获取状态信息。"""
        return {
            "state": self._state.value,
            "elapsed_time": self.get_elapsed_time(),
            "pending_tasks": self._task_checker() if self._task_checker else None,
        }


# 全局单例
shutdown_manager = GracefulShutdown()
