"""
客户端模块。

包含与评测平台交互的所有组件。
"""

from src.client.loop import main_loop
from src.client.platform import (
    RateLimiter,
    BackoffState,
    register,
    query_task,
    accept_task,
    reject_task,
    submit_results,
    process_task,
)
from src.client.task_holder import PriorityTaskHolder
from src.client.monitor import init_monitor, log_queried, log_submitted, log_inference

__all__ = [
    "main_loop",
    "RateLimiter",
    "BackoffState",
    "PriorityTaskHolder",
    "register",
    "query_task",
    "accept_task",
    "reject_task",
    "submit_results",
    "process_task",
    "init_monitor",
    "log_queried",
    "log_submitted",
    "log_inference",
]
