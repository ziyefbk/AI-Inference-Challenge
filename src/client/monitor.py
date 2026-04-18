"""
任务监控模块。

将任务事件记录到文件系统，用于事后分析和调试。
"""

import os
import json
import time
from typing import Dict, Any, List

from src.utils.logger import get_logger

logger = get_logger("client.monitor")

MONITOR_DIR = os.environ.get("MONITOR_DIR", "/tmp/task_monitor")
QUERIED_FILE = os.path.join(MONITOR_DIR, "queried_tasks.jsonl")
SUBMITTED_FILE = os.path.join(MONITOR_DIR, "submitted_results.jsonl")
INFERENCE_FILE = os.path.join(MONITOR_DIR, "inference_results.jsonl")

_monitor_initialized = False


def init_monitor() -> None:
    """初始化监控目录。"""
    global _monitor_initialized
    if _monitor_initialized:
        return
    try:
        os.makedirs(MONITOR_DIR, exist_ok=True)
        _monitor_initialized = True
    except Exception as e:
        logger.warning("monitor_init_failed", error=str(e))


def _write_record(filepath: str, record: Dict[str, Any]) -> None:
    """写入单条监控记录。"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("monitor_write_failed", filepath=filepath, error=str(e))


def log_queried(task_id: int, sla: str, reward: float, msg_count: int, deadline_ms: int) -> None:
    """记录查询到的任务。"""
    _write_record(QUERIED_FILE, {
        "event": "task_queried",
        "timestamp": time.time(),
        "task_id": task_id,
        "sla": sla,
        "reward": reward,
        "msg_count": msg_count,
        "deadline_ms": deadline_ms,
    })


def log_submitted(task_id: int, result_msg_count: int, sla: str, success: bool) -> None:
    """记录提交的任务。"""
    _write_record(SUBMITTED_FILE, {
        "event": "task_submitted",
        "timestamp": time.time(),
        "task_id": task_id,
        "result_msg_count": result_msg_count,
        "sla": sla,
        "success": success,
    })


def log_inference(task_id: int, messages: List[Dict[str, Any]],
                  results: List[Dict[str, Any]], sla_level: str) -> None:
    """记录推理的输入（prompt）和输出（response/accuracy）。"""
    for msg, result in zip(messages, results):
        _write_record(INFERENCE_FILE, {
            "event": "inference_result",
            "timestamp": time.time(),
            "task_id": task_id,
            "sla": sla_level,
            "eval_request_type": msg.get("eval_request_type"),
            "prompt": msg.get("prompt"),
            "continuation": msg.get("eval_continuation"),
            "response": result.get("response"),
            "accuracy": result.get("accuracy"),
        })
