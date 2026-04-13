"""
结构化日志模块。

功能:
- JSON 格式日志输出
- 带 context 的日志记录
- 支持标准 logging 模块
- trace_id 请求追踪

用法:
    from utils.logger import setup_logger, get_logger

    # 初始化
    setup_logger(level="INFO", json_output=True)

    # 获取 logger
    logger = get_logger("client")
    logger.info("task_completed", task_id=123, latency_ms=500)

    # 带 context
    logger = get_logger("inference").bind(request_id="abc")
    logger.warning("slow_request", duration=5.0)
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Any, Dict, Optional
from datetime import datetime
from functools import lru_cache


class JsonFormatter(logging.Formatter):
    """
    JSON 格式日志 formatter。

    输出格式:
    {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "level": "INFO",
        "logger": "client",
        "message": "task completed",
        "task_id": 123,
        "latency_ms": 500
    }
    """

    def __init__(self, include_thread: bool = False):
        super().__init__()
        self.include_thread = include_thread

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加 extra 字段
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加线程信息
        if self.include_thread:
            log_data["thread"] = threading.current_thread().name
            log_data["thread_id"] = threading.get_ident()

        return json.dumps(log_data, default=str)


class PlainFormatter(logging.Formatter):
    """
    人类可读的日志格式。

    输出格式:
    2024-01-01 12:00:00 [INFO] client: task completed task_id=123 latency_ms=500
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        logger = record.name

        parts = [f"{timestamp} [{level}] {logger}: {record.getMessage()}"]

        # 添加 extra 字段
        if hasattr(record, "extra_fields"):
            extra_parts = [f"{k}={v}" for k, v in record.extra_fields.items()]
            parts.append(" ".join(extra_parts))

        return " ".join(parts)


class StructuredLogger:
    """
    结构化日志包装器。

    提供链式调用风格的日志记录。
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs) -> "StructuredLogger":
        """绑定 context，返回新的 logger。"""
        new_logger = StructuredLogger(self._logger)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def unbind(self, *keys) -> "StructuredLogger":
        """移除 context 中的指定键。"""
        new_logger = StructuredLogger(self._logger)
        new_logger._context = {k: v for k, v in self._context.items() if k not in keys}
        return new_logger

    def _log(self, level: int, msg: str, **kwargs):
        """内部日志方法。"""
        extra = {**self._context, **kwargs}
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(unknown)",
            0,
            msg,
            (),
            None,
        )
        record.extra_fields = extra
        self._logger.handle(record)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)


# 全局 logger 注册表
_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.RLock()
_loggers_init_done = False


def setup_logger(
    level: str = "INFO",
    json_output: bool = None,
    include_thread: bool = False,
    log_file: str = None,
) -> logging.Logger:
    """
    初始化日志系统。

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: 是否使用 JSON 格式输出 (None=根据环境变量)
        include_thread: 是否包含线程信息
        log_file: 日志文件路径 (可选)
    """
    global _loggers_init_done

    # 确定输出格式
    if json_output is None:
        json_output = os.environ.get("LOG_JSON", "false").lower() == "true"

    level_value = getattr(logging, level.upper(), logging.INFO)

    # 创建根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)

    # 清除现有 handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_value)

    if json_output:
        formatter = JsonFormatter(include_thread=include_thread)
    else:
        formatter = PlainFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (可选)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level_value)
        # 文件总是使用 JSON 格式
        file_handler.setFormatter(JsonFormatter(include_thread=True))
        root_logger.addHandler(file_handler)

    _loggers_init_done = True
    return root_logger


@lru_cache(maxsize=128)
def get_logger(name: str) -> StructuredLogger:
    """
    获取结构化日志记录器。

    Args:
        name: logger 名称

    Returns:
        StructuredLogger 实例
    """
    with _loggers_lock:
        if name not in _loggers:
            base_logger = logging.getLogger(name)
            _loggers[name] = StructuredLogger(base_logger)
        return _loggers[name]


class TraceContext:
    """
    请求追踪上下文。

    自动生成 trace_id 并绑定到日志。
    """

    def __init__(self, name: str, trace_id: str = None):
        self.name = name
        self.trace_id = trace_id or self._generate_trace_id()
        self.logger = get_logger(name)

    @staticmethod
    def _generate_trace_id() -> str:
        """生成 trace_id。"""
        return f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident():04x}"

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log(self, level: str, msg: str, **kwargs):
        """记录带 trace_id 的日志。"""
        bound_logger = self.logger.bind(trace_id=self.trace_id, **kwargs)
        getattr(bound_logger, level.lower())(msg)

    def debug(self, msg: str, **kwargs):
        self.log("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self.log("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.log("warning", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.log("error", msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self.log("critical", msg, **kwargs)
