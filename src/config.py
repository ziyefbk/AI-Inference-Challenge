"""
统一配置加载模块。

从 config.yaml 加载所有配置，支持环境变量覆盖。
"""

import os
import yaml
from typing import Any, Dict, Optional


class Config:
    """全局配置单例。"""

    _instance: Optional["Config"] = None
    _data: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """加载配置文件。"""
        config_path = os.environ.get("CONFIG_PATH", "")
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """环境变量覆盖配置。"""
        env_mappings = {
            "TEAM_TOKEN": ("team", "token"),
            "TEAM_NAME": ("team", "name"),
            "PLATFORM_URL": ("platform", "url"),
            "MODEL_PATH": ("model", "path"),
            "CONTESTANT_PORT": ("server", "port"),
            "VLLM_PORT": ("vllm", "port"),
            "MAX_CONCURRENT_MESSAGES": ("inference", "max_concurrent_messages"),
            "VLLM_MAX_RETRIES": ("vllm", "max_retries"),
            "VLLM_TIMEOUT": ("vllm", "timeout"),
            "WARMUP_ENABLED": ("warmup", "enabled"),
            "LOG_LEVEL": ("logging", "level"),
            "SPECULATIVE_MODEL": ("speculative", "model"),
            "MAX_HELD_TASKS": ("task", "max_held"),
            "TASK_ACCEPT_TIMEOUT": ("task", "accept_timeout"),
            "CB_QUERY_FAILURE_THRESHOLD": ("circuit_breaker", "query", "failure_threshold"),
            "CB_QUERY_TIMEOUT": ("circuit_breaker", "query", "timeout"),
            "CB_ASK_FAILURE_THRESHOLD": ("circuit_breaker", "ask", "failure_threshold"),
            "CB_ASK_TIMEOUT": ("circuit_breaker", "ask", "timeout"),
            "CB_SUBMIT_FAILURE_THRESHOLD": ("circuit_breaker", "submit", "failure_threshold"),
            "CB_SUBMIT_TIMEOUT": ("circuit_breaker", "submit", "timeout"),
        }
        for env_key, path in env_mappings.items():
            val = os.environ.get(env_key)
            if val is not None:
                target = self._data
                for key in path[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                final_key = path[-1]
                # 类型转换
                if final_key in ("port", "max_retries", "timeout", "max_concurrent_messages",
                                 "max_held", "accept_timeout", "failure_threshold"):
                    val = int(val)
                elif final_key in ("enabled",):
                    val = val.lower() == "true"
                elif final_key in ("token", "name", "url", "path", "model", "level"):
                    val = str(val)
                target[final_key] = val

    def get(self, *keys: str, default: Any = None) -> Any:
        """按路径获取配置值，如 get("vllm", "timeout", default=120)。"""
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def team_token(self) -> str:
        return self.get("team", "token", default=os.environ.get("TEAM_TOKEN", ""))

    @property
    def team_name(self) -> str:
        return self.get("team", "name", default="contestant")

    @property
    def platform_url(self) -> str:
        return self.get("platform", "url",
                        default=os.environ.get("PLATFORM_URL", "http://127.0.0.1:8003"))

    @property
    def model_path(self) -> str:
        return self.get("model", "path", default=os.environ.get("MODEL_PATH", ""))

    @property
    def model_name(self) -> str:
        return self.model_path

    @property
    def vllm_timeout(self) -> float:
        return self.get("vllm", "timeout", default=120.0)

    @property
    def max_concurrent_messages(self) -> int:
        return self.get("inference", "max_concurrent_messages", default=20)

    @property
    def cb_query_failure_threshold(self) -> int:
        return self.get("circuit_breaker", "query", "failure_threshold", default=15)

    @property
    def cb_query_timeout(self) -> float:
        return self.get("circuit_breaker", "query", "timeout", default=30.0)

    @property
    def cb_ask_failure_threshold(self) -> int:
        return self.get("circuit_breaker", "ask", "failure_threshold", default=10)

    @property
    def cb_ask_timeout(self) -> float:
        return self.get("circuit_breaker", "ask", "timeout", default=60.0)

    @property
    def cb_submit_failure_threshold(self) -> int:
        return self.get("circuit_breaker", "submit", "failure_threshold", default=15)

    @property
    def cb_submit_timeout(self) -> float:
        return self.get("circuit_breaker", "submit", "timeout", default=120.0)

    def reload(self) -> None:
        """重新加载配置。"""
        self._load()


# 全局单例
config = Config()
