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
            "MODEL_NAME": ("model", "name"),
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
        }
        for env_key, (section, key) in env_mappings.items():
            val = os.environ.get(env_key)
            if val is not None:
                if section not in self._data:
                    self._data[section] = {}
                # 类型转换
                if key in ("port", "max_retries", "timeout", "max_concurrent_messages",
                           "max_held", "accept_timeout"):
                    val = int(val)
                elif key in ("enabled",):
                    val = val.lower() == "true"
                elif key in ("token", "name", "url", "path", "model", "level"):
                    val = str(val)
                self._data[section][key] = val

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
        return self.get("model", "name", default="Qwen3-32B")

    @property
    def vllm_timeout(self) -> float:
        return self.get("vllm", "timeout", default=120.0)

    @property
    def max_concurrent_messages(self) -> int:
        return self.get("inference", "max_concurrent_messages", default=20)

    def reload(self) -> None:
        """重新加载配置。"""
        self._load()


# 全局单例
config = Config()
