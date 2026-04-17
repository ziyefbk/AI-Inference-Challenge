"""
SLA 策略管理模块。

管理不同 SLA 级别的采样策略，支持:
- 配置加载
- 动态 SLA 降级
- 自适应策略选择
"""

import os
import json
from typing import Dict, Any, Optional, List

from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference.strategy")

# ── SLA 策略存储 ─────────────────────────────────────────────────────────

SAMPLING_PARAMS: Dict[str, Dict[str, float]] = {}
SLA_LEVELS: Dict[str, Dict[str, float]] = {}
SLA_STRATEGIES: Dict[str, Dict[str, Any]] = {}
SLA_DOWNGRADE_THRESHOLDS: Dict[str, float] = {
    "high_quality": 1.2,
    "standard": 1.3,
    "fast": 1.5,
    "express": 1.8,
}
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")


# ── 配置加载 ─────────────────────────────────────────────────────────────

def load_config() -> None:
    global SAMPLING_PARAMS, SLA_LEVELS, SLA_STRATEGIES, SLA_DOWNGRADE_THRESHOLDS
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        _build_default_strategies()
        return
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if "sampling_params" in config:
        SAMPLING_PARAMS = config["sampling_params"]
    if "sla_levels" in config:
        SLA_LEVELS = config["sla_levels"]
    if "sla_downgrade_thresholds" in config:
        SLA_DOWNGRADE_THRESHOLDS = config["sla_downgrade_thresholds"]
    _build_strategies_from_config()


def _build_strategies_from_config() -> None:
    """从配置文件构建 SLA_STRATEGIES。"""
    global SLA_STRATEGIES
    sampling_keys = ["Deterministic", "Normal", "HighEntropy", "ExtremePenalty"]
    sla_keys = list(SLA_LEVELS.keys())

    for i, sla in enumerate(sla_keys):
        sp_key = sampling_keys[i % len(sampling_keys)]
        sp = SAMPLING_PARAMS.get(sp_key, SAMPLING_PARAMS.get("Normal", {}))
        SLA_STRATEGIES[sla] = {
            "max_model_len": 32768,
            "temperature": sp.get("temperature", 0.1),
            "top_p": sp.get("top_p", 0.9),
            "top_k": int(sp.get("top_k", 50)),
            "repetition_penalty": sp.get("repetition_penalty", 1.0),
            "frequency_penalty": sp.get("frequency_penalty", 0.0),
            "presence_penalty": sp.get("presence_penalty", 0.0),
            "beam_size": 1,
            "logprobs_requested": 1,
            "n": 1,
            "max_gen_toks": 100000,
            "sampling_key": sp_key,
        }


def _build_default_strategies() -> None:
    """配置文件不存在时的兜底策略。"""
    global SLA_STRATEGIES, SLA_LEVELS
    SLA_LEVELS = {
        "express": {"ttft_avg": 0.5},
        "fast": {"ttft_avg": 2.0},
        "standard": {"ttft_avg": 5.0},
        "high_quality": {"ttft_avg": 15.0},
    }
    SLA_STRATEGIES = {
        "express": {
            "max_model_len": 32768, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_gen_toks": 128, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
            "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 1, "n": 1,
            "sampling_key": "Deterministic",
        },
        "fast": {
            "max_model_len": 32768, "temperature": 0.1, "top_p": 0.95, "top_k": 50,
            "max_gen_toks": 256, "repetition_penalty": 1.1, "frequency_penalty": 0.2,
            "presence_penalty": 0.1, "beam_size": 1, "logprobs_requested": 1, "n": 1,
            "sampling_key": "Normal",
        },
        "standard": {
            "max_model_len": 32768, "temperature": 0.1, "top_p": 0.9, "top_k": 50,
            "max_gen_toks": 512, "repetition_penalty": 1.1, "frequency_penalty": 0.2,
            "presence_penalty": 0.2, "beam_size": 1, "logprobs_requested": 10, "n": 1,
            "sampling_key": "Normal",
        },
        "high_quality": {
            "max_model_len": 32768, "temperature": 0.3, "top_p": 0.95, "top_k": 100,
            "max_gen_toks": 1024, "repetition_penalty": 1.2, "frequency_penalty": 0.3,
            "presence_penalty": 0.2, "beam_size": 1, "logprobs_requested": 10, "n": 1,
            "sampling_key": "HighEntropy",
        },
    }


# ── 策略获取 ─────────────────────────────────────────────────────────────

def get_sla_strategy(sla_level: str) -> Dict[str, Any]:
    if sla_level in SLA_STRATEGIES:
        return SLA_STRATEGIES[sla_level]
    for v in SLA_STRATEGIES.values():
        return v
    raise RuntimeError("No SLA_STRATEGIES available")


def get_sla_from_deadline(
    deadline_ms: Optional[int],
    estimated_duration: float = None,
    msg_count: int = None,
) -> str:
    """根据 deadline 和估算时长确定 SLA 级别。"""
    if deadline_ms is None:
        for key in SLA_LEVELS:
            return key
        return "standard"

    deadline_s = deadline_ms / 1000.0

    if SLA_LEVELS:
        ordered = sorted(SLA_LEVELS.keys(), key=lambda k: SLA_LEVELS[k].get("ttft_avg", 999))
        for sla in ordered:
            if deadline_s <= SLA_LEVELS[sla].get("ttft_avg", 999):
                base_sla = sla
                break
        else:
            base_sla = ordered[-1]
    else:
        base_sla = "standard"
        if deadline_s <= 1.0:
            base_sla = "express"
        elif deadline_s <= 5.0:
            base_sla = "fast"
        elif deadline_s <= 30.0:
            base_sla = "standard"
        else:
            base_sla = "high_quality"

    if estimated_duration and estimated_duration > 0:
        ratio = deadline_s / estimated_duration
        threshold = SLA_DOWNGRADE_THRESHOLDS.get(base_sla, 1.3)

        if ratio < threshold:
            from src.utils.metrics import metrics
            metrics.inc_counter("inference.sla_downgraded", labels={"from": base_sla})
            if ratio < 0.8:
                if SLA_LEVELS:
                    return sorted(SLA_LEVELS.keys(), key=lambda k: SLA_LEVELS[k].get("ttft_avg", 999))[0]
                return "express"
            elif ratio < 1.0:
                if SLA_LEVELS:
                    ordered = sorted(SLA_LEVELS.keys(), key=lambda k: SLA_LEVELS[k].get("ttft_avg", 999))
                    idx = next((i for i, s in enumerate(ordered) if s == base_sla), -1)
                    return ordered[min(idx + 1, len(ordered) - 1)]
                return "fast" if base_sla in ("standard", "high_quality") else "express"
            else:
                if SLA_LEVELS:
                    ordered = sorted(SLA_LEVELS.keys(), key=lambda k: SLA_LEVELS[k].get("ttft_avg", 999))
                    idx = next((i for i, s in enumerate(ordered) if s == base_sla), -1)
                    return ordered[min(idx + 1, len(ordered) - 1)]
                return "fast" if base_sla == "high_quality" else base_sla

    return base_sla


def get_adaptive_sla_strategy(
    deadline_ms: Optional[int] = None,
    msg_count: int = 1,
    base_sla: str = None,
) -> Dict[str, Any]:
    """获取自适应 SLA 策略。"""
    if base_sla is None:
        base_sla = get_sla_from_deadline(deadline_ms)

    estimated_duration = 0.0
    if deadline_ms and base_sla in SLA_LEVELS:
        base_per_msg = SLA_LEVELS[base_sla].get("ttft_avg", 0.3)
        estimated_duration = base_per_msg * min(msg_count, 10)

    final_sla = get_sla_from_deadline(deadline_ms, estimated_duration, msg_count)
    return get_sla_strategy(final_sla)


# 事件 SLA 字段到内部策略 key 的映射
SLA_LEVEL_TO_STRATEGY: Dict[str, str] = {
    "Bronze": "standard",
    "Silver": "standard",
    "Gold": "standard",
    "Diamond": "standard",
    "Platinum": "standard",
}


def prepare_sla(messages, sla_level: Optional[str], deadline_ms: Optional[int]):
    """准备 SLA 策略的辅助函数。"""
    msg_count = len(messages)
    if sla_level is None:
        adaptive_strategy = get_adaptive_sla_strategy(deadline_ms, msg_count)
        if adaptive_strategy.get("temperature", 0) == 0 and adaptive_strategy.get("max_gen_toks", 0) <= 32:
            effective_sla = "express"
        elif adaptive_strategy.get("max_gen_toks", 0) <= 64:
            effective_sla = "fast"
        elif adaptive_strategy.get("max_gen_toks", 0) <= 128:
            effective_sla = "standard"
        else:
            effective_sla = "high_quality"
    else:
        effective_sla = SLA_LEVEL_TO_STRATEGY.get(sla_level, "standard")
        adaptive_strategy = get_adaptive_sla_strategy(deadline_ms, msg_count, effective_sla)
    return adaptive_strategy, effective_sla


load_config()
