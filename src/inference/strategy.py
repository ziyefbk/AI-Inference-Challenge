"""
SLA 策略管理模块。

管理不同 SLA 级别的采样策略，支持:
- 配置加载（CONFIG_PATH → 内置基准 → config.example.yaml，三级兜底）
- Deep merge: contest.json 覆盖 defination_base.json 基准字段
- 动态 SLA 降级
- 自适应策略选择

字段名约定: penalty_repetition / penalty_frequency / penalty_presence
（与 defination_base.json / contest.json / config.example.yaml 保持一致）
"""

import os
import json
import copy
from typing import Dict, Any, Optional, List

import yaml

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


# ── Deep Merge ────────────────────────────────────────────────────────────


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两字典: override 覆盖 base，递归合并嵌套 dict。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── 内置基准默认值 ───────────────────────────────────────────────────────

_BUILTIN_DEFINITION_BASE: Dict[str, Any] = {
    "LenSpec": {
        "Small": [0, 128],
        "Medium": [128, 256],
        "Large": [256, 512],
        "XL": [512, 1024],
    },
    "SamplingParam": {
        "Deterministic": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "penalty_repetition": 1.0,
            "penalty_frequency": 0.0,
            "penalty_presence": 0.0,
        },
        "Normal": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "penalty_repetition": 1.1,
            "penalty_frequency": 0.2,
            "penalty_presence": 0.2,
        },
        "HighEntropy": {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 100,
            "penalty_repetition": 1.05,
            "penalty_frequency": 0.0,
            "penalty_presence": 0.0,
        },
        "ExtremePenalty": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20,
            "penalty_repetition": 1.8,
            "penalty_frequency": 1.2,
            "penalty_presence": 1.2,
        },
    },
    "SLA": {
        "Bronze": {"ttft_avg": 10, "tpot_p50": 1, "tpot_p75": 2},
        "Silver": {"ttft_avg": 8, "tpot_p50": 0.8, "tpot_p75": 1.6},
        "Gold": {"ttft_avg": 6, "tpot_p50": 0.4, "tpot_p75": 0.8},
        "Platinum": {"ttft_avg": 4, "tpot_p50": 0.2, "tpot_p75": 0.4},
        "Diamond": {"ttft_avg": 2, "tpot_p50": 0.1, "tpot_p75": 0.2},
        "Stellar": {"ttft_avg": 1.5, "tpot_p50": 0.08, "tpot_p75": 0.16},
        "Glorious": {"ttft_avg": 0.8, "tpot_p50": 0.04, "tpot_p75": 0.04},
        "Supreme": {"ttft_avg": 0.5, "tpot_p50": 0.02, "tpot_p75": 0.01},
    },
}


# ── 配置加载 ─────────────────────────────────────────────────────────────

def load_config() -> None:
    """
    加载比赛规则配置，支持三级兜底:

    1. CONFIG_PATH  (contest.json, JSON 格式)
       ↓ 深度合并 (覆盖基准)
    2. 内置 _BUILTIN_DEFINITION_BASE (等同于 defination_base.json)
       ↓ 深度合并 (填充缺失字段)
    3. config.example.yaml  (YAML 格式, penalty_* 字段名)
    4. 内置默认值 (仍不足时兜底)

    加载顺序优先级: CONFIG_PATH > defination_base > config.example.yaml > 内置默认值
    """
    global SAMPLING_PARAMS, SLA_LEVELS, SLA_STRATEGIES, SLA_DOWNGRADE_THRESHOLDS

    # 1. 从 CONFIG_PATH 加载 (JSON)，与内置基准做 deep merge
    base_config: Dict[str, Any] = {}
    if CONFIG_PATH and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                contest_override = json.load(f)
            base_config = _deep_merge(_BUILTIN_DEFINITION_BASE, contest_override)
            logger.info("Loaded config from CONFIG_PATH", path=CONFIG_PATH)
        except Exception as e:
            logger.warning("Failed to load CONFIG_PATH, falling back", path=CONFIG_PATH, error=str(e))
            base_config = copy.deepcopy(_BUILTIN_DEFINITION_BASE)
    else:
        base_config = copy.deepcopy(_BUILTIN_DEFINITION_BASE)

    # 2. 从 base_config["SLA"] 推导 SLA_LEVELS 和降级阈值
    _derive_sla_config(base_config)

    # 3. 尝试加载 config.example.yaml (YAML 优先级最低，只补充缺失字段)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    example_yaml_path = os.path.join(project_root, "config.example.yaml")
    if os.path.exists(example_yaml_path):
        try:
            with open(example_yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            if "sla_downgrade_thresholds" in yaml_config:
                for key, val in yaml_config["sla_downgrade_thresholds"].items():
                    if key not in SLA_DOWNGRADE_THRESHOLDS:
                        SLA_DOWNGRADE_THRESHOLDS[key] = val
        except Exception as e:
            logger.warning("Failed to load config.example.yaml", path=example_yaml_path, error=str(e))

    # 4. 若仍无 sampling_params，使用内置兜底
    if not SAMPLING_PARAMS:
        SAMPLING_PARAMS = {
            "Deterministic": {
                "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                "penalty_repetition": 1.0, "penalty_frequency": 0.0, "penalty_presence": 0.0,
            },
            "Normal": {
                "temperature": 0.1, "top_p": 0.9, "top_k": 50,
                "penalty_repetition": 1.1, "penalty_frequency": 0.2, "penalty_presence": 0.2,
            },
            "HighEntropy": {
                "temperature": 0.1, "top_p": 0.95, "top_k": 100,
                "penalty_repetition": 1.05, "penalty_frequency": 0.0, "penalty_presence": 0.0,
            },
            "ExtremePenalty": {
                "temperature": 0.1, "top_p": 0.9, "top_k": 20,
                "penalty_repetition": 1.8, "penalty_frequency": 1.2, "penalty_presence": 1.2,
            },
        }

    _build_strategies_from_config()


def _derive_sla_config(base_config: Dict[str, Any]) -> None:
    """
    从 base_config["SLA"] 填充 SLA_LEVELS 和 SLA_DOWNGRADE_THRESHOLDS。
    从 base_config["SamplingParam"] 补全 SAMPLING_PARAMS。
    """
    global SLA_LEVELS, SLA_DOWNGRADE_THRESHOLDS, SAMPLING_PARAMS

    if "SLA" in base_config:
        for sla_name, sla_params in base_config["SLA"].items():
            SLA_LEVELS[sla_name] = sla_params
            tpot_p50 = sla_params.get("tpot_p50", 0.3)
            if sla_name not in SLA_DOWNGRADE_THRESHOLDS:
                SLA_DOWNGRADE_THRESHOLDS[sla_name] = round(1.0 + tpot_p50 * 2, 2)

    if "SamplingParam" in base_config:
        for name, params in base_config["SamplingParam"].items():
            if name not in SAMPLING_PARAMS:
                SAMPLING_PARAMS[name] = params


# ── 构建策略表 ────────────────────────────────────────────────────────────

def _build_strategies_from_config() -> None:
    """从当前 SAMPLING_PARAMS 和 SLA_LEVELS 构建 SLA_STRATEGIES。"""
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
            "repetition_penalty": sp.get("penalty_repetition", sp.get("repetition_penalty", 1.0)),
            "frequency_penalty": sp.get("penalty_frequency", sp.get("frequency_penalty", 0.0)),
            "presence_penalty": sp.get("penalty_presence", sp.get("presence_penalty", 0.0)),
            "beam_size": 1,
            "logprobs_requested": 1,
            "n": 1,
            "max_gen_toks": 1000,
            "sampling_key": sp_key,
        }


def _build_default_strategies() -> None:
    """配置文件不存在时的兜底策略（仅在加载完全失败时使用）。"""
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


# 比赛 SLA 级别名称 → 内部采样策略 key 的映射
SLA_LEVEL_TO_STRATEGY: Dict[str, str] = {
    "Bronze": "standard",
    "Silver": "standard",
    "Gold": "standard",
    "Platinum": "standard",
    "Diamond": "high_quality",
    "Stellar": "high_quality",
    "Glorious": "express",
    "Supreme": "express",
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
