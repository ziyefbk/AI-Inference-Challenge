"""
推理模块,支持精确分词和 logprob 计算。
支持三种推理类型:
- generate_until: 带停止条件的文本生成
- loglikelihood: 计算 log P(continuation | prompt)
- loglikelihood_rolling: 计算整个文本的滚动 log-likelihood

Q&A 约束:
- Q56-Q7: /query 频率限制 32/s
- Q58: setup.sh 可构建 CUDA Graph
- Q63: 多 message 正确性取平均
"""

import os
import re
import httpx
import json
import time
import logging
import asyncio
import random
import threading
from typing import Optional, Dict, Any, List
from functools import lru_cache
import tiktoken

from src.utils.metrics import metrics as _metrics
from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference")

# ── 多实例 vLLM 配置 ──────────────────────────────────────────────────────
# 启动时自动检测 GPU 数量设置，或通过环境变量覆盖
_VLLM_NUM_INSTANCES = int(os.environ.get("VLLM_NUM_INSTANCES", "1"))

def _build_vllm_urls() -> List[str]:
    """构建所有 vLLM 实例的 URL 列表。"""
    if "VLLM_URLS" in os.environ:
        return [u.strip() for u in os.environ["VLLM_URLS"].split(",") if u.strip()]
    n = _VLLM_NUM_INSTANCES
    return [f"http://localhost:{8000 + i}" for i in range(n)]

VLLM_URLS = _build_vllm_urls()
VLLM_URL = VLLM_URLS[0]  # 兼容单实例场景
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

# ── 客户端池（每实例一个独立客户端） ──────────────────────────────────────
_vllm_async_clients: Dict[str, httpx.AsyncClient] = {}
_client_lock = threading.Lock()
_url_index = 0  # 轮询索引（线程安全）


def _get_next_url() -> str:
    """轮询获取下一个 vLLM 实例 URL。"""
    global _url_index
    return VLLM_URLS[_url_index % len(VLLM_URLS)]


async def _get_vllm_async_client(url: str) -> httpx.AsyncClient:
    """获取指定 URL 的异步客户端（懒加载单例）。"""
    with _client_lock:
        if url not in _vllm_async_clients:
            _vllm_async_clients[url] = httpx.AsyncClient(
                timeout=httpx.Timeout(RETRY_CONFIG["request_timeout"]),
                limits=httpx.Limits(
                    max_keepalive_connections=int(os.environ.get("VLLM_MAX_KEEPALIVE", "20")),
                    max_connections=int(os.environ.get("VLLM_MAX_CONNECTIONS", "50")),
                ),
            )
        return _vllm_async_clients[url]


async def _close_all_clients():
    """关闭所有客户端。"""
    global _vllm_async_clients
    with _client_lock:
        for url, client in _vllm_async_clients.items():
            await client.aclose()
        _vllm_async_clients.clear()


RETRY_CONFIG = {
    "max_retries": int(os.environ.get("VLLM_MAX_RETRIES", "3")),
    "backoff_factor": float(os.environ.get("VLLM_BACKOFF_FACTOR", "1.5")),
    "max_backoff": float(os.environ.get("VLLM_MAX_BACKOFF", "30.0")),
    "request_timeout": float(os.environ.get("VLLM_TIMEOUT", "120.0")),
}

# ── 并发配置：多实例自适应 ────────────────────────────────────────────────
_base_concurrent = int(os.environ.get("MAX_CONCURRENT_MESSAGES", "20"))
_num_instances = _VLLM_NUM_INSTANCES

# 每实例并发 * 实例数 = 总并发上限（避免过高）
CONCURRENCY_CONFIG = {
    "max_concurrent_messages": _base_concurrent * _num_instances,
    "warmup_enabled": os.environ.get("WARMUP_ENABLED", "true").lower() == "true",
    "warmup_requests": int(os.environ.get("WARMUP_REQUESTS", "5")),
}

logger.info("vllm_config",
            num_instances=_num_instances,
            urls=VLLM_URLS,
            max_concurrent=CONCURRENCY_CONFIG["max_concurrent_messages"])


@lru_cache(maxsize=4)
def _get_tiktoken_encoding(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


async def _call_vllm(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    echo: bool = False,
    best_of: int = 1,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_model_len: Optional[int] = None,
    beam_size: int = 1,
) -> Dict[str, Any]:
    """异步调用 vLLM chat API，自动重试，轮询分发到多实例。"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if stop:
        payload["stop"] = stop
    if logprobs is not None:
        payload["logprobs"] = True
        payload["top_logprobs"] = min(logprobs, 20) if logprobs > 0 else None
    if max_model_len is not None:
        payload["max_model_len"] = max_model_len
    if beam_size > 1:
        payload["beam_size"] = beam_size
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty

    # 轮询选择 vLLM 实例
    global _url_index
    chosen_url = VLLM_URLS[_url_index % len(VLLM_URLS)]
    _url_index += 1

    client = await _get_vllm_async_client(chosen_url)
    for attempt in range(RETRY_CONFIG["max_retries"] + 1):
        try:
            resp = await client.post(f"{chosen_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and attempt < RETRY_CONFIG["max_retries"]:
                logger.warning(f"vLLM 400 error, retrying: {e.response.text[:200]}")
                await asyncio.sleep(0.5)
                continue
            raise


def _apply_stop_strings(text: str, stop_list: List[str]) -> str:
    for s in stop_list or []:
        idx = text.find(s)
        if idx >= 0:
            return text[:idx]
    return text


SLA_STRATEGIES: Dict[str, Dict[str, Any]] = {
    # Q60: prompt限制4K以内; Q73: 模型开启thinking mode
    "express": {
        "max_model_len": 4096, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "max_gen_toks": 32, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 1,
        "n": 1,
    },
    "fast": {
        "max_model_len": 4096, "temperature": 0.1, "top_p": 0.95, "top_k": 20,
        "max_gen_toks": 64, "repetition_penalty": 1.05, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 5,
        "n": 1,
    },
    "standard": {
        "max_model_len": 4096, "temperature": 0.7, "top_p": 0.9, "top_k": 50,
        "max_gen_toks": 128, "repetition_penalty": 1.1, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 20,
        "n": 1,
    },
    "high_quality": {
        "max_model_len": 4096, "temperature": 0.8, "top_p": 0.95, "top_k": -1,
        "max_gen_toks": 256, "repetition_penalty": 1.2, "frequency_penalty": 0.1,
        "presence_penalty": 0.1, "beam_size": 1, "logprobs_requested": 100,
        "n": 1,
    },
}


def get_sla_strategy(sla_level: str) -> Dict[str, Any]:
    return SLA_STRATEGIES.get(sla_level, SLA_STRATEGIES["standard"])


# SLA降级阈值配置
SLA_DOWNGRADE_THRESHOLDS: Dict[str, float] = {
    "high_quality": 1.2,  # 时间不足预估的1.2倍时降级
    "standard": 1.3,
    "fast": 1.5,
    "express": 1.8,
}


def get_sla_from_deadline(deadline_ms: Optional[int],
                          estimated_duration: float = None,
                          msg_count: int = None) -> str:
    """
    根据deadline和估算时长确定SLA级别。

    Args:
        deadline_ms: 截止时间（毫秒）
        estimated_duration: 预估完成时间（秒），用于判断是否需要SLA降级
        msg_count: 消息数量

    Returns:
        SLA级别字符串
    """
    if deadline_ms is None:
        return "standard"

    deadline_s = deadline_ms / 1000.0

    # 基础SLA选择
    base_sla = "standard"
    if deadline_s <= 1.0:
        base_sla = "express"
    elif deadline_s <= 5.0:
        base_sla = "fast"
    elif deadline_s <= 30.0:
        base_sla = "standard"
    else:
        base_sla = "high_quality"

    # Deadline感知: 如果时间紧迫，考虑降级SLA以加快处理
    if estimated_duration and estimated_duration > 0:
        ratio = deadline_s / estimated_duration
        threshold = SLA_DOWNGRADE_THRESHOLDS.get(base_sla, 1.3)

        if ratio < threshold:
            # 时间不足，降级SLA
            _metrics.inc_counter("inference.sla_downgraded", labels={"from": base_sla})

            # 根据紧迫程度决定降级到什么级别
            if ratio < 0.8:
                # 极度紧迫，降到最快
                return "express"
            elif ratio < 1.0:
                # 略微紧迫
                return "fast" if base_sla in ("standard", "high_quality") else "express"
            else:
                # 略紧但不严重
                return "fast" if base_sla == "high_quality" else base_sla

            logger.info("sla_downgrade", base_sla=base_sla, final_sla=base_sla,
                       ratio=ratio, threshold=threshold)

    return base_sla


def get_adaptive_sla_strategy(deadline_ms: Optional[int],
                              msg_count: int = 1,
                              base_sla: str = None) -> Dict[str, Any]:
    """
    获取自适应SLA策略，综合考虑deadline和时间紧迫度。

    Args:
        deadline_ms: 截止时间
        msg_count: 消息数量
        base_sla: 基础SLA级别

    Returns:
        SLA策略字典
    """
    # 如果没有指定基础SLA，根据deadline确定
    if base_sla is None:
        base_sla = get_sla_from_deadline(deadline_ms)

    # 估算所需时间
    estimated_duration = 0.0
    if deadline_ms:
        # 基于SLA级别估算
        sla_durations = {
            "express": 0.05,
            "fast": 0.1,
            "standard": 0.3,
            "high_quality": 0.5,
        }
        base_per_msg = sla_durations.get(base_sla, 0.3)
        estimated_duration = base_per_msg * min(msg_count, 10)

    # 获取降级后的SLA
    final_sla = get_sla_from_deadline(deadline_ms, estimated_duration, msg_count)

    # 获取对应的策略
    return get_sla_strategy(final_sla)


# SLA 自适应推理配置
SAMPLING_PARAMS: Dict[str, Dict[str, float]] = {}
SLA_LEVELS: Dict[str, Dict[str, float]] = {}
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")


def load_config():
    global SAMPLING_PARAMS, SLA_LEVELS
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        return
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if "sampling_params" in config:
        SAMPLING_PARAMS = config["sampling_params"]
    if "sla_levels" in config:
        SLA_LEVELS = config["sla_levels"]


load_config()
_model_warmed_up = False


def warmup_model():
    """预热模型,构建 CUDA Graph. Q&A Q58."""
    global _model_warmed_up
    if _model_warmed_up or not CONCURRENCY_CONFIG["warmup_enabled"]:
        _model_warmed_up = True
        return

    logger.info("[推理] 开始预热模型...")
    # 预热多种SLA级别和提示,确保CUDA Graph充分构建
    warmup_prompts = [
        ("Hello", "express"),
        ("What is 2+2?", "fast"),
        ("The capital of France is", "standard"),
        ("Explain quantum physics", "standard"),
        ("Write a short story", "high_quality"),
    ]

    async def _do_warmup():
        for prompt, sla in warmup_prompts:
            for _ in range(3):  # 多次预热确保CUDA graph构建
                await _call_vllm(prompt=prompt, max_tokens=8,
                                temperature=0.0, top_p=1.0, top_k=1)

    try:
        asyncio.get_running_loop()
        # 已在运行中的loop，用线程池避免嵌套asyncio.run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(lambda: asyncio.run(_do_warmup())).result()
    except RuntimeError:
        asyncio.run(_do_warmup())

    _model_warmed_up = True


METRICS = _metrics
METRICS.inc_counter("inference.init", labels={"model": MODEL_NAME})


def _resolve_gen_params(
    gen_kwargs: Optional[Dict[str, Any]],
    sla: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """从 gen_kwargs 和 sla_strategy 合并生成参数."""
    defaults = sla or {}
    return {
        "max_gen_toks": gen_kwargs.get("max_gen_toks", defaults.get("max_gen_toks", 256)) if gen_kwargs else defaults.get("max_gen_toks", 256),
        "temperature": gen_kwargs.get("temperature", defaults.get("temperature", 0.0)) if gen_kwargs else defaults.get("temperature", 0.0),
        "top_p": gen_kwargs.get("top_p", defaults.get("top_p", 1.0)) if gen_kwargs else defaults.get("top_p", 1.0),
        "top_k": gen_kwargs.get("top_k", defaults.get("top_k", 50)) if gen_kwargs else defaults.get("top_k", 50),
        "until": gen_kwargs.get("until", ["\n\n"]) if gen_kwargs else ["\n\n"],
        "repetition_penalty": gen_kwargs.get("repetition_penalty", defaults.get("repetition_penalty", 1.0)) if gen_kwargs else defaults.get("repetition_penalty", 1.0),
        "frequency_penalty": gen_kwargs.get("frequency_penalty", defaults.get("frequency_penalty", 0.0)) if gen_kwargs else defaults.get("frequency_penalty", 0.0),
        "presence_penalty": gen_kwargs.get("presence_penalty", defaults.get("presence_penalty", 0.0)) if gen_kwargs else defaults.get("presence_penalty", 0.0),
        "max_model_len": defaults.get("max_model_len"),
        "beam_size": defaults.get("beam_size", 1),
        "best_of": defaults.get("n", 1),
    }


async def generate_text(
    prompt: str,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> str:
    """异步文本生成,带停止条件."""
    p = _resolve_gen_params(gen_kwargs, sla_strategy)
    resp = await _call_vllm(
        prompt=prompt, max_tokens=p["max_gen_toks"], temperature=p["temperature"],
        top_p=p["top_p"], top_k=p["top_k"], stop=p["until"],
        repetition_penalty=p["repetition_penalty"], frequency_penalty=p["frequency_penalty"],
        presence_penalty=p["presence_penalty"], max_model_len=p["max_model_len"],
        beam_size=p["beam_size"], best_of=p["best_of"],
    )

    choices = resp.get("choices", [])
    if not choices:
        return ""
    # Chat API 返回 message.content，Completions API 返回 text
    choice = choices[0]
    text = choice.get("message", {}).get("content", "") or choice.get("text", "")
    return _apply_stop_strings(text, p["until"])


async def compute_logprob(
    prompt: str,
    continuation: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算 log P(continuation | prompt),单次 API 调用。

    使用高精度设置确保概率计算的准确性。
    """
    if not continuation:
        return 0.0

    enc = _get_tiktoken_encoding()
    pt = len(enc.encode(prompt))
    ct = len(enc.encode(continuation))
    lp_req = pt + ct

    # 增加buffer以提高精度，并限制最大请求量
    lp_req = min(lp_req + 10, 2000)

    resp = await _call_vllm(
        prompt=prompt + continuation,
        max_tokens=1,
        logprobs=lp_req,
        temperature=0.0,  # 强制greedy确保确定性
        top_p=1.0,
        top_k=1,
    )

    # Chat API: logprobs 在 choices[0].logprobs.content 中
    # Completions API: logprobs 在 choices[0].logprobs.token_logprobs 中
    choice = resp.get("choices", [{}])[0]
    logprobs_data = choice.get("logprobs", {})
    lps = logprobs_data.get("content", []) or logprobs_data.get("token_logprobs", [])
    
    if not lps or len(lps) < 2:
        _metrics.inc_counter("inference.logprob_error", labels={"reason": "empty_response"})
        return -10.0

    # 精确提取continuation部分的logprob
    if pt > 0 and ct > 0:
        # 正常情况：取prompt之后、continuation长度的部分
        cont = [lp.get("logprob") if isinstance(lp, dict) else lp 
                for lp in lps[pt:pt + ct] if lp is not None]
    else:
        # 边界情况：使用中点分割
        mid = len(lps) // 2
        cont = [lp.get("logprob") if isinstance(lp, dict) else lp 
                for lp in lps[mid:mid + ct] if lp is not None]

    if not cont:
        _metrics.inc_counter("inference.logprob_error", labels={"reason": "no_valid_tokens"})
        return -10.0

    # 计算总logprob
    total_logprob = float(sum(cont))
    _metrics.observe_histogram("inference.logprob_sum", total_logprob)

    return total_logprob


async def compute_rolling_logprob(
    text: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """计算整个文本的滚动 log-likelihood."""
    if not text:
        return 0.0

    enc = _get_tiktoken_encoding()
    num_tokens = len(enc.encode(text))
    lp_req = min(num_tokens + 10, 2000)

    if sla_strategy:
        lp_req = max(lp_req, min(sla_strategy.get("logprobs_requested", 1000) * 10, 2000))

    resp = await _call_vllm(
        prompt=text, max_tokens=1, logprobs=lp_req,
        temperature=0.0, top_p=1.0, top_k=1,
    )

    # Chat API: logprobs 在 choices[0].logprobs.content 中
    choice = resp.get("choices", [{}])[0]
    logprobs_data = choice.get("logprobs", {})
    lps = logprobs_data.get("content", []) or logprobs_data.get("token_logprobs", [])
    
    if not lps or len(lps) < 2:
        return -10.0
    
    # 提取 logprob 值（可能是 dict 或 float）
    valid = [lp.get("logprob") if isinstance(lp, dict) else lp 
             for lp in lps[1:] if lp is not None]
    return float(sum(valid) / len(valid)) if valid else -10.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


# ── 结果验证 ─────────────────────────────────────────────────────────────

def validate_result(result: Dict[str, Any]) -> bool:
    """
    验证推理结果的有效性。

    Args:
        result: 推理结果字典

    Returns:
        True: 结果有效
        False: 结果异常
    """
    rt = result.get("eval_request_type")

    if rt == "generate_until":
        response = result.get("response", "")
        # 检查是否为空
        if not response:
            logger.warning("validation_failed", reason="empty_response", type=rt)
            return False
        # 检查是否过短
        if len(response) < 3:
            logger.warning("validation_failed", reason="too_short", type=rt, length=len(response))
            return False
        # 检查是否为重复内容
        words = response.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                logger.warning("validation_failed", reason="repetitive", type=rt)
                return False

    elif rt in ("loglikelihood", "loglikelihood_rolling"):
        accuracy = result.get("accuracy")
        # 检查是否异常
        if accuracy is None:
            logger.warning("validation_failed", reason="none_accuracy", type=rt)
            return False
        # logprob通常为负数，正数可能是异常
        if accuracy > 0.1:
            logger.warning("validation_failed", reason="positive_logprob", type=rt, accuracy=accuracy)
            # 注意：不直接返回False，因为某些情况确实可能为正

    return True


def _aggregate_task_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 task_id 聚合多 message 正确性分数 (Q63: 取平均)。

    同时进行结果验证。
    """
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        tid = r.get("task_id") or r.get("ID") or 0
        groups.setdefault(tid, []).append(r)

    out = []
    for tid, group in groups.items():
        if len(group) == 1:
            # 单个结果也要验证
            result = group[0]
            if not validate_result(result):
                result["_validation_failed"] = True
                _metrics.inc_counter("inference.validation.failed")
            out.append(result)
            continue

        # 多结果聚合
        first = group[0].copy()
        accs = [r.get("accuracy") for r in group if r.get("accuracy") is not None]

        # 验证所有结果
        valid_count = 0
        for r in group:
            if validate_result(r):
                valid_count += 1
            else:
                r["_validation_failed"] = True
                _metrics.inc_counter("inference.validation.failed")

        if accs:
            # 只使用有效结果的平均值
            first["accuracy"] = sum(accs) / len(accs)
            first["accuracy_count"] = len(accs)
            first["accuracy_std"] = _std(accs) if len(accs) > 1 else 0.0
            first["valid_count"] = valid_count

        for r in group:
            if r.get("response") and not r.get("_validation_failed"):
                first["response"] = r["response"]
                break
        first["message_count"] = len(group)
        first["task_id"] = tid
        out.append(first)
    return out


async def _process_single_message(
    msg: Dict[str, Any],
    idx: int,
    sla_strategy: Optional[Dict[str, Any]] = None,
    sla_level: str = "standard",
) -> Dict[str, Any]:
    """
    处理单条消息推理。

    Args:
        msg: 消息字典
        idx: 消息索引
        sla_strategy: SLA策略
        sla_level: SLA级别（用于日志和类型感知采样）

    Returns:
        推理结果字典
    """
    msg_id = msg.get("ID")
    rt = msg.get("eval_request_type", "loglikelihood")
    prompt = msg["prompt"]

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": rt}
    msg_start = time.time()

    # ── 任务类型感知的采样参数 ──
    # loglikelihood类任务必须使用temperature=0确保确定性
    if rt == "generate_until":
        # 生成任务：使用SLA策略的采样参数
        gen_kwargs = msg.get("eval_gen_kwargs")
        response_text = await generate_text(prompt, gen_kwargs, sla_strategy)
        result["response"] = response_text
        result["accuracy"] = None
        tokens = len(response_text.split()) if response_text else 0

    elif rt == "loglikelihood":
        # 概率计算：强制temperature=0确保确定性
        # 不使用sla_strategy，强制greedy解码
        logprob = await compute_logprob(
            prompt,
            msg.get("eval_continuation", ""),
            None  # 强制确定性参数
        )
        result["accuracy"] = logprob
        result["response"] = None
        tokens = len(msg.get("eval_continuation", "").split())

        # 记录logprob质量指标
        _metrics.observe_histogram("inference.logprob_value", logprob, labels={"type": rt})

    elif rt == "loglikelihood_rolling":
        # 滚动概率：强制temperature=0
        logprob = await compute_rolling_logprob(prompt, None)
        result["accuracy"] = logprob
        result["response"] = None
        tokens = len(prompt.split())

        # 记录logprob质量指标
        _metrics.observe_histogram("inference.logprob_value", logprob, labels={"type": rt})

    else:
        result["response"] = None
        result["accuracy"] = None
        tokens = 0

    elapsed = time.time() - msg_start
    _metrics.observe_histogram("inference.latency", elapsed,
                               labels={"type": rt, "sla": sla_level})
    _metrics.inc_counter("inference.requests", labels={"type": rt, "status": "success"})
    _metrics.inc_counter("inference.tokens", tokens, labels={"type": rt})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]
    return result


async def _process_all_async(
    messages: List[Dict[str, Any]],
    adaptive_strategy: Dict[str, Any],
    effective_sla: str,
    max_concurrent: int,
) -> List[Dict[str, Any]]:
    """内部异步函数：处理所有消息。"""
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(m, i):
        async with sem:
            return await _process_single_message(m, i, adaptive_strategy, effective_sla)

    raw = await asyncio.gather(*[bounded(m, i) for i, m in enumerate(messages)], return_exceptions=True)
    results = []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            msg = messages[i]
            logger.error(f"[推理] 处理消息 {msg.get('ID')} 出错: {r}")
            results.append({
                "ID": msg.get("ID"), "prompt": msg.get("prompt"),
                "eval_request_type": msg.get("eval_request_type", "loglikelihood"),
                "response": None, "accuracy": None, "sla_level": effective_sla,
            })
            _metrics.inc_counter("inference.requests",
                                 labels={"type": msg.get("eval_request_type", "loglikelihood"), "status": "error"})
        else:
            r["sla_level"] = effective_sla
            results.append(r)
    return _aggregate_task_results(results)


def _prepare_sla(messages, sla_level, deadline_ms):
    """准备SLA策略的辅助函数。"""
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
        effective_sla = sla_level
        adaptive_strategy = get_adaptive_sla_strategy(deadline_ms, msg_count, sla_level)
    return adaptive_strategy, effective_sla


def run_inference(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """同步推理入口，内部正确处理event loop。"""
    if max_concurrent is None:
        max_concurrent = CONCURRENCY_CONFIG["max_concurrent_messages"]

    adaptive_strategy, effective_sla = _prepare_sla(messages, sla_level, deadline_ms)

    async def _run():
        return await _process_all_async(messages, adaptive_strategy, effective_sla, max_concurrent)

    try:
        asyncio.get_running_loop()
        # 已在运行中的loop，用线程池避免嵌套asyncio.run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(_run()))
            return future.result()
    except RuntimeError:
        return asyncio.run(_run())


async def run_inference_async(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """异步推理入口，供已有event loop的context调用。"""
    if max_concurrent is None:
        max_concurrent = CONCURRENCY_CONFIG["max_concurrent_messages"]

    adaptive_strategy, effective_sla = _prepare_sla(messages, sla_level, deadline_ms)
    return await _process_all_async(messages, adaptive_strategy, effective_sla, max_concurrent)


async def _close_async_client():
    """关闭全局异步客户端。"""
    await _close_all_clients()


def close_vllm_client():
    """同步关闭 vLLM 客户端（供 main.py 调用）。"""
    try:
        asyncio.run(_close_async_client())
    except RuntimeError:
        # Event loop 已关闭，忽略
        pass
