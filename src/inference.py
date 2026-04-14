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
from typing import Optional, Dict, Any, List
from functools import lru_cache
import tiktoken

from src.utils.metrics import metrics as _metrics
from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

_vllm_async_client: Optional[httpx.AsyncClient] = None

RETRY_CONFIG = {
    "max_retries": int(os.environ.get("VLLM_MAX_RETRIES", "3")),
    "backoff_factor": float(os.environ.get("VLLM_BACKOFF_FACTOR", "1.5")),
    "max_backoff": float(os.environ.get("VLLM_MAX_BACKOFF", "30.0")),
    "request_timeout": float(os.environ.get("VLLM_TIMEOUT", "120.0")),
}

CONCURRENCY_CONFIG = {
    "max_concurrent_messages": int(os.environ.get("MAX_CONCURRENT_MESSAGES", "20")),
    "warmup_enabled": os.environ.get("WARMUP_ENABLED", "true").lower() == "true",
    "warmup_requests": int(os.environ.get("WARMUP_REQUESTS", "5")),
}


@lru_cache(maxsize=4)
def _get_tiktoken_encoding(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


async def _get_vllm_async_client() -> httpx.AsyncClient:
    global _vllm_async_client
    if _vllm_async_client is None:
        _vllm_async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(RETRY_CONFIG["request_timeout"]),
            limits=httpx.Limits(
                max_keepalive_connections=int(os.environ.get("VLLM_MAX_KEEPALIVE", "20")),
                max_connections=int(os.environ.get("VLLM_MAX_CONNECTIONS", "50")),
            ),
        )
    return _vllm_async_client


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
    """异步调用 vLLM completions API,自动重试."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "best_of": best_of,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if stop:
        payload["stop"] = stop
    if logprobs is not None:
        payload["logprobs"] = logprobs
        payload["logprobs_per_token"] = True
    if echo:
        payload["echo"] = echo
    if max_model_len is not None:
        payload["max_model_len"] = max_model_len
    if beam_size > 1:
        payload["beam_size"] = beam_size

    client = await _get_vllm_async_client()
    for attempt in range(RETRY_CONFIG["max_retries"] + 1):
        try:
            resp = await client.post(f"{VLLM_URL}/v1/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt < RETRY_CONFIG["max_retries"]:
                await asyncio.sleep(
                    min(RETRY_CONFIG["backoff_factor"] ** attempt, RETRY_CONFIG["max_backoff"])
                    * (0.5 + random.random())
                )
            else:
                raise


def _apply_stop_strings(text: str, stop_list: List[str]) -> str:
    for s in stop_list or []:
        idx = text.find(s)
        if idx >= 0:
            return text[:idx]
    return text


SLA_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "express": {
        "max_model_len": 2048, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "max_gen_toks": 64, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 1,
        "n": 1,
    },
    "fast": {
        "max_model_len": 4096, "temperature": 0.1, "top_p": 0.95, "top_k": 20,
        "max_gen_toks": 128, "repetition_penalty": 1.05, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 5,
        "n": 1,
    },
    "standard": {
        "max_model_len": 8192, "temperature": 0.7, "top_p": 0.9, "top_k": 50,
        "max_gen_toks": 256, "repetition_penalty": 1.1, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "beam_size": 1, "logprobs_requested": 20,
        "n": 1,
    },
    "high_quality": {
        "max_model_len": 8192, "temperature": 0.8, "top_p": 0.95, "top_k": -1,
        "max_gen_toks": 512, "repetition_penalty": 1.2, "frequency_penalty": 0.1,
        "presence_penalty": 0.1, "beam_size": 2, "logprobs_requested": 100,
        "n": 4,
    },
}


def get_sla_strategy(sla_level: str) -> Dict[str, Any]:
    return SLA_STRATEGIES.get(sla_level, SLA_STRATEGIES["standard"])


def get_sla_from_deadline(deadline_ms: Optional[int]) -> str:
    if deadline_ms is None:
        return "standard"
    deadline_s = deadline_ms / 1000.0
    if deadline_s <= 1.0:
        return "express"
    elif deadline_s <= 5.0:
        return "fast"
    elif deadline_s <= 30.0:
        return "standard"
    return "high_quality"


# SLA 自适应推理配置
SAMPLING_PARAMS: Dict[str, Dict[str, float]] = {}
SLA_LEVELS: Dict[str, Dict[str, float]] = {}
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")


def load_config():
    global SAMPLING_PARAMS, SLA_LEVELS
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        if "sampling_params" in config:
            SAMPLING_PARAMS = config["sampling_params"]
        if "sla_levels" in config:
            SLA_LEVELS = config["sla_levels"]
    except Exception:
        pass


load_config()
_model_warmed_up = False


def warmup_model():
    """预热模型,构建 CUDA Graph. Q&A Q58."""
    global _model_warmed_up
    if _model_warmed_up or not CONCURRENCY_CONFIG["warmup_enabled"]:
        _model_warmed_up = True
        return

    logger.info("[推理] 开始预热模型...")
    prompts = ["Hello", "What is 2+2?", "The capital of France", "Once upon", "In the"]
    for i in range(CONCURRENCY_CONFIG["warmup_requests"]):
        try:
            asyncio.run(_call_vllm(prompt=prompts[i % len(prompts)], max_tokens=8,
                                   temperature=0.0, top_p=1.0, top_k=1))
        except Exception:
            pass
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
    try:
        resp = await _call_vllm(
            prompt=prompt, max_tokens=p["max_gen_toks"], temperature=p["temperature"],
            top_p=p["top_p"], top_k=p["top_k"], stop=p["until"],
            repetition_penalty=p["repetition_penalty"], frequency_penalty=p["frequency_penalty"],
            presence_penalty=p["presence_penalty"], max_model_len=p["max_model_len"],
            beam_size=p["beam_size"], best_of=p["best_of"],
        )
    except Exception as e:
        logger.error(f"[Inference] 文本生成失败: {e}")
        return ""

    choices = resp.get("choices", [])
    if not choices:
        return ""
    text = choices[0].get("text", "")
    return _apply_stop_strings(text, p["until"])


async def compute_logprob(
    prompt: str,
    continuation: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """计算 log P(continuation | prompt),单次 API 调用."""
    if not continuation:
        return 0.0

    try:
        enc = _get_tiktoken_encoding()
        pt = len(enc.encode(prompt))
        ct = len(enc.encode(continuation))
        lp_req = pt + ct
    except Exception:
        lp_req = 200  # fallback: 确保覆盖

    try:
        resp = await _call_vllm(
            prompt=prompt + continuation, max_tokens=1, logprobs=lp_req,
            echo=True, temperature=0.0, top_p=1.0, top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] logprob 调用失败: {e}")
        return -10.0

    lps = resp.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
    if not lps or len(lps) < 2:
        return -10.0

    if pt > 0 and ct > 0:
        cont = [lp for lp in lps[pt:pt + ct] if lp is not None]
    else:
        mid = len(lps) // 2
        cont = [lp for lp in lps[mid:] if lp is not None]
    return float(sum(cont)) if cont else -10.0


async def compute_rolling_logprob(
    text: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """计算整个文本的滚动 log-likelihood."""
    if not text:
        return 0.0

    try:
        enc = _get_tiktoken_encoding()
        num_tokens = len(enc.encode(text))
        lp_req = min(num_tokens + 10, 2000)
    except Exception:
        lp_req = 1000

    if sla_strategy:
        lp_req = max(lp_req, min(sla_strategy.get("logprobs_requested", 1000) * 10, 2000))

    try:
        resp = await _call_vllm(
            prompt=text, max_tokens=1, logprobs=lp_req,
            echo=True, temperature=0.0, top_p=1.0, top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] rolling logprob 调用失败: {e}")
        return -10.0

    lps = resp.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
    if not lps or len(lps) < 2:
        return -10.0
    valid = [lp for lp in lps[1:] if lp is not None]
    return float(sum(valid) / len(valid)) if valid else -10.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def _aggregate_task_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 task_id 聚合多 message 正确性分数 (Q63: 取平均)."""
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        tid = r.get("task_id") or r.get("ID") or 0
        groups.setdefault(tid, []).append(r)

    out = []
    for tid, group in groups.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        first = group[0].copy()
        accs = [r.get("accuracy") for r in group if r.get("accuracy") is not None]
        if accs:
            first["accuracy"] = sum(accs) / len(accs)
            first["accuracy_count"] = len(accs)
            first["accuracy_std"] = _std(accs) if len(accs) > 1 else 0.0
        for r in group:
            if r.get("response"):
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
) -> Dict[str, Any]:
    msg_id = msg.get("ID")
    rt = msg.get("eval_request_type", "loglikelihood")
    prompt = msg["prompt"]

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": rt}
    msg_start = time.time()

    try:
        if rt == "generate_until":
            response_text = await generate_text(prompt, msg.get("eval_gen_kwargs"), sla_strategy)
            result["response"] = response_text
            result["accuracy"] = None
            tokens = len(response_text.split()) if response_text else 0
        elif rt == "loglikelihood":
            logprob = await compute_logprob(prompt, msg.get("eval_continuation", ""), sla_strategy)
            result["accuracy"] = logprob
            result["response"] = None
            tokens = len(msg.get("eval_continuation", "").split())
        elif rt == "loglikelihood_rolling":
            logprob = await compute_rolling_logprob(prompt, sla_strategy)
            result["accuracy"] = logprob
            result["response"] = None
            tokens = len(prompt.split())
        else:
            result["response"] = None
            result["accuracy"] = None
            tokens = 0

        elapsed = time.time() - msg_start
        _metrics.observe_histogram("inference.latency", elapsed,
                                   labels={"type": rt, "sla": (sla_strategy or {}).get("sla_level", "none")})
        _metrics.inc_counter("inference.requests", labels={"type": rt, "status": "success"})
        _metrics.inc_counter("inference.tokens", tokens, labels={"type": rt})
    except Exception as e:
        logger.error(f"[Inference] 处理消息 {msg_id} 出错: {e}")
        result["response"] = None
        result["accuracy"] = None
        _metrics.inc_counter("inference.requests", labels={"type": rt, "status": "error"})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]
    return result


def run_inference(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if sla_level is None:
        sla_level = get_sla_from_deadline(deadline_ms)
    sla_strategy = get_sla_strategy(sla_level)
    if max_concurrent is None:
        max_concurrent = CONCURRENCY_CONFIG["max_concurrent_messages"]

    async def process_all():
        sem = asyncio.Semaphore(max_concurrent)

        async def bounded(m, i):
            async with sem:
                return await _process_single_message(m, i, sla_strategy)

        return await asyncio.gather(*[bounded(m, i) for i, m in enumerate(messages)], return_exceptions=True)

    raw = asyncio.run(process_all())
    results = []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            msg = messages[i]
            logger.error(f"[推理] 处理消息 {msg.get('ID')} 出错: {r}")
            results.append({
                "ID": msg.get("ID"), "prompt": msg.get("prompt"),
                "eval_request_type": msg.get("eval_request_type", "loglikelihood"),
                "response": None, "accuracy": None, "sla_level": sla_level,
            })
            _metrics.inc_counter("inference.requests",
                                 labels={"type": msg.get("eval_request_type", "loglikelihood"), "status": "error"})
        else:
            r["sla_level"] = sla_level
            results.append(r)

    return _aggregate_task_results(results)
