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


@lru_cache(maxsize=4)
def _get_tiktoken_encoding(encoding_name: str = "cl100k_base"):
    """获取 tiktoken 编码器,带缓存避免重复创建。"""
    return tiktoken.get_encoding(encoding_name)

# 导入工具模块
from src.utils.metrics import metrics as _metrics_collector
from src.utils.logger import setup_logger, get_logger

# 初始化日志
setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference")


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

# vLLM URL 和模型名
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

# vLLM HTTP 客户端单例 (同步连接复用)
_vllm_client: Optional[httpx.Client] = None
# vLLM 异步客户端单例
_vllm_async_client: Optional[httpx.AsyncClient] = None

# 重试配置
RETRY_CONFIG = {
    "max_retries": int(os.environ.get("VLLM_MAX_RETRIES", "3")),
    "backoff_factor": float(os.environ.get("VLLM_BACKOFF_FACTOR", "1.5")),
    "max_backoff": float(os.environ.get("VLLM_MAX_BACKOFF", "30.0")),
}


def _get_vllm_client() -> httpx.Client:
    """获取或创建 vLLM HTTP 客户端单例。"""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = httpx.Client(
            timeout=float(os.environ.get("VLLM_TIMEOUT", "120.0")),
            limits=httpx.Limits(
                max_keepalive_connections=int(os.environ.get("VLLM_MAX_KEEPALIVE", "20")),
                max_connections=int(os.environ.get("VLLM_MAX_CONNECTIONS", "50")),
            ),
        )
    return _vllm_client


async def _get_vllm_async_client() -> httpx.AsyncClient:
    """获取或创建 vLLM 异步 HTTP 客户端单例。"""
    global _vllm_async_client
    if _vllm_async_client is None:
        _vllm_async_client = httpx.AsyncClient(
            timeout=float(os.environ.get("VLLM_TIMEOUT", "120.0")),
            limits=httpx.Limits(
                max_keepalive_connections=int(os.environ.get("VLLM_MAX_KEEPALIVE", "20")),
                max_connections=int(os.environ.get("VLLM_MAX_CONNECTIONS", "50")),
            ),
        )
    return _vllm_async_client


def close_vllm_client():
    """关闭单例客户端 (程序退出时调用)。"""
    global _vllm_client, _vllm_async_client
    if _vllm_client is not None:
        _vllm_client.close()
        _vllm_client = None
    if _vllm_async_client is not None:
        asyncio.get_event_loop().run_until_complete(_vllm_async_client.aclose())
        _vllm_async_client = None


async def _call_vllm_async(
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
    max_retries: int = None,
    backoff_factor: float = None,
) -> Dict[str, Any]:
    """
    异步调用 vLLM completions API,带重试机制。

    Args:
        max_retries: 最大重试次数
        backoff_factor: 退避因子
    """
    if max_retries is None:
        max_retries = RETRY_CONFIG["max_retries"]
    if backoff_factor is None:
        backoff_factor = RETRY_CONFIG["backoff_factor"]

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

    last_error = None
    client = await _get_vllm_async_client()

    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                f"{VLLM_URL}/v1/completions",
                json=payload,
                timeout=httpx.Timeout(RETRY_CONFIG.get("request_timeout", 120.0)),
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                wait_time = min(backoff_factor ** attempt, RETRY_CONFIG["max_backoff"])
                wait_time *= (0.5 + random.random())  # 添加 jitter
                await asyncio.sleep(wait_time)
                continue
            raise last_error

    raise last_error

# 采样参数预设 (从配置文件加载)
SAMPLING_PARAMS: Dict[str, Dict[str, float]] = {}
SLA_LEVELS: Dict[str, Dict[str, float]] = {}

# 配置文件路径
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")

# 可配置的并发度参数
CONCURRENCY_CONFIG = {
    "max_concurrent_messages": int(os.environ.get("MAX_CONCURRENT_MESSAGES", "20")),
    "max_pending_tasks": int(os.environ.get("MAX_PENDING_TASKS", "3")),
    "request_timeout": float(os.environ.get("REQUEST_TIMEOUT", "60.0")),
    "warmup_enabled": os.environ.get("WARMUP_ENABLED", "true").lower() == "true",
    "warmup_requests": int(os.environ.get("WARMUP_REQUESTS", "5")),
}

# CUDA Graph 预热标志
_model_warmed_up = False


def warmup_model():
    """
    预热模型,构建 CUDA Graph。
    通过发送一些 dummy 请求来触发模型的 JIT 编译和 CUDA Graph 捕获。
    这可以显著加速后续推理请求。

    Q&A Q58: setup.sh 可以构建 CUDA Graph。
    """
    global _model_warmed_up
    if _model_warmed_up:
        return

    if not CONCURRENCY_CONFIG["warmup_enabled"]:
        logger.info("[推理] 预热已禁用")
        _model_warmed_up = True
        return

    logger.info("[推理] 开始预热模型 (构建 CUDA Graph)...")
    warmup_count = CONCURRENCY_CONFIG["warmup_requests"]

    # 预热请求列表
    warmup_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "The capital of France is",
        "Once upon a time",
        "In the beginning",
    ]

    for i in range(warmup_count):
        prompt = warmup_prompts[i % len(warmup_prompts)]
        try:
            _call_vllm(
                prompt=prompt,
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            logger.info(f"[推理] 预热请求 {i+1}/{warmup_count} 完成")
        except Exception as e:
            logger.warning(f"[推理] 预热请求 {i+1} 失败: {e}")

    _model_warmed_up = True
    logger.info("[推理] 模型预热完成")

# SLA 自适应推理配置
# SLA 级别名称到推理策略的映射
SLA_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "express": {
        "max_model_len": 2048,          # 短上下文
        "temperature": 0.0,             # 确定性生成
        "top_p": 1.0,
        "top_k": 1,                    # 贪婪采样
        "max_gen_toks": 64,            # 最小生成量
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "beam_size": 1,                # 不使用 beam search
        "logprobs_requested": 1,       # 最小 logprobs
        "speculative": False,          # 禁用投机解码
        "n": 1,                        # 候选数
    },
    "fast": {
        "max_model_len": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 20,
        "max_gen_toks": 128,
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "beam_size": 1,
        "logprobs_requested": 5,
        "speculative": False,
        "n": 1,
    },
    "standard": {
        "max_model_len": 8192,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_gen_toks": 256,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "beam_size": 1,
        "logprobs_requested": 20,
        "speculative": False,
        "n": 1,
    },
    "high_quality": {
        "max_model_len": 8192,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,                  # 不限制 top-k
        "max_gen_toks": 512,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "beam_size": 2,                # 使用 beam search
        "logprobs_requested": 100,
        "speculative": True,           # 启用投机解码
        "n": 4,                        # 4 个候选
    },
}


def get_sla_strategy(sla_level: str) -> Dict[str, Any]:
    """
    根据 SLA 级别获取推理策略。
    如果级别不存在,回退到 'standard'。
    """
    return SLA_STRATEGIES.get(sla_level, SLA_STRATEGIES["standard"])


def get_sla_from_deadline(deadline_ms: Optional[int]) -> str:
    """
    根据截止时间(毫秒)推断 SLA 级别。
    返回对应的策略名称。
    """
    if deadline_ms is None:
        return "standard"

    # 转换为秒
    deadline_s = deadline_ms / 1000.0

    if deadline_s <= 1.0:
        return "express"
    elif deadline_s <= 5.0:
        return "fast"
    elif deadline_s <= 30.0:
        return "standard"
    else:
        return "high_quality"

# 性能指标类
class InferenceMetrics:
    """追踪推理性能指标。"""

    def __init__(self):
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.errors = 0
        self.by_type = {}  # 按 eval_request_type 追踪

    def record(self, eval_type: str, tokens: int, elapsed: float, success: bool):
        self.total_calls += 1
        self.total_tokens += tokens
        self.total_time += elapsed
        if not success:
            self.errors += 1

        if eval_type not in self.by_type:
            self.by_type[eval_type] = {"count": 0, "tokens": 0, "time": 0.0}
        self.by_type[eval_type]["count"] += 1
        self.by_type[eval_type]["tokens"] += tokens
        self.by_type[eval_type]["time"] += elapsed

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time, 2),
            "avg_tokens_per_sec": round(self.total_tokens / max(self.total_time, 0.01), 2),
            "error_rate": round(self.errors / max(self.total_calls, 1) * 100, 2),
            "by_type": self.by_type,
        }


# 使用新的 metrics 采集器
METRICS = _metrics_collector
METRICS.inc_counter("inference.init", labels={"model": MODEL_NAME})


def load_config():
    """加载竞赛配置文件(如果存在)。"""
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
    except Exception as e:
        logger.warning("config_load_failed", error=str(e))


load_config()


def _call_vllm(
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
) -> Dict[str, Any]:
    """
    调用 vLLM completions API。
    """
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

    client = _get_vllm_client()
    resp = client.post(f"{VLLM_URL}/v1/completions", json=payload)
    resp.raise_for_status()
    return resp.json()


def _apply_stop_strings(text: str, stop_list: List[str]) -> str:
    """应用停止字符串裁剪。"""
    if not stop_list or not text:
        return text
    for stop_tok in stop_list:
        idx = text.find(stop_tok)
        if idx >= 0:
            return text[:idx]
    return text


def compute_logprob(
    prompt: str,
    continuation: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算给定 prompt 条件下 continuation 的 log 概率。
    使用 vLLM 的 logprobs 特性,echo=True 获取完整文本的 logprobs,
    然后仅对 continuation 对应的 token 求和。

    可选地使用 SLA 自适应设置以加快计算速度。
    """
    if not continuation:
        return 0.0

    full_text = prompt + continuation
    logprobs_request = 100
    if sla_strategy is not None:
        logprobs_request = sla_strategy.get("logprobs_requested", 100)

    # 尝试获取精确的 token 数
    prompt_tokens = 0
    continuation_tokens = 0
    try:
        enc = _get_tiktoken_encoding()
        prompt_tokens = len(enc.encode(prompt))
        continuation_tokens = len(enc.encode(continuation))
        logprobs_request = prompt_tokens + continuation_tokens
    except Exception:
        # tiktoken 失败时，使用较大值确保覆盖
        logprobs_request = 200

    try:
        response = _call_vllm(
            prompt=full_text,
            max_tokens=1,
            logprobs=logprobs_request,
            echo=True,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] vLLM 调用失败 (logprob): {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0

    # 如果有精确 token 数，使用精确边界；否则用中间点
    if prompt_tokens > 0 and continuation_tokens > 0:
        start_idx = prompt_tokens
        end_idx = prompt_tokens + continuation_tokens
        cont_logprobs = token_logprobs[start_idx:end_idx]
    else:
        # 启发式：后半部分为 continuation
        n = len(token_logprobs)
        start_idx = n // 2
        cont_logprobs = token_logprobs[start_idx:]

    cont_logprobs = [lp for lp in cont_logprobs if lp is not None]
    if not cont_logprobs:
        return -10.0

    return float(sum(cont_logprobs))


def compute_rolling_logprob(
    text: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算整个文本的滚动 log-likelihood。
    使用 vLLM 的 logprobs 计算困惑度。
    可选地使用 SLA 自适应设置。
    """
    if not text:
        return 0.0

    # 根据 SLA 策略确定 logprobs 数量
    logprobs_request = 1000  # 默认值
    if sla_strategy is not None:
        logprobs_request = min(
            sla_strategy.get("logprobs_requested", 1000) * 10,
            2000
        )

    try:
        enc = _get_tiktoken_encoding()
        num_tokens = len(enc.encode(text))
        # 请求足够的 logprobs 以覆盖所有 token
        logprobs_request = max(logprobs_request, min(num_tokens + 10, 2000))
    except Exception:
        pass  # 使用 SLA 确定的值

    try:
        response = _call_vllm(
            prompt=text,
            max_tokens=1,
            logprobs=logprobs_request,
            echo=True,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] vLLM 调用失败 (rolling logprob): {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])

    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0

    # token_logprobs[0] 对应第一个 token (BOS),跳过它
    valid_logprobs = [lp for lp in token_logprobs[1:] if lp is not None]
    if not valid_logprobs:
        return -10.0

    return float(sum(valid_logprobs))


async def generate_text_async(
    prompt: str,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> str:
    """异步文本生成。"""
    if gen_kwargs is None:
        gen_kwargs = {}

    if sla_strategy is not None:
        max_tokens = gen_kwargs.get("max_gen_toks", sla_strategy.get("max_gen_toks", 256))
        temperature = gen_kwargs.get("temperature", sla_strategy.get("temperature", 0.0))
        top_p = gen_kwargs.get("top_p", sla_strategy.get("top_p", 1.0))
        top_k = gen_kwargs.get("top_k", sla_strategy.get("top_k", 50))
        until = gen_kwargs.get("until", ["\n\n"])
        repetition_penalty = gen_kwargs.get("repetition_penalty", sla_strategy.get("repetition_penalty", 1.0))
        frequency_penalty = gen_kwargs.get("frequency_penalty", sla_strategy.get("frequency_penalty", 0.0))
        presence_penalty = gen_kwargs.get("presence_penalty", sla_strategy.get("presence_penalty", 0.0))
        max_model_len = sla_strategy.get("max_model_len")
        beam_size = sla_strategy.get("beam_size", 1)
        best_of = sla_strategy.get("n", 1)
    else:
        max_tokens = gen_kwargs.get("max_gen_toks", 256)
        temperature = gen_kwargs.get("temperature", 0.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        top_k = gen_kwargs.get("top_k", 50)
        until = gen_kwargs.get("until", ["\n\n"])
        repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
        frequency_penalty = gen_kwargs.get("frequency_penalty", 0.0)
        presence_penalty = gen_kwargs.get("presence_penalty", 0.0)
        max_model_len = None
        beam_size = 1
        best_of = 1

    try:
        response = await _call_vllm_async(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=until,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_model_len=max_model_len,
            beam_size=beam_size,
            best_of=best_of,
        )
    except Exception as e:
        logger.error(f"[Inference] 异步文本生成失败: {e}")
        return ""

    choices = response.get("choices", [])
    if not choices:
        return ""

    text = choices[0].get("text", "")
    text = _apply_stop_strings(text, until)
    return text


async def compute_logprob_async(
    prompt: str,
    continuation: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """异步计算 log P(continuation | prompt)，优化为单次 API 调用。"""
    if not continuation:
        return 0.0

    full_text = prompt + continuation
    logprobs_request = 100
    if sla_strategy is not None:
        logprobs_request = sla_strategy.get("logprobs_requested", 100)

    prompt_tokens = 0
    continuation_tokens = 0
    try:
        enc = _get_tiktoken_encoding()
        prompt_tokens = len(enc.encode(prompt))
        continuation_tokens = len(enc.encode(continuation))
        logprobs_request = prompt_tokens + continuation_tokens
    except Exception:
        logprobs_request = 200

    try:
        response = await _call_vllm_async(
            prompt=full_text, max_tokens=1, logprobs=logprobs_request,
            echo=True, temperature=0.0, top_p=1.0, top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] 异步 logprob 调用失败: {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0
    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0

    if prompt_tokens > 0 and continuation_tokens > 0:
        start_idx = prompt_tokens
        cont_logprobs = [lp for lp in token_logprobs[start_idx:start_idx + continuation_tokens] if lp is not None]
    else:
        n = len(token_logprobs)
        cont_logprobs = [lp for lp in token_logprobs[n // 2:] if lp is not None]
    return float(sum(cont_logprobs)) if cont_logprobs else -10.0


async def compute_rolling_logprob_async(
    text: str,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> float:
    """异步计算整个文本的滚动 log-likelihood。"""
    if not text:
        return 0.0

    logprobs_request = 1000
    if sla_strategy is not None:
        logprobs_request = min(sla_strategy.get("logprobs_requested", 1000) * 10, 2000)

    try:
        enc = _get_tiktoken_encoding()
        num_tokens = len(enc.encode(text))
        logprobs_request = max(logprobs_request, min(num_tokens + 10, 2000))
    except Exception:
        pass

    try:
        response = await _call_vllm_async(
            prompt=text, max_tokens=1, logprobs=logprobs_request,
            echo=True, temperature=0.0, top_p=1.0, top_k=1,
        )
    except Exception as e:
        logger.error(f"[Inference] 异步 rolling logprob 调用失败: {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0
    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0
    valid_logprobs = [lp for lp in token_logprobs[1:] if lp is not None]
    return float(sum(valid_logprobs) / len(valid_logprobs)) if valid_logprobs else -10.0


def generate_text(
    prompt: str,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> str:
    """
    按停止条件生成文本。
    对生成结果应用停止字符串裁剪。
    可选地使用 SLA 自适应策略以加快推理速度。

    Q&A Q58: 支持 CUDA Graph 预热
    """
    # 优先使用提供的 gen_kwargs,然后用 SLA 策略覆盖
    if gen_kwargs is None:
        gen_kwargs = {}

    # 如果提供了 SLA 策略,用它作为基础(但尊重显式 gen_kwargs)
    if sla_strategy is not None:
        max_tokens = gen_kwargs.get("max_gen_toks", sla_strategy.get("max_gen_toks", 256))
        temperature = gen_kwargs.get("temperature", sla_strategy.get("temperature", 0.0))
        top_p = gen_kwargs.get("top_p", sla_strategy.get("top_p", 1.0))
        top_k = gen_kwargs.get("top_k", sla_strategy.get("top_k", 50))
        until = gen_kwargs.get("until", ["\n\n"])
        repetition_penalty = gen_kwargs.get("repetition_penalty", sla_strategy.get("repetition_penalty", 1.0))
        frequency_penalty = gen_kwargs.get("frequency_penalty", sla_strategy.get("frequency_penalty", 0.0))
        presence_penalty = gen_kwargs.get("presence_penalty", sla_strategy.get("presence_penalty", 0.0))
        max_model_len = sla_strategy.get("max_model_len")
        # 使用 SLA 策略中的 beam_size 和 n
        beam_size = sla_strategy.get("beam_size", 1)
        n = sla_strategy.get("n", 1)
        # speculative (投机解码) 目前仅支持通过 best_of 模拟
        best_of = sla_strategy.get("n", 1)
    else:
        max_tokens = gen_kwargs.get("max_gen_toks", 256)
        temperature = gen_kwargs.get("temperature", 0.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        top_k = gen_kwargs.get("top_k", 50)
        until = gen_kwargs.get("until", ["\n\n"])
        repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
        frequency_penalty = gen_kwargs.get("frequency_penalty", 0.0)
        presence_penalty = gen_kwargs.get("presence_penalty", 0.0)
        max_model_len = None
        beam_size = 1
        n = 1
        best_of = 1

    try:
        response = _call_vllm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=until,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_model_len=max_model_len,
            beam_size=beam_size,
            best_of=best_of,
        )
    except Exception as e:
        logger.error(f"[Inference] 文本生成失败: {e}")
        return ""

    choices = response.get("choices", [])
    if not choices:
        return ""

    text = choices[0].get("text", "")
    # 作为二次保险应用停止字符串裁剪
    text = _apply_stop_strings(text, until)
    return text


def run_inference(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    并发处理消息列表,支持 SLA 自适应推理。

    参数:
        messages: 消息字典列表,包含 'prompt', 'eval_request_type' 等字段
        sla_level: 显式 SLA 级别 ('express', 'fast', 'standard', 'high_quality')
        deadline_ms: 截止时间(毫秒),如果未提供 sla_level 则自动推断
        max_concurrent: 最大并发数 (默认从环境变量 MAX_CONCURRENT_MESSAGES 读取)
    """
    # 确定 SLA 策略
    if sla_level is None:
        sla_level = get_sla_from_deadline(deadline_ms)

    sla_strategy = get_sla_strategy(sla_level)

    # 确定并发度
    if max_concurrent is None:
        max_concurrent = CONCURRENCY_CONFIG["max_concurrent_messages"]

    logger.info(f"[推理] 使用 SLA '{sla_level}' 处理 {len(messages)} 条消息 (并发度: {max_concurrent})")

    results = []
    start_time = time.time()

    # 并发运行所有消息,使用信号量限制并发数
    async def process_all():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_process(msg, idx):
            async with semaphore:
                return await _process_single_message(msg, idx, sla_strategy)

        tasks = [
            bounded_process(msg, i)
            for i, msg in enumerate(messages)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # 运行异步处理
    raw_results = asyncio.run(process_all())

    # 收集结果并更新指标
    for i, result_or_exc in enumerate(raw_results):
        if isinstance(result_or_exc, Exception):
            msg = messages[i]
            logger.error(f"[推理] 处理消息 {msg.get('ID')} 出错: {result_or_exc}")
            results.append({
                "ID": msg.get("ID"),
                "prompt": msg.get("prompt"),
                "eval_request_type": msg.get("eval_request_type", "loglikelihood"),
                "response": None,
                "accuracy": None,
                "sla_level": sla_level,
            })
            METRICS.inc_counter(
                "inference.requests",
                labels={"type": msg.get("eval_request_type", "loglikelihood"), "status": "error"}
            )
        else:
            result_or_exc["sla_level"] = sla_level
            results.append(result_or_exc)

    total_elapsed = time.time() - start_time
    stats = METRICS.get_stats()
    # 获取延迟分位数
    latency_stats = METRICS.get_histogram_stats("inference.latency")
    logger.info(
        f"[推理] 完成 {len(messages)} 条消息,耗时 {total_elapsed:.2f}s (SLA: {sla_level}). "
        f"P50={latency_stats.get('p50', 0)*1000:.0f}ms, "
        f"P95={latency_stats.get('p95', 0)*1000:.0f}ms, "
        f"P99={latency_stats.get('p99', 0)*1000:.0f}ms"
    )

    # 按任务 ID 聚合多个 message 的正确性分数 (Q63: 取平均)
    results = _aggregate_task_results(results)

    return results


def _aggregate_task_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按任务 ID 聚合多个 message 的正确性分数。
    Q&A Q63: 多个请求的正确性分数会取平均。

    聚合策略:
    1. 按 task_id 或 ID 分组
    2. 对同组内的 accuracy 取平均
    3. 对 generate_until 保留 response
    """
    # 按 task_id 分组
    task_groups: Dict[int, List[Dict[str, Any]]] = {}

    for result in results:
        # 使用 task_id 或 ID 来分组
        task_id = result.get("task_id") or result.get("ID") or 0
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(result)

    aggregated = []

    for task_id, group in task_groups.items():
        if len(group) == 1:
            # 只有一个 message,直接使用
            aggregated.append(group[0])
        else:
            # 多个 message,聚合正确性分数
            first = group[0].copy()

            # 收集所有非 None 的 accuracy
            accuracies = [r.get("accuracy") for r in group if r.get("accuracy") is not None]

            if accuracies:
                # 取平均 (Q63)
                avg_accuracy = sum(accuracies) / len(accuracies)
                first["accuracy"] = avg_accuracy
                first["accuracy_count"] = len(accuracies)
                first["accuracy_std"] = _std(accuracies) if len(accuracies) > 1 else 0.0

            # 保留第一个 response (generate_until)
            for r in group:
                if r.get("response"):
                    first["response"] = r["response"]
                    break

            # 记录聚合信息
            first["message_count"] = len(group)
            first["task_id"] = task_id

            aggregated.append(first)

    return aggregated


def _std(values: List[float]) -> float:
    """计算标准差。"""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


async def _process_single_message(
    msg: Dict[str, Any],
    idx: int,
    sla_strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """使用 SLA 策略异步处理单条消息。"""
    msg_id = msg.get("ID")
    rt = msg.get("eval_request_type", "loglikelihood")
    prompt = msg["prompt"]

    result = {
        "ID": msg_id,
        "prompt": prompt,
        "eval_request_type": rt,
    }

    msg_start = time.time()
    try:
        if rt == "generate_until":
            gen_kwargs = msg.get("eval_gen_kwargs", {})
            response_text = await generate_text_async(prompt, gen_kwargs, sla_strategy)
            result["response"] = response_text
            result["accuracy"] = None
            tokens = len(response_text.split()) if response_text else 0

        elif rt == "loglikelihood":
            continuation = msg.get("eval_continuation", "")
            logprob = await compute_logprob_async(prompt, continuation, sla_strategy)
            result["accuracy"] = logprob
            result["response"] = None
            tokens = len(continuation.split()) if continuation else 0

        elif rt == "loglikelihood_rolling":
            logprob = await compute_rolling_logprob_async(prompt, sla_strategy)
            result["accuracy"] = logprob
            result["response"] = None
            tokens = len(prompt.split())

        else:
            result["response"] = None
            result["accuracy"] = None
            tokens = 0
            logger.warning(f"[Inference] 未知 eval_request_type: {rt}")

        msg_elapsed = time.time() - msg_start
        METRICS.observe_histogram(
            "inference.latency",
            msg_elapsed,
            labels={"type": rt, "sla": sla_strategy.get("sla_level", "unknown") if sla_strategy else "none"}
        )
        METRICS.inc_counter("inference.requests", labels={"type": rt, "status": "success"})
        METRICS.inc_counter("inference.tokens", tokens, labels={"type": rt})
        logger.debug(f"[Inference] 消息 {idx+1} ({rt}) 完成,耗时 {msg_elapsed:.2f}s")

    except Exception as e:
        logger.error(f"[Inference] 处理消息 {msg_id} 出错: {e}")
        result["response"] = None
        result["accuracy"] = None
        METRICS.inc_counter("inference.requests", labels={"type": rt, "status": "error"})

    # 保留必要字段
    if "eval_req_id" in msg:
        result["eval_req_id"] = msg["eval_req_id"]
    if "eval_gen_kwargs" in msg:
        result["eval_gen_kwargs"] = msg["eval_gen_kwargs"]
    if "eval_continuation" in msg:
        result["eval_continuation"] = msg["eval_continuation"]

    return result


# 快速测试
if __name__ == "__main__":
    # 测试基本生成功能
        logger.info("[Inference] 测试基本文本生成...")
    try:
        result = generate_text("Hello, how are you?", {"max_gen_toks": 20})
        logger.info(f"[Inference] 生成结果: {result[:100]}")
    except Exception as e:
        logger.warning(f"[Inference] 测试失败 (vLLM 未运行属正常): {e}")
