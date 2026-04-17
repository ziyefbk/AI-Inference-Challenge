"""
vLLM 调用封装模块。

提供与 vLLM 服务通信的异步接口:
- Chat Completions API (generate_text)
- Completions API (compute_logprob, compute_rolling_logprob)
- 多实例负载均衡
"""

import os
import asyncio
import threading
from typing import Dict, Any, List, Optional

import httpx

from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference.vllm")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

# 重试配置
RETRY_CONFIG = {
    "max_retries": int(os.environ.get("VLLM_MAX_RETRIES", "3")),
    "backoff_factor": float(os.environ.get("VLLM_BACKOFF_FACTOR", "1.5")),
    "max_backoff": float(os.environ.get("VLLM_MAX_BACKOFF", "30.0")),
    "request_timeout": float(os.environ.get("VLLM_TIMEOUT", "120.0")),
}


# ── vLLM 实例管理 ─────────────────────────────────────────────────────────

def _build_vllm_urls() -> List[str]:
    if "VLLM_URLS" in os.environ:
        return [u.strip() for u in os.environ["VLLM_URLS"].split(",") if u.strip()]
    n = int(os.environ.get("VLLM_NUM_INSTANCES", "1"))
    return [f"http://localhost:{8000 + i}" for i in range(n)]


def get_vllm_urls() -> List[str]:
    return _build_vllm_urls()


# ── 客户端池 ─────────────────────────────────────────────────────────────

_vllm_async_clients: Dict[str, httpx.AsyncClient] = {}
_client_lock = threading.Lock()
_url_index = 0


def _get_next_url() -> str:
    global _url_index
    urls = get_vllm_urls()
    return urls[_url_index % len(urls)]


async def _get_client(url: str) -> httpx.AsyncClient:
    with _client_lock:
        if url not in _vllm_async_clients:
            _vllm_async_clients[url] = httpx.AsyncClient(
                timeout=httpx.Timeout(RETRY_CONFIG["request_timeout"]),
                limits=httpx.Limits(
                    max_keepalive_connections=int(os.environ.get("VLLM_MAX_KEEPALIVE", "100")),
                    max_connections=int(os.environ.get("VLLM_MAX_CONNECTIONS", "200")),
                ),
            )
        return _vllm_async_clients[url]


async def close_all_clients() -> None:
    global _vllm_async_clients
    with _client_lock:
        for url, client in _vllm_async_clients.items():
            await client.aclose()
        _vllm_async_clients.clear()


# ── Chat API (generate_until 用) ────────────────────────────────────────────

async def chat_completions(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    max_model_len: Optional[int] = None,
    beam_size: int = 1,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    调用 vLLM Chat Completions API。

    用于 generate_until 任务（需要 chat 格式）。
    """
    global _url_index
    urls = get_vllm_urls()
    chosen_url = urls[_url_index % len(urls)]
    _url_index += 1

    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if stop is not None:
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

    client = await _get_client(chosen_url)
    for attempt in range(RETRY_CONFIG["max_retries"] + 1):
        try:
            resp = await client.post(f"{chosen_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and attempt < RETRY_CONFIG["max_retries"]:
                logger.warning("chat_400_retry", attempt=attempt, text=e.response.text[:200])
                await asyncio.sleep(0.5)
                continue
            raise


# ── Completions API (loglikelihood 用) ──────────────────────────────────────

async def completions(
    prompt: str,
    max_tokens: int = 1,
    logprobs: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    echo: bool = False,
    stop: Optional[List[str]] = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    best_of: int = 1,
    max_model_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    调用 vLLM Completions API（无 chat 格式）。

    用于:
    - loglikelihood: 计算 P(continuation | prompt)
    - generate_until: 直接生成文本
    - loglikelihood_rolling: 计算整文档 perplexity
    """
    global _url_index
    urls = get_vllm_urls()
    chosen_url = urls[_url_index % len(urls)]
    _url_index += 1

    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": logprobs,
        "echo": echo,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if stop is not None:
        payload["stop"] = stop
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty
    if frequency_penalty != 0.0:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty != 0.0:
        payload["presence_penalty"] = presence_penalty
    if best_of > 1:
        payload["best_of"] = best_of
    if max_model_len is not None:
        payload["max_model_len"] = max_model_len

    client = await _get_client(chosen_url)
    for attempt in range(RETRY_CONFIG["max_retries"] + 1):
        try:
            resp = await client.post(f"{chosen_url}/v1/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and attempt < RETRY_CONFIG["max_retries"]:
                logger.warning("completions_400_retry", attempt=attempt, text=e.response.text[:200])
                await asyncio.sleep(0.5)
                continue
            raise
