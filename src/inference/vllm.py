"""
vLLM 调用封装模块。

提供与 vLLM 服务通信的异步接口:
- Completions API (/v1/completions): 用于 generate_until / loglikelihood / loglikelihood_rolling
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

# 用于在关闭时快速终止重试
_shutdown_interrupt = False


def interrupt_vllm_requests() -> None:
    """通知 vLLM 模块停止重试并快速失败。"""
    global _shutdown_interrupt
    _shutdown_interrupt = True


MODEL_PATH = os.environ["MODEL_PATH"]
# Qwen2.5-0.5B context window = 32768, reserve headroom for prompt tokens
MODEL_MAX_LEN = int(os.environ.get("MODEL_MAX_LEN", "1000"))

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


# ── 健康检查 ────────────────────────────────────────────────────────────

async def check_vllm_healthy(url: str, timeout: float = 2.0) -> bool:
    """检查指定 vLLM 实例是否可达。"""
    try:
        client = httpx.AsyncClient(timeout=timeout)
        try:
            resp = await client.get(f"{url}/v1/models")
            return resp.status_code == 200
        finally:
            await client.aclose()
    except Exception:
        return False


class VLLMUnavailableError(Exception):
    """vLLM 服务不可用（未启动或崩溃）。"""
    pass


def raise_vllm_unavailable(urls: List[str]) -> None:
    """抛出明确的 vLLM 不可用错误，列出所有尝试过的地址。"""
    raise VLLMUnavailableError(
        f"vLLM 服务不可用，所有实例均无法连接: {urls}。"
        f"请确认 vLLM 进程已启动且监听对应端口。启动命令参考: "
        f"python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8000"
    )


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
    调用 vLLM /v1/completions 端点实现 Chat Completions 语义。

    将 system prompt 拼入 prompt，响应格式适配为 chat completions 风格。
    用于 generate_until 任务。
    """
    SYSTEM_PREFIX = "You are a helpful assistant.\n\n"
    prefixed_prompt = SYSTEM_PREFIX + prompt

    raw = await completions(
        prompt=prefixed_prompt,
        max_tokens=max_tokens,
        logprobs=logprobs if logprobs is not None else 0,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_model_len=max_model_len,
        n=1,
    )

    choices = raw.get("choices", [])
    if choices:
        choice = choices[0]
        text = choice.get("text", "")
        choice["message"] = {"role": "assistant", "content": text}

    return raw


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
    n: int = 1,
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
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if logprobs > 0:
        payload["logprobs"] = logprobs
        payload["echo"] = echo
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
        payload["n"] = max(n, best_of)
    elif n > 1:
        payload["n"] = n

    # Clamp max_tokens to avoid vLLM 400: max_tokens exceeds max_model_len
    payload["max_tokens"] = min(payload["max_tokens"], MODEL_MAX_LEN)

    client = await _get_client(chosen_url)
    logger.debug("vllm_request", url=f"{chosen_url}/v1/completions", payload=payload)

    for attempt in range(RETRY_CONFIG["max_retries"] + 1):
        if _shutdown_interrupt:
            raise asyncio.CancelledError("shutdown in progress")
        try:
            resp = await client.post(f"{chosen_url}/v1/completions", json=payload)
            if resp.status_code == 400:
                body = resp.content.decode(errors="replace")
                logger.warning("vllm_400_detail", body=body[:500], payload_keys=list(payload.keys()))
                resp.raise_for_status()
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices", [])
            if not choices:
                if attempt < RETRY_CONFIG["max_retries"]:
                    logger.warning("completions_empty_choices_retry", attempt=attempt, prompt_len=len(prompt))
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
            return result
        except asyncio.CancelledError:
            raise
        except (httpx.TimeoutException, httpx.HTTPStatusError, OSError) as e:
            is_retryable = False
            body_hint = ""
            if isinstance(e, httpx.HTTPStatusError):
                sc = e.response.status_code
                is_retryable = sc >= 500 or sc == 429
                # Don't retry 400 - it means bad request, won't change
                if sc == 400:
                    body = resp.content.decode(errors="replace")
                    body_hint = f" | vLLM: {body[:300] if body else 'empty'}"
                    logger.warning("vllm_400_detail", body=body[:500] if body else "empty", payload_keys=list(payload.keys()))
            elif isinstance(e, httpx.TimeoutException):
                is_retryable = True
            elif isinstance(e, OSError):
                is_retryable = True

            if is_retryable and attempt < RETRY_CONFIG["max_retries"]:
                backoff = min(RETRY_CONFIG["backoff_factor"] ** attempt, RETRY_CONFIG["max_backoff"])
                err_msg = str(e) if str(e) else repr(e)
                if not err_msg or err_msg == "''":
                    err_msg = f"{type(e).__name__}"
                logger.warning(f"completions_retry{body_hint}", attempt=attempt + 1, error=err_msg)
                await asyncio.sleep(backoff)
                continue
            err_msg = str(e) if str(e) else repr(e)
            if not err_msg or err_msg == "''":
                err_msg = f"{type(e).__name__}"
            logger.error(f"completions_failed{body_hint}", error=err_msg)
            # OSError (ConnectError) 全部重试失败，检查 vLLM 是否真的未启动
            if isinstance(e, OSError) and attempt == RETRY_CONFIG["max_retries"]:
                if not await check_vllm_healthy(chosen_url):
                    raise VLLMUnavailableError(
                        f"vLLM 实例 {chosen_url} 不可达（连接被拒绝）。"
                        f"请确认 vLLM 进程已启动。启动命令: "
                        f"python -m vllm.entrypoints.openai.api_server "
                        f"--model {MODEL_PATH} --port {chosen_url.split(':')[-1]}"
                    ) from e
            raise
