"""
loglikelihood 任务处理器。

计算 log P(continuation | prompt)，用于选择题、是非题等判断任务。
"""

import time
from typing import Dict, Any, Optional, List

from src.inference.vllm import completions
from src.inference.strategy import get_sla_strategy, get_adaptive_sla_strategy
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger("inference.tasks.loglikelihood")


async def compute_logprob(
    prompt: str,
    continuation: str,
) -> float:
    """
    计算 log P(continuation | prompt)，单次 API 调用。

    使用 vLLM completions API 的 logprobs 功能，
    通过 echo=True 获取 prompt 中每个 token 的 logprob，
    定位到 continuation 部分后累加。
    """
    if not continuation:
        return 0.0

    resp = await completions(
        prompt=prompt,
        max_tokens=len(continuation.split()),
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=True,
    )

    choices = resp.get("choices", [{}])
    if not choices:
        metrics.inc_counter("inference.logprob_error", labels={"reason": "no_choices"})
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    lp_list: List[Any] = logprobs_data.get("token_logprobs", [])

    if not lp_list:
        metrics.inc_counter("inference.logprob_error", labels={"reason": "empty_logprobs"})
        return -10.0

    # 从 vLLM 返回的 token_logprobs 中提取 continuation 部分
    # 注意: lp_list[0] 通常为 null (第一个 token 没有前驱)
    # lp_list[1:] 才是实际的 token logprob
    # 我们需要用 vLLM tokenizer 的 token 数来定位 prompt 的边界
    prompt_len = None
    for i, lp in enumerate(lp_list):
        if lp is not None:
            prompt_len = i
            break

    if prompt_len is None:
        metrics.inc_counter("inference.logprob_error", labels={"reason": "all_null_logprobs"})
        return -10.0

    cont_logprobs: List[float] = []
    for i in range(prompt_len + 1, len(lp_list)):
        val = lp_list[i]
        if val is not None:
            cont_logprobs.append(float(val))

    if not cont_logprobs:
        metrics.inc_counter("inference.logprob_error", labels={"reason": "no_valid_tokens"})
        return -10.0

    total_logprob = float(sum(cont_logprobs))
    metrics.observe_histogram("inference.logprob_sum", total_logprob)
    metrics.observe_histogram("inference.logprob_value", total_logprob, labels={"type": "loglikelihood"})

    return total_logprob


async def compute_rolling_logprob(text: str) -> float:
    """
    计算整个文本的总 log-likelihood（perplexity）。

    使用 echo=True 让 vLLM 返回每个 token 的 logprob，
    对文本部分求和。
    """
    if not text:
        return 0.0

    resp = await completions(
        prompt=text,
        max_tokens=1,
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=True,
    )

    choices = resp.get("choices", [{}])
    if not choices:
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    lp_list: List[Any] = logprobs_data.get("token_logprobs", [])

    if not lp_list:
        return -10.0

    # lp_list[0] 为 null，从 [1:] 提取
    valid = [float(lp) for lp in lp_list[1:] if lp is not None]
    total = float(sum(valid)) if valid else -10.0

    # 返回总 logprob（平台按 Q10/Q11 要求总 logprob）
    if valid:
        metrics.observe_histogram("inference.logprob_value", total, labels={"type": "loglikelihood_rolling"})
        return total

    return -10.0


async def process_loglikelihood(
    msg: Dict[str, Any],
    sla_strategy: Optional[Dict[str, Any]],
    sla_level: str,
) -> Dict[str, Any]:
    """处理 loglikelihood 任务。"""
    msg_id = msg.get("ID")
    prompt = msg["prompt"]
    continuation = msg.get("eval_continuation", "")

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": "loglikelihood"}
    msg_start = time.time()

    logprob = await compute_logprob(prompt, continuation)
    elapsed = time.time() - msg_start

    result["accuracy"] = logprob
    result["response"] = None

    metrics.observe_histogram("inference.latency", elapsed, labels={"type": "loglikelihood", "sla": sla_level})
    metrics.inc_counter("inference.requests", labels={"type": "loglikelihood", "status": "success"})
    metrics.inc_counter("inference.tokens", len(continuation.split()), labels={"type": "loglikelihood"})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]

    return result


async def process_loglikelihood_rolling(
    msg: Dict[str, Any],
    sla_strategy: Optional[Dict[str, Any]],
    sla_level: str,
) -> Dict[str, Any]:
    """处理 loglikelihood_rolling 任务。"""
    msg_id = msg.get("ID")
    prompt = msg["prompt"]

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": "loglikelihood_rolling"}
    msg_start = time.time()

    logprob = await compute_rolling_logprob(prompt)
    elapsed = time.time() - msg_start

    result["accuracy"] = logprob
    result["response"] = None

    metrics.observe_histogram("inference.latency", elapsed, labels={"type": "loglikelihood_rolling", "sla": sla_level})
    metrics.inc_counter("inference.requests", labels={"type": "loglikelihood_rolling", "status": "success"})
    metrics.inc_counter("inference.tokens", len(prompt.split()), labels={"type": "loglikelihood_rolling"})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]

    return result
