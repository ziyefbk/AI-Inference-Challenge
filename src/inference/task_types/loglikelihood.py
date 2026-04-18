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

    数学定义:
        L = sum_{i=1}^{n} log P(t_i^cont | prompt_tokens, t_1^cont, ..., t_{i-1}^cont)

    实现策略:
        1. 将 prompt + continuation 整体作为 prompt 传入 vLLM completions API
        2. 设置 max_tokens=0，不生成任何新 token
        3. 设置 echo=True，让 vLLM 返回完整输入序列（prompt + continuation）
           中每个 token 的 log P(token_i | token_1...token_{i-1})
        4. 定位 prompt token 边界，取 continuation 部分累加

    vLLM 返回的 token_logprobs 结构（echo=True, max_tokens=0）:
        lp_list[0] = null              (第一个 token 无前驱)
        lp_list[1..m] = logP(prompt tokens)
        lp_list[m+1..m+n] = logP(continuation tokens)
        其中 m = prompt token 数，n = continuation token 数
    """
    if not continuation:
        return 0.0

    full_text = prompt + continuation
    resp = await completions(
        prompt=full_text,
        max_tokens=0,
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=True,
    )

    choices = resp.get("choices", [{}])
    if not choices:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "no_choices"})
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    lp_list: List[Any] = logprobs_data.get("token_logprobs", [])

    if not lp_list:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "empty_logprobs"})
        return -10.0

    # 定位 prompt 的 token 边界（第一个非 null 的索引 = 实际 prompt token 数）
    prompt_token_count = None
    for i, lp in enumerate(lp_list):
        if lp is not None:
            prompt_token_count = i
            break

    if prompt_token_count is None:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "all_null_logprobs"})
        return -10.0

    # 累加 continuation tokens 的 logprob
    # vLLM tokenize 是增量式的：从 lp_list[prompt_token_count+1] 开始才是 continuation
    cont_logprobs: List[float] = []
    for i in range(prompt_token_count + 1, len(lp_list)):
        val = lp_list[i]
        if val is not None:
            cont_logprobs.append(float(val))

    if not cont_logprobs:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "no_valid_tokens"})
        return -10.0

    total_logprob = float(sum(cont_logprobs))
    # metrics.observe_histogram("inference.logprob_sum", total_logprob)
    # metrics.observe_histogram("inference.logprob_value", total_logprob, labels={"type": "loglikelihood"})

    return total_logprob


async def compute_rolling_logprob(text: str) -> float:
    """
    计算整个文本的总 log-likelihood。

    数学定义:
        L = sum_{i=1}^{N} log P(t_i | t_1, t_2, ..., t_{i-1})

    实现策略:
        将 text 整体作为 prompt，max_tokens=0（不生成），echo=True。
        vLLM 返回 text 中每个 token 的自回归 logprob：
            lp_list[0] = null
            lp_list[1..N] = logP(t_1), logP(t_2 | t_1), ..., logP(t_N | t_1..t_{N-1})
        累加 lp_list[1:] 即为所求。

        注意：旧实现用 max_tokens=1，lp_list[1:] 会多包含 1 个生成的 token，
        导致结果偏大。
    """
    if not text:
        return 0.0

    resp = await completions(
        prompt=text,
        max_tokens=0,
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=True,
    )

    choices = resp.get("choices", [{}])
    if not choices:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "no_choices"})
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    lp_list: List[Any] = logprobs_data.get("token_logprobs", [])

    if not lp_list:
        # metrics.inc_counter("inference.logprob_error", labels={"reason": "empty_logprobs"})
        return -10.0

    valid = [float(lp) for lp in lp_list if lp is not None]
    total = float(sum(valid)) if valid else -10.0

    if valid:
        # metrics.observe_histogram("inference.logprob_value", total, labels={"type": "loglikelihood_rolling"})
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

    # metrics.observe_histogram("inference.latency", elapsed, labels={"type": "loglikelihood", "sla": sla_level})
    # metrics.inc_counter("inference.requests", labels={"type": "loglikelihood", "status": "success"})
    # metrics.inc_counter("inference.tokens", len(continuation.split()), labels={"type": "loglikelihood"})

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

    # metrics.observe_histogram("inference.latency", elapsed, labels={"type": "loglikelihood_rolling", "sla": sla_level})
    # metrics.inc_counter("inference.requests", labels={"type": "loglikelihood_rolling", "status": "success"})
    # metrics.inc_counter("inference.tokens", len(prompt.split()), labels={"type": "loglikelihood_rolling"})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]

    return result
