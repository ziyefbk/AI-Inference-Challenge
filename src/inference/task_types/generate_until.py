"""
generate_until 任务处理器。

处理需要模型生成完整文本响应的任务。
"""

import os
import time
import re
from typing import Dict, Any, Optional, List

from src.inference.vllm import chat_completions, completions
from src.inference.strategy import get_sla_strategy, get_adaptive_sla_strategy
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger("inference.tasks.generate")


def extract_final_answer(text: str) -> Optional[str]:
    """
    从模型响应中提取最终答案。

    优先级:
    1. #### 分隔符后的内容 (GSM8K 标准格式)
    2. \\boxed{} 中的内容
    3. 最后一个等号后的数字
    4. 最后一个独立数字
    """
    if not text:
        return None

    if "####" in text:
        answer = text.split("####")[-1].strip()
        if answer:
            return answer

    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    eq_matches = re.findall(
        r'=\s*([-\d,.]+(?:\s*(?:dollars?|percent|°C|°F|kg|lb|hours?|minutes?|days?|years?|miles?|quarts?|pieces?|cards?|people?|boxes?|pairs?|quarters?|hectares?)?)?)',
        text, re.IGNORECASE
    )
    if eq_matches:
        return eq_matches[-1].strip()

    num_matches = re.findall(r'(?<![.\d])-?\d+(?:\.\d+)?(?![.\d])', text)
    if num_matches:
        return num_matches[-1]

    return None


def _resolve_gen_params(
    prompt: str,
    gen_kwargs: Optional[Dict[str, Any]],
    sla: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并生成参数。"""
    defaults = sla or {}
    return {
        "max_gen_toks": gen_kwargs.get("max_gen_toks", defaults.get("max_gen_toks", 256)) if gen_kwargs else defaults.get("max_gen_toks", 256),
        "temperature": gen_kwargs.get("temperature", defaults.get("temperature", 0.0)) if gen_kwargs else defaults.get("temperature", 0.0),
        "top_p": gen_kwargs.get("top_p", defaults.get("top_p", 1.0)) if gen_kwargs else defaults.get("top_p", 1.0),
        "top_k": gen_kwargs.get("top_k", defaults.get("top_k", 50)) if gen_kwargs else defaults.get("top_k", 50),
        "until": gen_kwargs.get("until", defaults.get("until", ["\n\n"])) if gen_kwargs else defaults.get("until", ["\n\n"]),
        "repetition_penalty": gen_kwargs.get("repetition_penalty", defaults.get("repetition_penalty", 1.0)) if gen_kwargs else defaults.get("repetition_penalty", 1.0),
        "frequency_penalty": gen_kwargs.get("frequency_penalty", defaults.get("frequency_penalty", 0.0)) if gen_kwargs else defaults.get("frequency_penalty", 0.0),
        "presence_penalty": gen_kwargs.get("presence_penalty", defaults.get("presence_penalty", 0.0)) if gen_kwargs else defaults.get("presence_penalty", 0.0),
        "max_model_len": defaults.get("max_model_len"),
        "prompt_tokens": len(prompt.split()),
        "beam_size": defaults.get("beam_size", 1),
        "best_of": defaults.get("n", 1),
    }


async def process_generate_until(
    msg: Dict[str, Any],
    sla_strategy: Optional[Dict[str, Any]],
    sla_level: str,
) -> Dict[str, Any]:
    """
    处理 generate_until 任务。

    使用 completions API 直接生成文本，避免 chat 格式干扰 stop 条件。
    """
    msg_id = msg.get("ID")
    prompt = msg["prompt"]
    gen_kwargs = msg.get("eval_gen_kwargs")

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": "generate_until"}
    msg_start = time.time()

    # generate_until 必须使用非零 temperature 避免无限循环
    if sla_strategy and sla_strategy.get("sampling_key") == "Deterministic":
        gen_strategy = {**sla_strategy, "temperature": 0.1}
    else:
        gen_strategy = sla_strategy

    p = _resolve_gen_params(prompt, gen_kwargs, gen_strategy)

    max_toks = p["max_gen_toks"]
    if p["max_model_len"] and p["max_model_len"] > p["prompt_tokens"]:
        max_toks = min(max_toks, p["max_model_len"] - p["prompt_tokens"])

    resp = await completions(
        prompt=prompt,
        max_tokens=max_toks,
        temperature=p["temperature"],
        top_p=p["top_p"],
        top_k=p["top_k"],
        stop=None,
        repetition_penalty=p["repetition_penalty"],
        frequency_penalty=p["frequency_penalty"],
        presence_penalty=p["presence_penalty"],
        best_of=p["best_of"],
        max_model_len=p.get("max_model_len"),
        logprobs=0,
    )

    choices = resp.get("choices", [])
    if not choices:
        text = ""
    else:
        text = choices[0].get("text", "")

    elapsed = time.time() - msg_start

    result["response"] = text
    result["accuracy"] = None
    result["extracted_answer"] = extract_final_answer(text)

    metrics.observe_histogram("inference.latency", elapsed, labels={"type": "generate_until", "sla": sla_level})
    metrics.inc_counter("inference.requests", labels={"type": "generate_until", "status": "success"})
    metrics.inc_counter("inference.tokens", len(text.split()), labels={"type": "generate_until"})

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]

    return result
