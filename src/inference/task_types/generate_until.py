"""
generate_until 任务处理器。

处理需要模型生成完整文本响应的任务。
"""

from typing import Dict, Any, Optional

from src.inference.vllm import completions


async def process_generate_until(
    msg: Dict[str, Any],
    sla_strategy: Optional[Dict[str, Any]],
    sla_level: str,
) -> Dict[str, Any]:
    """处理 generate_until 任务。"""
    msg_id = msg.get("ID")
    prompt = msg["prompt"]
    gen_kwargs = msg.get("eval_gen_kwargs")

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": "generate_until"}

    defaults = sla_strategy or {}

    temperature = (
        gen_kwargs.get("temperature") if gen_kwargs and "temperature" in gen_kwargs
        else defaults.get("temperature", 0.0)
    )
    max_gen_toks = (
        gen_kwargs.get("max_gen_toks") if gen_kwargs and "max_gen_toks" in gen_kwargs
        else defaults.get("max_gen_toks", 256)
    )
    top_p = (
        gen_kwargs.get("top_p") if gen_kwargs and "top_p" in gen_kwargs
        else defaults.get("top_p", 1.0)
    )
    top_k = (
        gen_kwargs.get("top_k") if gen_kwargs and "top_k" in gen_kwargs
        else defaults.get("top_k", 50)
    )
    until = (
        gen_kwargs.get("until") if gen_kwargs and "until" in gen_kwargs
        else defaults.get("until", ["\n\n"])
    )
    repetition_penalty = (
        gen_kwargs.get("repetition_penalty") if gen_kwargs and "repetition_penalty" in gen_kwargs
        else defaults.get("repetition_penalty", 1.0)
    )
    frequency_penalty = (
        gen_kwargs.get("frequency_penalty") if gen_kwargs and "frequency_penalty" in gen_kwargs
        else defaults.get("frequency_penalty", 0.0)
    )
    presence_penalty = (
        gen_kwargs.get("presence_penalty") if gen_kwargs and "presence_penalty" in gen_kwargs
        else defaults.get("presence_penalty", 0.0)
    )
    max_model_len = defaults.get("max_model_len")

    max_gen_toks = min(max_gen_toks, 1024)
    if max_model_len and max_model_len > 10:
        max_gen_toks = min(max_gen_toks, max_model_len - 1)

    resp = await completions(
        prompt=prompt,
        max_tokens=max_gen_toks,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=until,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=0,
    )

    choices = resp.get("choices", [])
    text = choices[0].get("text", "") if choices else ""

    result["response"] = text
    result["accuracy"] = None

    for k in ("eval_req_id", "eval_gen_kwargs", "eval_continuation"):
        if k in msg:
            result[k] = msg[k]

    return result
