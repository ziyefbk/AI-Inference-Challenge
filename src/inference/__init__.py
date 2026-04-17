"""
推理引擎模块。

统一入口，导出推理相关功能。

子模块:
  vllm        - vLLM 调用封装
  strategy    - SLA 策略管理
  task_types  - 任务类型处理器
    generate_until      - 文本生成任务
    loglikelihood       - 条件概率任务
"""

import os
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional

from src.inference.vllm import close_all_clients
from src.inference.strategy import load_config
from src.inference.task_types import (
    process_generate_until,
    process_loglikelihood,
    process_loglikelihood_rolling,
    extract_final_answer,
)
from src.inference.strategy import (
    prepare_sla,
    get_adaptive_sla_strategy,
    get_sla_from_deadline,
    get_sla_strategy,
    load_config,
)
from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("inference")

# ── 并发配置 ─────────────────────────────────────────────────────────────

def get_concurrency_config() -> Dict[str, Any]:
    num_instances = int(os.environ.get("VLLM_NUM_INSTANCES", "1"))
    base = int(os.environ.get("MAX_CONCURRENT_MESSAGES", "20"))
    return {
        "max_concurrent_messages": base * num_instances,
        "warmup_enabled": os.environ.get("WARMUP_ENABLED", "true").lower() == "true",
        "warmup_requests": int(os.environ.get("WARMUP_REQUESTS", "5")),
    }


# ── 预热 ─────────────────────────────────────────────────────────────

_model_warmed_up = False


def warmup_model() -> None:
    global _model_warmed_up
    config = get_concurrency_config()
    if _model_warmed_up or not config["warmup_enabled"]:
        _model_warmed_up = True
        return

    logger.info("开始预热模型...")

    async def _do_warmup():
        from src.inference.vllm import chat_completions
        prompts = [
            ("Hello", "express"),
            ("What is 2+2?", "fast"),
            ("The capital of France is", "standard"),
        ]
        for prompt, sla in prompts:
            for _ in range(3):
                await chat_completions(prompt=prompt, max_tokens=8, temperature=0.0, top_p=1.0, top_k=1)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(lambda: asyncio.run(_do_warmup())).result()
    except RuntimeError:
        asyncio.run(_do_warmup())

    _model_warmed_up = True


# ── 指标 ─────────────────────────────────────────────────────────────

from src.utils.metrics import metrics
METRICS = metrics
METRICS.inc_counter("inference.init")


# ── 内部处理函数 ─────────────────────────────────────────────────────────

def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def validate_result(result: Dict[str, Any]) -> bool:
    """验证推理结果的有效性。"""
    rt = result.get("eval_request_type")

    if rt == "generate_until":
        response = result.get("response", "")
        if not response:
            return False
        if len(response) < 3:
            return False
        words = response.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False

    elif rt in ("loglikelihood", "loglikelihood_rolling"):
        accuracy = result.get("accuracy")
        if accuracy is None:
            return False
        # 注意: loglikelihood_rolling 现在返回平均 logprob，可能为正

    return True


def _aggregate_task_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 task_id 聚合多 message 结果。

    重要: 对于 loglikelihood / loglikelihood_rolling 类型，同一题的多条候选
    message 不做 accuracy 平均——每条消息的 logprob 独立存在，平台会自己做
    argmax。聚合只是把同 task_id 的消息归拢到同一个结果对象中。

    对于 generate_until 类型，多 message 的正确性分数会由平台取平均（见 Q63）。
    """
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        tid = r.get("task_id") or r.get("ID") or 0
        groups.setdefault(tid, []).append(r)

    out = []
    for tid, group in groups.items():
        if len(group) == 1:
            result = group[0]
            if not validate_result(result):
                result["_validation_failed"] = True
                METRICS.inc_counter("inference.validation.failed")
            out.append(result)
            continue

        # 同 task_id 有多条 message: 保留所有原始 accuracy，
        # 不做 logprob 平均（平台对 loglikelihood 自己做 argmax）
        first = group[0].copy()

        # 收集所有 accuracy 用于统计/监控（不出现在提交结果中）
        accs = [r.get("accuracy") for r in group if r.get("accuracy") is not None]

        valid_count = 0
        for r in group:
            if validate_result(r):
                valid_count += 1
            else:
                r["_validation_failed"] = True
                METRICS.inc_counter("inference.validation.failed")

        # 保留第一条消息的 accuracy 作为代表性值，
        # 并附加统计元数据供监控使用
        if accs:
            first["accuracy"] = group[0].get("accuracy")
            first["accuracy_count"] = len(accs)
            first["accuracy_std"] = _std(accs) if len(accs) > 1 else 0.0
            first["valid_count"] = valid_count

        # 收集所有 response（generate_until 可能有多个）
        responses = [r.get("response") for r in group if r.get("response")]
        if responses:
            first["response"] = responses[0]
        first["message_count"] = len(group)
        first["task_id"] = tid
        out.append(first)
    return out


async def _process_single_message(
    msg: Dict[str, Any],
    idx: int,
    sla_strategy: Optional[Dict[str, Any]],
    sla_level: str,
) -> Dict[str, Any]:
    """处理单条消息推理。"""
    msg_id = msg.get("ID")
    rt = msg.get("eval_request_type", "loglikelihood")
    prompt = msg["prompt"]

    result = {"ID": msg_id, "prompt": prompt, "eval_request_type": rt}

    if rt == "generate_until":
        result.update(await process_generate_until(msg, sla_strategy, sla_level))
    elif rt == "loglikelihood":
        result.update(await process_loglikelihood(msg, sla_strategy, sla_level))
    elif rt == "loglikelihood_rolling":
        result.update(await process_loglikelihood_rolling(msg, sla_strategy, sla_level))
    else:
        result["response"] = None
        result["accuracy"] = None

    result["sla_level"] = sla_level
    return result


async def _process_all_async(
    messages: List[Dict[str, Any]],
    adaptive_strategy: Dict[str, Any],
    effective_sla: str,
    max_concurrent: int,
) -> List[Dict[str, Any]]:
    """内部异步: 处理所有消息。"""
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(m, i):
        async with sem:
            return await _process_single_message(m, i, adaptive_strategy, effective_sla)

    raw = await asyncio.gather(
        *[bounded(m, i) for i, m in enumerate(messages)],
        return_exceptions=True
    )
    results = []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            msg = messages[i]
            logger.error(f"处理消息 {msg.get('ID')} 出错: {r}")
            results.append({
                "ID": msg.get("ID"), "prompt": msg.get("prompt"),
                "eval_request_type": msg.get("eval_request_type", "loglikelihood"),
                "response": None, "accuracy": None, "sla_level": effective_sla,
            })
            METRICS.inc_counter("inference.requests",
                              labels={"type": msg.get("eval_request_type", "loglikelihood"), "status": "error"})
        else:
            results.append(r)
    return _aggregate_task_results(results)


# ── 公共 API ─────────────────────────────────────────────────────────────

def run_inference(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """同步推理入口。"""
    if max_concurrent is None:
        max_concurrent = get_concurrency_config()["max_concurrent_messages"]

    adaptive_strategy, effective_sla = prepare_sla(messages, sla_level, deadline_ms)

    async def _run():
        return await _process_all_async(messages, adaptive_strategy, effective_sla, max_concurrent)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(lambda: asyncio.run(_run())).result()
    except RuntimeError:
        return asyncio.run(_run())


async def run_inference_async(
    messages: List[Dict[str, Any]],
    sla_level: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """异步推理入口。"""
    if max_concurrent is None:
        max_concurrent = get_concurrency_config()["max_concurrent_messages"]

    adaptive_strategy, effective_sla = prepare_sla(messages, sla_level, deadline_ms)
    return await _process_all_async(messages, adaptive_strategy, effective_sla, max_concurrent)


def close_vllm_client() -> None:
    """同步关闭 vLLM 客户端。"""
    try:
        asyncio.run(close_all_clients())
    except RuntimeError:
        pass
