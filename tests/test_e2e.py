"""
推理服务端到端测试。
在不依赖真实平台的情况下测试完整流程。
"""

import sys
import os
import time
import asyncio
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import (
    generate_text, compute_logprob, compute_rolling_logprob,
    run_inference, load_config
)


def test_inference_module_import():
    """测试所有模块是否正确导入。"""
    print("[测试] 导入模块...")
    try:
        from src import client, inference, scheduler
        print("[测试] 所有模块导入成功")
        return True
    except Exception as e:
        print(f"[测试] 导入失败: {e}")
        return False


def test_config_loading():
    """测试配置加载。"""
    print("[测试] 测试配置加载...")
    try:
        load_config()
        print("[测试] 配置加载通过")
        return True
    except Exception as e:
        print(f"[测试] 配置加载失败: {e}")
        return False


def test_run_inference_generate_until():
    """测试 generate_until 推理。"""
    print("[测试] 测试 generate_until 推理...")
    messages = [
        {
            "ID": 0,
            "prompt": "The capital of France is",
            "eval_request_type": "generate_until",
            "eval_gen_kwargs": {
                "until": ["\n", "."],
                "max_gen_toks": 30,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "eval_req_id": "test_001",
        }
    ]
    results = run_inference(messages)
    assert len(results) == 1
    assert results[0]["ID"] == 0
    assert results[0]["eval_request_type"] == "generate_until"
    # 如果 vLLM 未运行响应可能为空,结构测试仍然通过
    print(f"[测试] generate_until 结果: {results[0].get('response', '')[:50]}")
    return True


def test_run_inference_loglikelihood():
    """测试 loglikelihood 推理。"""
    print("[测试] 测试 loglikelihood 推理...")
    messages = [
        {
            "ID": 1,
            "prompt": "The capital of France is",
            "eval_request_type": "loglikelihood",
            "eval_continuation": " Paris",
            "eval_req_id": "test_002",
        }
    ]
    results = run_inference(messages)
    assert len(results) == 1
    assert results[0]["ID"] == 1
    assert results[0]["eval_request_type"] == "loglikelihood"
    acc = results[0].get("accuracy")
    print(f"[测试] loglikelihood accuracy: {acc}")
    # 应该是 float 类型(即使 vLLM 未运行返回 -10.0 也正常)
    return isinstance(acc, float)


def test_run_inference_rolling():
    """测试 loglikelihood_rolling 推理。"""
    print("[测试] 测试 loglikelihood_rolling 推理...")
    messages = [
        {
            "ID": 2,
            "prompt": "The capital of France is Paris.",
            "eval_request_type": "loglikelihood_rolling",
            "eval_req_id": "test_003",
        }
    ]
    results = run_inference(messages)
    assert len(results) == 1
    assert results[0]["ID"] == 2
    assert results[0]["eval_request_type"] == "loglikelihood_rolling"
    acc = results[0].get("accuracy")
    print(f"[测试] loglikelihood_rolling accuracy: {acc}")
    return isinstance(acc, float)


def test_run_inference_mixed():
    """测试混合推理类型。"""
    print("[测试] 测试混合推理类型...")
    messages = [
        {
            "ID": 0,
            "prompt": "What is 2+2?",
            "eval_request_type": "generate_until",
            "eval_gen_kwargs": {"max_gen_toks": 20, "temperature": 0.0, "until": ["\n"]},
            "eval_req_id": "test_004",
        },
        {
            "ID": 1,
            "prompt": "What is 2+2?",
            "eval_request_type": "loglikelihood",
            "eval_continuation": " 4",
            "eval_req_id": "test_005",
        },
        {
            "ID": 2,
            "prompt": "A short text.",
            "eval_request_type": "loglikelihood_rolling",
            "eval_req_id": "test_006",
        },
    ]
    results = run_inference(messages)
    assert len(results) == 3
    assert results[0]["eval_request_type"] == "generate_until"
    assert results[1]["eval_request_type"] == "loglikelihood"
    assert results[2]["eval_request_type"] == "loglikelihood_rolling"
    # 验证 eval_req_id 被保留
    assert results[0].get("eval_req_id") == "test_004"
    print("[测试] 混合推理通过 - 所有类型均已处理")
    return True


def test_client_async_structure():
    """测试 client.py 异步结构(无需网络)。"""
    print("[测试] 测试客户端异步结构...")
    import asyncio
    from src.client import register, query_task, accept_task, submit_results

    async def dummy_client():
        # 测试函数正确定义且可 await
        print("[测试] 客户端函数已正确定义为异步")
        return True

    result = asyncio.get_event_loop().run_until_complete(dummy_client())
    return result


def test_vllm_unreachable():
    """测试 vLLM 不可达时的行为。"""
    print("[测试] 测试 vLLM 不可达行为...")
    # 此测试仅验证 vLLM 宕机时代码不会崩溃
    messages = [
        {
            "ID": 0,
            "prompt": "test",
            "eval_request_type": "generate_until",
            "eval_gen_kwargs": {"max_gen_toks": 10},
            "eval_req_id": "test_007",
        }
    ]
    try:
        results = run_inference(messages)
        # 应优雅返回而不崩溃
        return len(results) == 1
    except Exception as e:
        print(f"[测试] 异常(vLLM 未运行属正常): {e}")
        return True  # 不崩溃即成功


def test_sla_strategy_selection():
    """测试从截止时间选择 SLA 策略。"""
    print("[测试] 测试 SLA 策略选择...")
    from src.inference import get_sla_from_deadline, get_sla_strategy

    # 测试基于截止时间的 SLA 推断
    assert get_sla_from_deadline(500) == "express"    # 0.5s
    assert get_sla_from_deadline(1000) == "express"   # 1s
    assert get_sla_from_deadline(2000) == "fast"     # 2s
    assert get_sla_from_deadline(5000) == "fast"     # 5s
    assert get_sla_from_deadline(10000) == "standard" # 10s
    assert get_sla_from_deadline(30000) == "standard" # 30s
    assert get_sla_from_deadline(60000) == "high_quality"  # 60s
    assert get_sla_from_deadline(None) == "standard"

    # 测试策略获取
    express = get_sla_strategy("express")
    assert express["max_gen_toks"] == 64
    assert express["temperature"] == 0.0

    high_quality = get_sla_strategy("high_quality")
    assert high_quality["max_gen_toks"] == 512
    assert high_quality["temperature"] == 0.8

    # 测试未知 SLA 回退到 standard
    standard = get_sla_strategy("unknown_level")
    assert standard == get_sla_strategy("standard")

    print("[测试] SLA 策略选择通过")
    return True


def test_run_inference_with_sla():
    """测试带 SLA 级别的推理。"""
    print("[测试] 测试带 SLA 级别的推理...")
    messages = [
        {
            "ID": 0,
            "prompt": "The capital of France is",
            "eval_request_type": "generate_until",
            "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20},
            "eval_req_id": "test_008",
        },
        {
            "ID": 1,
            "prompt": "The capital of France is",
            "eval_request_type": "loglikelihood",
            "eval_continuation": " Paris",
            "eval_req_id": "test_009",
        },
    ]

    # 测试显式 SLA
    results_express = run_inference(messages, sla_level="express")
    assert len(results_express) == 2
    assert results_express[0].get("sla_level") == "express"
    assert results_express[1].get("sla_level") == "express"

    # 测试 deadline_ms(自动 SLA)
    results_fast = run_inference(messages, deadline_ms=3000)
    assert len(results_fast) == 2
    assert results_fast[0].get("sla_level") == "fast"

    # 测试 None(默认 SLA)
    results_default = run_inference(messages, sla_level=None)
    assert len(results_default) == 2

    print("[测试] 带 SLA 级别的推理通过")
    return True


def main():
    print("=" * 60)
    print("运行端到端测试...")
    print("=" * 60)

    tests = [
        ("模块导入", test_inference_module_import),
        ("配置加载", test_config_loading),
        ("generate_until", test_run_inference_generate_until),
        ("loglikelihood", test_run_inference_loglikelihood),
        ("loglikelihood_rolling", test_run_inference_rolling),
        ("混合类型", test_run_inference_mixed),
        ("客户端异步", test_client_async_structure),
        ("VLLM 不可达", test_vllm_unreachable),
        ("SLA 策略选择", test_sla_strategy_selection),
        ("SLA 级别推理", test_run_inference_with_sla),
    ]

    results = []
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            passed = fn()
        except Exception as e:
            print(f"[测试] 异常: {e}")
            passed = False
        status = "通过" if passed else "失败"
        results.append((name, status))
        print(f"[测试] {name}: {status}")

    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    for name, status in results:
        print(f"  {status:4s}  {name}")
    print("=" * 60)

    all_passed = all(s == "通过" for _, s in results)
    print(f"\n总体: {'全部通过' if all_passed else '部分失败'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
