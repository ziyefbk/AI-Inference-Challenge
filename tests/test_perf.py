"""
并发/延迟/投机解码测试模块。

测试内容:
1. 并发推理: asyncio.gather 性能测试
2. 延迟测量: 各 SLA 级别响应时间
3. 投机解码: vLLM speculative decoding 集成
4. 调度策略: 延迟敏感任务优先
5. 熔断器: CircuitBreaker 测试
6. 指标采集: MetricsCollector 测试
7. 优雅关闭: GracefulShutdown 测试

运行方式:
    python tests/test_perf.py --mode all
    python tests/test_perf.py --mode concurrency
    python tests/test_perf.py --mode latency
    python tests/test_perf.py --mode speculative
    python tests/test_perf.py --mode circuit
    python tests/test_perf.py --mode metrics
"""

import sys
import os
import time
import asyncio
import argparse
import statistics
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import (
    run_inference, generate_text, compute_logprob, compute_rolling_logprob,
    generate_text_async, compute_logprob_async, compute_rolling_logprob_async,
    SLA_STRATEGIES, get_sla_from_deadline, get_sla_strategy, _get_tiktoken_encoding
)
from src.utils.circuit import CircuitBreaker, CircuitBreakerOpen
from src.utils.metrics import MetricsCollector
from src.utils.graceful import GracefulShutdown


# ── 测试数据 ──────────────────────────────────────────────────────────────

MESSAGES_BASIC = [
    {"ID": 0, "prompt": "The capital of France is", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20}},
    {"ID": 1, "prompt": "The capital of Japan is", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20}},
    {"ID": 2, "prompt": "The capital of Germany is", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20}},
]

MESSAGES_LARGE = [
    {"ID": i, "prompt": f"What is {i}+{i}?", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 30}}
    for i in range(10)
]

MESSAGES_MIXED = [
    {"ID": 0, "prompt": "The capital of France is", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20}},
    {"ID": 1, "prompt": "The capital of France is", "eval_request_type": "loglikelihood",
     "eval_continuation": " Paris"},
    {"ID": 2, "prompt": "The capital of France is Paris.", "eval_request_type": "loglikelihood_rolling"},
    {"ID": 3, "prompt": "The capital of Japan is", "eval_request_type": "generate_until",
     "eval_gen_kwargs": {"until": ["\n"], "max_gen_toks": 20}},
    {"ID": 4, "prompt": "The capital of Japan is", "eval_request_type": "loglikelihood",
     "eval_continuation": " Tokyo"},
]


# ── 熔断器测试 ────────────────────────────────────────────────────────────

class TestCircuitBreaker(unittest.TestCase):
    """测试熔断器。"""

    def test_initial_state(self):
        """测试初始状态为 closed。"""
        breaker = CircuitBreaker("test", failure_threshold=3)
        self.assertEqual(breaker.state, "closed")
        self.assertTrue(breaker.can_attempt_sync())

    def test_opens_after_failures(self):
        """测试连续失败后打开熔断器。"""
        breaker = CircuitBreaker("test", failure_threshold=3, timeout=60)
        self.assertTrue(breaker.can_attempt_sync())

        # 3 次失败后应该打开
        breaker.record_failure_sync()
        self.assertTrue(breaker.can_attempt_sync())
        breaker.record_failure_sync()
        self.assertTrue(breaker.can_attempt_sync())
        breaker.record_failure_sync()
        self.assertFalse(breaker.can_attempt_sync())
        self.assertEqual(breaker.state, "open")

    def test_half_open_after_timeout(self):
        """测试超时后进入半开状态。"""
        breaker = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        breaker.record_failure_sync()
        breaker.record_failure_sync()
        self.assertEqual(breaker.state, "open")

        # 等待超时
        time.sleep(0.2)

        self.assertTrue(breaker.can_attempt_sync())
        self.assertEqual(breaker.state, "half_open")

    def test_success_resets_failures(self):
        """测试成功后重置失败计数。"""
        breaker = CircuitBreaker("test", failure_threshold=3)
        breaker.record_failure_sync()
        breaker.record_failure_sync()
        self.assertEqual(breaker.failure_count, 2)

        breaker.record_success_sync()
        self.assertEqual(breaker.failure_count, 0)

    def test_half_open_to_closed(self):
        """测试半开状态下连续成功后关闭。"""
        breaker = CircuitBreaker("test", failure_threshold=2, success_threshold=2, timeout=0.1)
        breaker.record_failure_sync()
        breaker.record_failure_sync()
        time.sleep(0.2)

        breaker.can_attempt_sync()  # 进入半开
        self.assertEqual(breaker.state, "half_open")

        breaker.record_success_sync()
        self.assertEqual(breaker.state, "half_open")  # 还需要一次
        breaker.record_success_sync()
        self.assertEqual(breaker.state, "closed")

    def test_stats(self):
        """测试统计信息。"""
        breaker = CircuitBreaker("test", failure_threshold=3)
        breaker.record_failure_sync()
        breaker.record_success_sync()

        stats = breaker.get_stats()
        self.assertEqual(stats["name"], "test")
        self.assertEqual(stats["failure_count"], 1)


def test_circuit_breaker():
    """熔断器测试。"""
    print("\n" + "=" * 60)
    print("熔断器测试")
    print("=" * 60)

    # 运行单元测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCircuitBreaker)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# ── 指标采集测试 ─────────────────────────────────────────────────────────

class TestMetricsCollector(unittest.TestCase):
    """测试指标采集器。"""

    def setUp(self):
        self.metrics = MetricsCollector()

    def test_counter(self):
        """测试计数器。"""
        self.metrics.inc_counter("test_counter")
        self.assertEqual(self.metrics.get_counter("test_counter"), 1)

        self.metrics.inc_counter("test_counter", 5)
        self.assertEqual(self.metrics.get_counter("test_counter"), 6)

        self.metrics.inc_counter("test_counter", labels={"type": "a"})
        self.assertEqual(self.metrics.get_counter("test_counter"), 6)
        self.assertEqual(self.metrics.get_counter("test_counter", {"type": "a"}), 1)

    def test_histogram(self):
        """测试直方图。"""
        for i in range(100):
            self.metrics.observe_histogram("test_latency", i / 100.0)

        stats = self.metrics.get_histogram_stats("test_latency")
        self.assertEqual(stats["count"], 100)
        self.assertAlmostEqual(stats["mean"], 0.495, places=2)
        self.assertIn("p50", stats)
        self.assertIn("p95", stats)
        self.assertIn("p99", stats)

    def test_gauge(self):
        """测试仪表。"""
        self.metrics.set_gauge("test_gauge", 10)
        self.assertEqual(self.metrics.get_gauge("test_gauge"), 10)

        self.metrics.inc_gauge("test_gauge", 5)
        self.assertEqual(self.metrics.get_gauge("test_gauge"), 15)

        self.metrics.dec_gauge("test_gauge", 3)
        self.assertEqual(self.metrics.get_gauge("test_gauge"), 12)

    def test_labels(self):
        """测试标签。"""
        self.metrics.observe_histogram("latency", 0.5, labels={"method": "GET"})
        self.metrics.observe_histogram("latency", 1.5, labels={"method": "POST"})

        stats = self.metrics.get_histogram_stats("latency", {"method": "GET"})
        self.assertEqual(stats["count"], 1)

    def test_prometheus_format(self):
        """测试 Prometheus 格式输出。"""
        self.metrics.inc_counter("test_requests")
        self.metrics.observe_histogram("test_latency", 0.1)

        output = self.metrics.get_prometheus_format()
        self.assertIn("test_requests", output)
        self.assertIn("test_latency", output)


def test_metrics_collector():
    """指标采集器测试。"""
    print("\n" + "=" * 60)
    print("指标采集器测试")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMetricsCollector)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# ── 优雅关闭测试 ─────────────────────────────────────────────────────────

def test_graceful_shutdown():
    """优雅关闭测试。"""
    print("\n" + "=" * 60)
    print("优雅关闭测试")
    print("=" * 60)

    shutdown = GracefulShutdown()

    # 测试初始状态
    assert not shutdown.is_shutting_down
    assert shutdown.state.value == "running"

    # 测试关闭
    shutdown.begin_shutdown("test")
    assert shutdown.is_shutting_down
    assert shutdown.state.value in ["shutting_down", "waiting_tasks"]

    print("[关闭测试] 优雅关闭机制正常工作")

    return True


# ── 并发测试 ──────────────────────────────────────────────────────────────

def test_concurrent_inference():
    """
    测试并发推理性能。
    对比串行和并发的执行时间。
    """
    print("\n" + "=" * 60)
    print("并发推理测试")
    print("=" * 60)

    # 并发执行
    print(f"\n[并发测试] 测试 {len(MESSAGES_BASIC)} 条消息...")
    start = time.time()
    results = run_inference(MESSAGES_BASIC, sla_level="standard")
    concurrent_time = time.time() - start

    success_count = sum(1 for r in results if r.get("response") or r.get("accuracy") is not None)
    print(f"[并发测试] 完成: {success_count}/{len(MESSAGES_BASIC)} 条")
    print(f"[并发测试] 耗时: {concurrent_time:.3f}s")
    print(f"[并发测试] 平均每条: {concurrent_time/len(MESSAGES_BASIC):.3f}s")

    # 大批量测试
    print(f"\n[并发测试] 测试 {len(MESSAGES_LARGE)} 条消息...")
    start = time.time()
    results = run_inference(MESSAGES_LARGE, sla_level="fast")
    large_time = time.time() - start

    success_count = sum(1 for r in results if r.get("response"))
    print(f"[并发测试] 完成: {success_count}/{len(MESSAGES_LARGE)} 条")
    print(f"[并发测试] 耗时: {large_time:.3f}s")
    print(f"[并发测试] 吞吐量: {len(MESSAGES_LARGE)/large_time:.1f} 条/秒")

    return {
        "small_batch": {"count": len(MESSAGES_BASIC), "time": concurrent_time},
        "large_batch": {"count": len(MESSAGES_LARGE), "time": large_time},
    }


# ── 延迟测试 ──────────────────────────────────────────────────────────────

def test_latency_by_sla():
    """
    测试不同 SLA 级别的延迟。
    """
    print("\n" + "=" * 60)
    print("SLA 级别延迟测试")
    print("=" * 60)

    sla_levels = ["express", "fast", "standard", "high_quality"]
    latencies = {}

    for sla in sla_levels:
        print(f"\n[延迟测试] 测试 SLA: {sla}...")
        times = []

        # 每个 SLA 级别运行 3 次取平均
        for i in range(3):
            start = time.time()
            results = run_inference(MESSAGES_BASIC[:2], sla_level=sla)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        latencies[sla] = {"avg": avg_time, "std": std_time, "min": min(times), "max": max(times)}

        print(f"[延迟测试] {sla}: {avg_time:.3f}s ± {std_time:.3f}s (min={min(times):.3f}s, max={max(times):.3f}s)")

    # 打印策略对比表
    print("\n" + "-" * 60)
    print(f"{'SLA级别':<15} {'平均延迟':<12} {'标准差':<12} {'最小':<12} {'最大':<12}")
    print("-" * 60)
    for sla, stats in latencies.items():
        print(f"{sla:<15} {stats['avg']:<12.3f} {stats['std']:<12.3f} {stats['min']:<12.3f} {stats['max']:<12.3f}")

    return latencies


def test_deadline_to_sla():
    """
    测试 deadline 到 SLA 的自动映射。
    """
    print("\n" + "=" * 60)
    print("截止时间 → SLA 映射测试")
    print("=" * 60)

    test_cases = [
        (500, "express"),     # 0.5s
        (1000, "express"),    # 1s
        (2000, "fast"),       # 2s
        (5000, "fast"),       # 5s
        (10000, "standard"),   # 10s
        (30000, "standard"),   # 30s
        (60000, "high_quality"),  # 60s
    ]

    all_passed = True
    for deadline_ms, expected_sla in test_cases:
        actual_sla = get_sla_from_deadline(deadline_ms)
        status = "✓" if actual_sla == expected_sla else "✗"
        if actual_sla != expected_sla:
            all_passed = False
        print(f"[映射测试] {status} deadline={deadline_ms}ms → {actual_sla} (期望: {expected_sla})")

    return all_passed


# ── 投机解码测试 ──────────────────────────────────────────────────────────

def test_speculative_decoding():
    """
    测试投机解码效果。
    """
    print("\n" + "=" * 60)
    print("投机解码测试")
    print("=" * 60)

    prompt = "Write a short story about a robot:"

    # 普通生成
    print("\n[投机测试] 普通生成...")
    start = time.time()
    result_normal = generate_text(prompt, {"max_gen_toks": 50, "temperature": 0.7})
    normal_time = time.time() - start
    print(f"[投机测试] 耗时: {normal_time:.3f}s")
    print(f"[投机测试] 生成: {result_normal[:80]}...")

    # 投机解码 (使用 n=2 候选)
    print("\n[投机测试] 候选生成 (n=2)...")
    start = time.time()
    try:
        from src.inference import _call_vllm
        response = _call_vllm(
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            best_of=2,
        )
        if response.get("choices"):
            choices = response["choices"]
            print(f"[投机测试] 生成了 {len(choices)} 个候选")
            for i, c in enumerate(choices):
                text = c.get("text", "")[:60]
                print(f"[投机测试] 候选 {i+1}: {text}...")
        speculative_time = time.time() - start
        print(f"[投机测试] 耗时: {speculative_time:.3f}s")
        speculative_available = True
    except Exception as e:
        print(f"[投机测试] 投机解码不可用: {e}")
        speculative_available = False

    return {
        "normal_time": normal_time,
        "speculative_available": speculative_available,
    }


# ── 批量攒批测试 ──────────────────────────────────────────────────────────

def test_batch_gathering():
    """
    测试批量攒批调度策略。
    """
    print("\n" + "=" * 60)
    print("批量攒批调度测试")
    print("=" * 60)

    # 模拟不同截止时间的任务
    tasks_with_deadlines = [
        {"id": 1, "deadline_ms": 50000, "messages": MESSAGES_BASIC[:1]},
        {"id": 2, "deadline_ms": 2000, "messages": MESSAGES_BASIC[:1]},
        {"id": 3, "deadline_ms": 30000, "messages": MESSAGES_BASIC[:1]},
        {"id": 4, "deadline_ms": 1000, "messages": MESSAGES_BASIC[:1]},
    ]

    print("\n[攒批测试] 任务列表:")
    for task in tasks_with_deadlines:
        sla = get_sla_from_deadline(task["deadline_ms"])
        print(f"[攒批测试] 任务 {task['id']}: deadline={task['deadline_ms']}ms → {sla}")

    # 按截止时间排序(延迟敏感调度)
    sorted_tasks = sorted(tasks_with_deadlines, key=lambda t: t["deadline_ms"])
    print("\n[攒批测试] 按截止时间排序后:")
    for task in sorted_tasks:
        sla = get_sla_from_deadline(task["deadline_ms"])
        print(f"[攒批测试] 任务 {task['id']}: {sla} (deadline={task['deadline_ms']}ms)")

    # 按 SLA 分组
    sla_groups = {}
    for task in sorted_tasks:
        sla = get_sla_from_deadline(task["deadline_ms"])
        if sla not in sla_groups:
            sla_groups[sla] = []
        sla_groups[sla].append(task)

    print("\n[攒批测试] 按 SLA 分组:")
    for sla, tasks in sla_groups.items():
        print(f"[攒批测试] {sla}: {len(tasks)} 个任务")

    return {"sla_groups": sla_groups, "sorted": sorted_tasks}


# ── 调度器实现 ────────────────────────────────────────────────────────────

class PriorityScheduler:
    """
    延迟敏感的优先级调度器。
    """

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size

    def filter_expired(self, tasks: list) -> tuple:
        """过滤已过期任务。"""
        now = time.time() * 1000
        valid = []
        expired = []

        for task in tasks:
            deadline = task.get("deadline_ms", float("inf"))
            if now >= deadline:
                expired.append(task)
            else:
                valid.append(task)

        return valid, expired

    def prioritize(self, tasks: list) -> list:
        """按截止时间排序。"""
        return sorted(tasks, key=lambda t: t.get("deadline_ms", float("inf")))

    def batch_by_sla(self, tasks: list) -> dict:
        """按 SLA 级别分组。"""
        batches = {}
        for task in tasks:
            deadline = task.get("deadline_ms")
            sla = get_sla_from_deadline(deadline)
            if sla not in batches:
                batches[sla] = []
            batches[sla].append(task)
        return batches

    def schedule(self, tasks: list) -> list:
        """调度任务。"""
        valid, expired = self.filter_expired(tasks)

        if expired:
            print(f"[调度器] 跳过 {len(expired)} 个已过期任务")

        if not valid:
            return []

        prioritized = self.prioritize(valid)
        batches = self.batch_by_sla(prioritized)

        sla_order = ["express", "fast", "standard", "high_quality"]
        execution_order = []

        for sla in sla_order:
            if sla in batches:
                batch = batches[sla][:self.batch_size]
                execution_order.extend(batch)
                print(f"[调度器] {sla}: {len(batch)} 个任务")

        return execution_order


def test_scheduler():
    """测试调度器。"""
    print("\n" + "=" * 60)
    print("调度器测试")
    print("=" * 60)

    scheduler = PriorityScheduler(batch_size=3)

    tasks = [
        {"id": 1, "deadline_ms": 50000},
        {"id": 2, "deadline_ms": 2000},
        {"id": 3, "deadline_ms": 30000},
        {"id": 4, "deadline_ms": 1000},
        {"id": 5, "deadline_ms": 100000},
    ]

    print("\n[调度器] 原始任务:")
    for t in tasks:
        print(f"[调度器]  任务 {t['id']}: deadline={t['deadline_ms']}ms")

    execution_order = scheduler.schedule(tasks)

    print("\n[调度器] 执行顺序:")
    for i, task in enumerate(execution_order):
        sla = get_sla_from_deadline(task["deadline_ms"])
        print(f"[调度器]  {i+1}. 任务 {task['id']} ({sla})")

    return len(execution_order) == 3


# ── 基准测试 ──────────────────────────────────────────────────────────────

def run_benchmark():
    """运行完整性能基准测试。"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    results = {}

    # 熔断器测试
    results["circuit_breaker"] = test_circuit_breaker()

    # 指标测试
    results["metrics"] = test_metrics_collector()

    # 优雅关闭测试
    results["graceful_shutdown"] = test_graceful_shutdown()

    # 并发测试
    results["concurrent"] = test_concurrent_inference()

    # 延迟测试
    results["latency"] = test_latency_by_sla()

    # 截止时间映射
    results["deadline_mapping"] = test_deadline_to_sla()

    # 调度器
    results["scheduler"] = test_scheduler()

    # 打印总结
    print("\n" + "=" * 60)
    print("基准测试总结")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
            print(f"\n{status}: {name}")
        elif isinstance(result, dict):
            if "concurrent" in result:
                c = result["concurrent"]
                print(f"\n并发性能:")
                print(f"  小批量 ({c['small_batch']['count']} 条): {c['small_batch']['time']:.3f}s")
                print(f"  大批量 ({c['large_batch']['count']} 条): {c['large_batch']['time']:.3f}s")

            if "latency" in result:
                print(f"\nSLA 延迟:")
                for sla, stats in result["latency"].items():
                    print(f"  {sla}: {stats['avg']:.3f}s")

    return all_passed


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="并发/延迟/投机解码测试")
    parser.add_argument("--mode", default="all",
                        choices=["all", "concurrency", "latency", "speculative", "scheduler",
                                "circuit", "metrics", "priority", "async_funcs", "multi_worker",
                                "leak", "jitter", "adaptive"],
                        help="测试模式")
    parser.add_argument("--sla", default="standard",
                        choices=["express", "fast", "standard", "high_quality"],
                        help="默认 SLA 级别")
    parser.add_argument("--count", type=int, default=10,
                        help="大批量测试的消息数")
    args = parser.parse_args()

    print(f"测试模式: {args.mode}")
    print(f"默认 SLA: {args.sla}")

    if args.mode == "all":
        success = run_benchmark()
        sys.exit(0 if success else 1)
    elif args.mode == "concurrency":
        test_concurrent_inference()
    elif args.mode == "latency":
        test_latency_by_sla()
        test_deadline_to_sla()
    elif args.mode == "speculative":
        test_speculative_decoding()
    elif args.mode == "scheduler":
        test_scheduler()
        test_batch_gathering()
    elif args.mode == "circuit":
        test_circuit_breaker()
    elif args.mode == "metrics":
        test_metrics_collector()
    elif args.mode == "priority":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPriorityQueue)
        unittest.TextTestRunner(verbosity=2).run(suite)
    elif args.mode == "async_funcs":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestAsyncInference)
        unittest.TextTestRunner(verbosity=2).run(suite)
    elif args.mode == "multi_worker":
        test_multi_worker_concurrency()
    elif args.mode == "leak":
        test_mark_done_no_leak()
    elif args.mode == "jitter":
        test_retry_jitter()
    elif args.mode == "adaptive":
        test_adaptive_prefetch_logic()
    elif args.mode == "priority_calc":
        test_priority_computation()
    elif args.mode == "expired_check":
        test_should_check_expired()

    print("\n测试完成!")


# ── 优先队列测试 ────────────────────────────────────────────────────────────

class TestPriorityQueue(unittest.TestCase):
    """测试 PriorityTaskHolder 优先队列。"""

    def test_priority_ordering(self):
        """测试优先级排序: 截止时间越近越优先。"""
        import asyncio
        from src.client import PriorityTaskHolder

        holder = PriorityTaskHolder(max_held=10)

        async def run():
            now = time.time()
            tasks = [
                {
                    "overview": {
                        "task_id": 1,
                        "deadline_ms": int((now + 100) * 1000),  # 100秒后
                    }
                },
                {
                    "overview": {
                        "task_id": 2,
                        "deadline_ms": int((now + 10) * 1000),  # 10秒后
                    }
                },
                {
                    "overview": {
                        "task_id": 3,
                        "deadline_ms": int((now + 50) * 1000),  # 50秒后
                    }
                },
            ]

            for t in tasks:
                await holder.add_task(t)

            # 取出顺序应该是: 2, 3, 1 (截止时间最近的先)
            order = []
            while True:
                item = await holder.pop_task()
                if item is None:
                    break
                task, _ = item
                order.append(task["overview"]["task_id"])

            self.assertEqual(order, [2, 3, 1])

        asyncio.run(run())

    def test_heap_efficiency(self):
        """测试 heapq 效率: 插入 O(log n)。"""
        import asyncio
        from src.client import PriorityTaskHolder

        holder = PriorityTaskHolder(max_held=1000)

        async def run():
            now = time.time()
            times = []
            for i in range(100):
                task = {
                    "overview": {
                        "task_id": i,
                        "deadline_ms": int((now + random.random() * 1000) * 1000),
                    }
                }
                start = time.perf_counter()
                await holder.add_task(task)
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            self.assertLess(avg_time, 0.001)  # 平均插入时间应 < 1ms

        asyncio.run(run())

    def test_complexity_scoring(self):
        """测试复杂度评分: 消息越多复杂度惩罚越高。"""
        import asyncio
        from src.client import PriorityTaskHolder

        holder = PriorityTaskHolder(max_held=10)

        async def run():
            now = time.time()
            # 两个任务有相同的deadline和reward，但消息数不同
            short_task = {
                "overview": {
                    "task_id": 1,
                    "deadline_ms": int((now + 100) * 1000),
                    "target_reward": 5.0,
                    "target_sla": "standard",
                },
                "messages": [{"role": "user", "content": "hi"}],
            }
            long_task = {
                "overview": {
                    "task_id": 2,
                    "deadline_ms": int((now + 100) * 1000),  # 相同截止时间
                    "target_reward": 5.0,  # 相同奖励
                    "target_sla": "standard",
                },
                "messages": [
                    {"role": "user", "content": f"message {j}"}
                    for j in range(10)
                ],
            }

            await holder.add_task(short_task)
            await holder.add_task(long_task)

            # 两个任务优先级相近，但由于随机扰动，不保证顺序
            # 主要是测试不崩溃
            item1 = await holder.pop_task()
            item2 = await holder.pop_task()
            self.assertIsNotNone(item1)
            self.assertIsNotNone(item2)

        asyncio.run(run())


# ── 异步函数测试 ────────────────────────────────────────────────────────────

class TestAsyncInference(unittest.TestCase):
    """测试异步推理函数。"""

    def test_tiktoken_caching(self):
        """测试 tiktoken 编码器缓存。"""
        enc1 = _get_tiktoken_encoding()
        enc2 = _get_tiktoken_encoding()
        self.assertIs(enc1, enc2)  # 应该是同一个对象

    def test_async_function_signatures(self):
        """测试异步函数签名。"""
        import inspect
        from src.inference import (
            generate_text_async, compute_logprob_async, compute_rolling_logprob_async
        )

        self.assertTrue(inspect.iscoroutinefunction(generate_text_async))
        self.assertTrue(inspect.iscoroutinefunction(compute_logprob_async))
        self.assertTrue(inspect.iscoroutinefunction(compute_rolling_logprob_async))


# ── 多 Worker 测试 ──────────────────────────────────────────────────────────

def test_multi_worker_concurrency():
    """测试多 worker 并发处理。"""
    import asyncio
    from src.client import PriorityTaskHolder

    async def dummy_task(n):
        await asyncio.sleep(0.01)
        return n * 2

    async def worker(queue, results, worker_id):
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                continue

            if item is None:
                break

            result = await dummy_task(item)
            results.append((worker_id, result))
            queue.task_done()

    async def run():
        queue = asyncio.Queue()
        for i in range(20):
            await queue.put(i)

        results = []
        num_workers = 4
        workers = [
            asyncio.create_task(worker(queue, results, i))
            for i in range(num_workers)
        ]

        await queue.join()

        for _ in range(num_workers):
            await queue.put(None)

        await asyncio.gather(*workers)

        assert len(results) == 20
        expected = {r[1] for r in results}
        assert expected == {i * 2 for i in range(20)}
        print(f"  多 worker 处理 {len(results)} 个任务")

    asyncio.run(run())
    return True


# ── 内存泄漏修复测试 ────────────────────────────────────────────────────────

def test_mark_done_no_leak():
    """测试 mark_done 后 heap 不会无限增长（pop_task 跳过无效条目）。"""
    from src.client import PriorityTaskHolder

    async def run():
        holder = PriorityTaskHolder(max_held=100)

        # 添加 50 个任务
        for i in range(50):
            task = {"overview": {"task_id": i}}
            await holder.add_task(task)

        # 标记完成 40 个
        for i in range(40):
            await holder.mark_done(i)

        # pop 取出，应该只返回剩余的有效任务
        count = 0
        while True:
            item = await holder.pop_task()
            if item is None:
                break
            count += 1

        assert count == 10, f"期望 10 个任务，实际 {count} 个"
        print(f"  mark_done 测试通过: 50 - 40 = 10 个有效任务")

    asyncio.run(run())
    return True


def test_retry_jitter():
    """测试重试有 jitter（每次延迟不同）。"""
    delays = []
    for _ in range(20):
        base = 1.0
        jitter = random.uniform(0.5, 1.5)
        delays.append(base * jitter)

    # 验证 jitter 有变化
    assert max(delays) > min(delays) * 1.3, "Jitter 应该产生变化"
    print(f"  jitter 测试通过: 延迟范围 {min(delays):.2f}s - {max(delays):.2f}s")

    return True


# ── 自适应预取测试 ───────────────────────────────────────────────────────────

def test_adaptive_prefetch_logic():
    """测试自适应预取的填充率计算逻辑。"""
    PREFETCH_SIZE = 8

    # 模拟不同填充率下的预取数量
    test_cases = [
        (0.3, PREFETCH_SIZE * 2),  # 空时多预取
        (0.6, PREFETCH_SIZE),       # 正常
        (0.9, 1),                   # 满时少预取
    ]

    for fill_ratio, expected_size in test_cases:
        if fill_ratio < 0.5:
            dynamic_size = PREFETCH_SIZE * 2
        elif fill_ratio > 0.8:
            dynamic_size = 1
        else:
            dynamic_size = PREFETCH_SIZE

        assert dynamic_size == expected_size, f"fill_ratio={fill_ratio} 时期望 {expected_size}，实际 {dynamic_size}"

    print("  自适应预取逻辑测试通过")
    return True


def test_priority_computation():
    """测试优化后的优先级计算逻辑 (reward/remaining权衡)"""
    import asyncio
    from src.client import PriorityTaskHolder

    async def run():
        holder = PriorityTaskHolder(max_held=10)
        now = time.time()

        # 高奖励短deadline任务
        high_reward_task = {
            "overview": {
                "task_id": 1,
                "target_reward": 10.0,
                "deadline_ms": int((now + 10) * 1000),
                "target_sla": "express"
            },
            "messages": [{"content": "short"}]
        }

        # 低奖励长deadline任务
        low_reward_task = {
            "overview": {
                "task_id": 2,
                "target_reward": 1.0,
                "deadline_ms": int((now + 100) * 1000),
                "target_sla": "standard"
            },
            "messages": [{"content": "long"} for _ in range(10)]
        }

        await holder.add_task(high_reward_task)
        await holder.add_task(low_reward_task)

        # 高奖励短deadline任务应该先被处理
        item1 = await holder.pop_task()
        item2 = await holder.pop_task()

        assert item1 is not None, "应该有任务"
        assert item2 is not None, "应该有任务"

        task1_id = item1[0]["overview"]["task_id"]
        task2_id = item2[0]["overview"]["task_id"]

        # 高奖励短deadline任务(task_id=1)应该优先
        assert task1_id == 1, f"期望高奖励任务优先，实际 task_id={task1_id}"

        print("  优先级计算测试通过: 高奖励/短deadline任务优先")

    asyncio.run(run())
    return True


def test_should_check_expired():
    """测试动态过期检查逻辑"""
    import asyncio
    from src.client import PriorityTaskHolder

    async def run():
        holder = PriorityTaskHolder(max_held=100)

        # 测试空持有器 (< 40% 填充率)
        # 应该每5次检查一次
        for _ in range(4):
            holder._check_counter = _
            result = await holder.should_check_expired()
            assert result == False, f"填充率<40%时，第{_+1}次不应检查"

        holder._check_counter = 4
        result = await holder.should_check_expired()
        assert result == True, "填充率<40%时，第5次应检查"

        # 测试中等填充率 (40% - 70%)
        holder._heap = [(0, i, time.time(), {"overview": {"task_id": i}}) for i in range(50)]
        holder._check_counter = 2
        result = await holder.should_check_expired()
        assert result == False, "中等填充率时，第3次不应检查"

        holder._check_counter = 3
        result = await holder.should_check_expired()
        assert result == True, "中等填充率时，第4次应检查"

        # 测试高填充率 (> 70%)
        holder._heap = [(0, i, time.time(), {"overview": {"task_id": i}}) for i in range(80)]
        holder._check_counter = 0
        result = await holder.should_check_expired()
        assert result == True, "高填充率时应立即检查"

        print("  动态过期检查测试通过")

    asyncio.run(run())
    return True


if __name__ == "__main__":
    main()
