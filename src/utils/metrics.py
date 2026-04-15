"""
统一指标采集模块，支持 Prometheus 格式和分位数统计。

功能:
- Counter: 累加计数器
- Histogram: 延迟/大小分布统计 (自动计算 P50/P95/P99)
- Gauge: 当前值
- 带 labels 的多维指标
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class HistogramBucket:
    """直方图桶。"""
    values: List[float] = field(default_factory=list)

    def observe(self, value: float):
        """记录一个观测值。"""
        self.values.append(value)

    def get_percentiles(self, percentiles: List[float] = None) -> Dict[str, float]:
        """计算百分位数。"""
        if percentiles is None:
            percentiles = [0.5, 0.9, 0.95, 0.99]

        if not self.values:
            return {f"p{int(p*100)}": 0.0 for p in percentiles}

        sorted_values = sorted(self.values)
        result = {}

        for p in percentiles:
            idx = int(len(sorted_values) * p)
            if idx >= len(sorted_values):
                idx = len(sorted_values) - 1
            result[f"p{int(p*100)}"] = sorted_values[idx]

        return result

    def get_stats(self) -> Dict[str, float]:
        """获取完整统计信息。"""
        if not self.values:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(self.values),
            "sum": sum(self.values),
            "mean": statistics.mean(self.values),
            "min": min(self.values),
            "max": max(self.values),
            **self.get_percentiles(),
        }

    def clear(self):
        """清空数据。"""
        self.values.clear()


class MetricsCollector:
    """
    统一指标采集器。

    示例:
        metrics = MetricsCollector()

        # 计数
        metrics.inc_counter("tasks_completed", labels={"sla": "fast"})

        # 直方图
        metrics.observe_histogram("inference_latency", 0.5, labels={"type": "generate"})

        # 获取统计
        stats = metrics.get_stats()
        print(stats["inference_latency"]["p95"])
    """

    def __init__(self, max_samples_per_histogram: int = 10000):
        self.max_samples = max_samples_per_histogram
        self._counters: Dict[str, float] = defaultdict(float)
        self._counters_with_labels: Dict[str, Dict[tuple, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: Dict[str, HistogramBucket] = {}
        self._histograms_with_labels: Dict[str, Dict[tuple, HistogramBucket]] = defaultdict(
            lambda: {}
        )
        self._gauges: Dict[str, float] = {}
        self._gauges_with_labels: Dict[str, Dict[tuple, float]] = defaultdict(dict)
        self._lock = threading.RLock()

        # 记录开始时间
        self._start_time = time.time()

    def _make_label_key(self, labels: Dict[str, str]) -> tuple:
        """将 labels 字典转换为可哈希的元组。"""
        return tuple(sorted(labels.items()))

    # ── Counter ──────────────────────────────────────────────────────────

    def inc_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """递增计数器。"""
        with self._lock:
            if labels is None:
                self._counters[name] += value
            else:
                key = self._make_label_key(labels)
                self._counters_with_labels[name][key] += value

    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取计数器当前值。"""
        with self._lock:
            if labels is None:
                return self._counters.get(name, 0.0)
            else:
                key = self._make_label_key(labels)
                return self._counters_with_labels[name].get(key, 0.0)

    # ── Histogram ────────────────────────────────────────────────────────

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ):
        """记录直方图观测值。"""
        with self._lock:
            if labels is None:
                if name not in self._histograms:
                    self._histograms[name] = HistogramBucket()
                bucket = self._histograms[name]
            else:
                key = self._make_label_key(labels)
                if name not in self._histograms_with_labels:
                    self._histograms_with_labels[name] = {}
                if key not in self._histograms_with_labels[name]:
                    self._histograms_with_labels[name][key] = HistogramBucket()
                bucket = self._histograms_with_labels[name][key]

            bucket.observe(value)

            # 限制样本数量，防止内存泄漏
            if len(bucket.values) > self.max_samples:
                # 保留最近的样本
                bucket.values = bucket.values[-self.max_samples // 2:]

    def get_histogram_stats(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Dict[str, float]:
        """获取直方图统计信息。"""
        with self._lock:
            if labels is None:
                bucket = self._histograms.get(name)
            else:
                key = self._make_label_key(labels)
                bucket = self._histograms_with_labels.get(name, {}).get(key)

            if bucket is None:
                return {"count": 0, "sum": 0.0, "mean": 0.0}

            return bucket.get_stats()

    # ── Gauge ────────────────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表值。"""
        with self._lock:
            if labels is None:
                self._gauges[name] = value
            else:
                key = self._make_label_key(labels)
                self._gauges_with_labels[name][key] = value

    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """获取仪表当前值。"""
        with self._lock:
            if labels is None:
                return self._gauges.get(name)
            else:
                key = self._make_label_key(labels)
                return self._gauges_with_labels.get(name, {}).get(key)

    def inc_gauge(self, name: str, delta: float = 1.0, labels: Dict[str, str] = None):
        """增加仪表值。"""
        with self._lock:
            current = self.get_gauge(name, labels) or 0.0
            self.set_gauge(name, current + delta, labels)

    def dec_gauge(self, name: str, delta: float = 1.0, labels: Dict[str, str] = None):
        """减少仪表值。"""
        with self._lock:
            current = self.get_gauge(name, labels) or 0.0
            self.set_gauge(name, current - delta, labels)

    # ── 综合统计 ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        获取所有指标统计。

        Returns:
            包含 counters, histograms, gauges 的字典
        """
        with self._lock:
            uptime = time.time() - self._start_time

            # 计算 QPS
            counters_summary = {}
            for name, value in self._counters.items():
                counters_summary[name] = {
                    "value": value,
                    "rate": value / uptime if uptime > 0 else 0.0,
                }

            # Histograms with labels
            histograms_summary = {}
            for name, buckets in self._histograms_with_labels.items():
                histograms_summary[name] = {}
                for key, bucket in buckets.items():
                    histograms_summary[name][str(dict(key))] = bucket.get_stats()

            # 简单 histograms
            for name, bucket in self._histograms.items():
                if name not in histograms_summary:
                    histograms_summary[name] = bucket.get_stats()

            # Gauges
            gauges_summary = dict(self._gauges)
            for name, label_values in self._gauges_with_labels.items():
                gauges_summary[name] = {str(dict(k)): v for k, v in label_values.items()}

            return {
                "uptime_seconds": uptime,
                "counters": counters_summary,
                "histograms": histograms_summary,
                "gauges": gauges_summary,
            }

    def get_prometheus_format(self) -> str:
        """
        获取 Prometheus 格式的指标输出。

        用于 /metrics 端点。
        """
        with self._lock:
            lines = []
            uptime = time.time() - self._start_time

            # Help 和 Type 注释
            lines.append('# HELP uptime_seconds Time since metrics initialization')
            lines.append('# TYPE uptime_seconds gauge')
            lines.append(f'uptime_seconds {uptime}')

            # Counters
            for name, value in self._counters.items():
                safe_name = name.replace(".", "_").replace("-", "_")
                lines.append(f'# HELP {safe_name} Counter')
                lines.append(f'# TYPE {safe_name} counter')
                lines.append(f'{safe_name} {value}')

            # Counters with labels
            for name, label_values in self._counters_with_labels.items():
                safe_name = name.replace(".", "_").replace("-", "_")
                lines.append(f'# HELP {safe_name} Counter with labels')
                lines.append(f'# TYPE {safe_name} counter')
                for labels_tuple, value in label_values.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
                    lines.append(f'{safe_name}{{{label_str}}} {value}')

            # Histograms
            for name, bucket in self._histograms.items():
                safe_name = name.replace(".", "_").replace("-", "_")
                if not bucket.values:
                    continue

                stats = bucket.get_stats()
                lines.append(f'# HELP {safe_name} Histogram')
                lines.append(f'# TYPE {safe_name} histogram')

                # Sum 和 Count
                lines.append(f'{safe_name}_sum {stats["sum"]}')
                lines.append(f'{safe_name}_count {stats["count"]}')

                # Percentiles
                for p, value in stats.items():
                    if p.startswith("p"):
                        lines.append(f'{safe_name}{{"quantile="{p[1:]}"}} {value}')

            # Gauges
            for name, value in self._gauges.items():
                safe_name = name.replace(".", "_").replace("-", "_")
                lines.append(f'# HELP {safe_name} Gauge')
                lines.append(f'# TYPE {safe_name} gauge')
                lines.append(f'{safe_name} {value}')

            return "\n".join(lines)

    def clear(self):
        """清空所有指标。"""
        with self._lock:
            self._counters.clear()
            self._counters_with_labels.clear()
            self._histograms.clear()
            self._histograms_with_labels.clear()
            self._gauges.clear()
            self._gauges_with_labels.clear()
            self._start_time = time.time()


# 全局单例
metrics = MetricsCollector()
