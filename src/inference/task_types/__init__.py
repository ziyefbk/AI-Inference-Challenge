"""
任务类型处理器。

包含:
- generate_until: 文本生成任务
- loglikelihood: 条件概率计算任务
- loglikelihood_rolling: 滚动概率计算任务
"""

from src.inference.task_types.generate_until import process_generate_until
from src.inference.task_types.loglikelihood import (
    process_loglikelihood,
    process_loglikelihood_rolling,
    compute_logprob,
    compute_rolling_logprob,
)

__all__ = [
    "process_generate_until",
    "process_loglikelihood",
    "process_loglikelihood_rolling",
    "compute_logprob",
    "compute_rolling_logprob",
]
