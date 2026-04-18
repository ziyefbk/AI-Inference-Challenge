"""
跨平台进程管理工具。

目前用于清理 vLLM EngineCore 孤儿进程。
当 vLLM 进程被强制杀死时，其 EngineCore 子进程会变成孤儿继续占显存，
启动新实例前需要清理这些残留进程。
"""

import os
import signal
import subprocess
import time
from typing import List


def cleanup_orphaned_enginecores(logger=None) -> List[int]:
    """
    杀掉残留的 VLLM EngineCore 孤儿进程，返回被清理的 PID 列表。

    Args:
        logger: 可选日志器，传入时会在清理前后输出日志。

    Returns:
        被成功发送 SIGKILL 的进程 PID 列表。
    """
    killed: List[int] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return killed

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0].strip())
            except ValueError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass

        if killed:
            time.sleep(2)
            if logger:
                logger.info(f"已清理残留 EngineCore 进程: {killed}")
    except Exception as e:
        if logger:
            logger.warning(f"清理残留 EngineCore 进程失败: {e}")
    return killed
