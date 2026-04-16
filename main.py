"""
竞赛推理服务主入口。
启动三个组件:
1. vLLM 推理引擎 (后台运行, 端口 8000)
2. HTTP 服务器 (端口 9000, 健康检查)
3. 调度客户端循环 (后台线程)

Q&A 约束:
- Q43: 使用 Docker 环境 (nvcr.io/nvidia/pytorch:25.11-py3)
- Q58: setup.sh 可构建 CUDA Graph
"""

import os
import sys
import asyncio
import signal
import threading
import time
import argparse
import httpx
import subprocess
import shutil
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.client import main_loop
from src.inference import load_config, METRICS, close_vllm_client, warmup_model

# 全局变量
_vllm_proc = None
_shutdown_requested = False

os.environ.setdefault('OMP_NUM_THREADS', '1')


# ── 配置验证 ─────────────────────────────────────────────────────────────

def validate_config():
    errors = []

    # TOKEN 验证
    TOKEN = os.environ.get("TEAM_TOKEN", "")
    if not TOKEN:
        errors.append("TEAM_TOKEN 未设置")

    # MODEL_PATH 验证
    MODEL_PATH = os.environ.get("MODEL_PATH", "")
    if not MODEL_PATH:
        errors.append("MODEL_PATH 未设置")
    elif not os.path.exists(MODEL_PATH):
        errors.append(f"MODEL_PATH 不存在: {MODEL_PATH}")

    # 端口验证
    for name in ["CONTESTANT_PORT", "VLLM_PORT"]:
        port_str = os.environ.get(name, "")
        try:
            port = int(port_str) if port_str else 9000 if name == "CONTESTANT_PORT" else 8000
            if not (1 <= port <= 65535):
                errors.append(f"无效端口 {name}={port}")
        except ValueError:
            errors.append(f"{name} 必须是整数: {port_str}")

    # Platform URL 验证
    PLATFORM_URL = os.environ.get("PLATFORM_URL", "")
    if not PLATFORM_URL:
        errors.append("PLATFORM_URL 未设置")

    if errors:
        raise ValueError(f"配置错误:\n" + "\n".join(f"  - {e}" for e in errors))

    print("[Config] 配置验证通过")


# ── 信号处理 ─────────────────────────────────────────────────────────────

def setup_signal_handlers():
    """设置信号处理器。"""

    def sigterm_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        print("\n[Main] 收到 SIGTERM,准备优雅关闭...")

        # 关闭 vLLM
        cleanup_vllm()

    def sigint_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        print("\n[Main] 收到 Ctrl+C,准备关闭...")
        cleanup_vllm()

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    # 忽略 SIGCHLD (子进程退出信号)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


# ── vLLM 进程管理 ───────────────────────────────────────────────────────

def cleanup_vllm():
    """清理 vLLM 进程及其子进程。"""
    global _vllm_proc
    if _vllm_proc is None:
        return

    try:
        # 使用进程组杀死整个进程树
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(_vllm_proc.pid), signal.SIGTERM)
                print("[Main] vLLM 进程组已终止")
            except ProcessLookupError:
                pass
        else:
            _vllm_proc.terminate()

        # 等待进程结束
        try:
            _vllm_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _vllm_proc.kill()
            print("[Main] vLLM 进程已强制杀死")
    except Exception as e:
        print(f"[Main] 清理 vLLM 进程失败: {e}")
    finally:
        _vllm_proc = None


def start_http_server(port: int):
    """在指定端口启动 HTTP 服务器。"""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[Main] HTTP 服务器已启动, 监听端口 {port}")
    server.serve_forever()


def start_vllm_background():
    """
    后台启动 vLLM 推理引擎并等待其就绪。
    返回子进程对象。
    """
    global _vllm_proc

    MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/model/Qwen3-32B")
    VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))

    # 检测 GPU 数量,用于 tensor parallelism
    num_gpus = 1
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split("\n"))
    except Exception:
        pass

    # 查找可用的 Python
    for py in ["python3.12", "python3.11", "python3"]:
        if shutil.which(py):
            PYTHON_BIN = py
            break
    else:
        PYTHON_BIN = "python3"

    # vLLM 启动命令,包含优化参数
    vllm_cmd = [
        PYTHON_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.9",
        # 性能优化参数
        "--max-model-len", "8192",  # 限制上下文长度以节省显存
        "--enable-prefix-caching",  # 前缀缓存,减少重复计算
        "--no-enable-log-requests",  # 减少日志开销
    ]

    # speculative decoding 支持 (Q33: 允许使用小模型进行投机解码)
    SPECULATIVE_MODEL = os.environ.get("SPECULATIVE_MODEL", "")
    if SPECULATIVE_MODEL:
        vllm_cmd.extend([
            "--speculative-model", SPECULATIVE_MODEL,
            "--num-speculative-tokens", os.environ.get("SPECULATIVE_DRAFT_TOKENS", "4"),
        ])
        print(f"[Main] 启用投机解码: {SPECULATIVE_MODEL}")

    # 多 GPU 时启用 tensor parallelism
    if num_gpus > 1:
        tp_size = min(num_gpus, 2)  # Qwen3-32B 最多用 2 卡
        vllm_cmd.extend(["--tensor-parallel-size", str(tp_size)])
        print(f"[Main] 使用 tensor parallelism, {tp_size} 张 GPU")

    print(f"[Main] 启动 vLLM: {' '.join(vllm_cmd)}")

    # 创建新进程组
    proc = subprocess.Popen(
        vllm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    _vllm_proc = proc

    # 等待 vLLM 就绪,最多 55 秒
    vllm_url = f"http://localhost:{VLLM_PORT}"
    start_time = time.time()
    check_interval = 1.0  # 开始时快速检查,每秒一次
    max_wait = 55  # 留出 buffer 给 HTTP 服务器启动

    print(f"[Main] 等待 vLLM 就绪 (最多 {max_wait}s)...")
    for i in range(100):  # 最多迭代次数
        elapsed = time.time() - start_time
        if elapsed >= max_wait:
            print(f"[Main] 警告: vLLM 在 {max_wait}s 内未就绪,继续运行")
            return proc

        # 检查进程是否退出
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"[Main] vLLM 进程异常退出: {proc.returncode}")
            if stdout:
                print(f"[Main] stdout: {stdout.decode()[:500]}")
            if stderr:
                print(f"[Main] stderr: {stderr.decode()[:500]}")
            return None

        try:
            resp = httpx.get(f"{vllm_url}/v1/models", timeout=2)
            if resp.status_code == 200:
                print(f"[Main] vLLM 引擎就绪,耗时 {elapsed:.1f}s")
                return proc
        except Exception:
            pass

        # 自适应 sleep: 快超时时间隔更短
        remaining = max_wait - elapsed
        if remaining > 30:
            time.sleep(check_interval)
        elif remaining > 10:
            time.sleep(0.5)
        else:
            time.sleep(0.2)

    print("[Main] 警告: vLLM 未及时就绪,继续运行")
    return proc


# ── HTTP 服务器 (端口 9000) ────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    """健康检查处理器,支持多级健康检查和指标暴露。"""

    def log_message(self, format, *args):
        pass  # 禁用默认日志输出

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._handle_health()
        elif self.path == "/health/live":
            self._handle_liveness()
        elif self.path == "/health/ready":
            self._handle_readiness()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_health(self):
        """综合健康检查。"""
        vllm_ok = self._check_vllm()
        if vllm_ok:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "vllm": "ok"}')
        else:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "unhealthy", "vllm": "error"}')

    def _handle_liveness(self):
        """存活探针。"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "alive"}')

    def _handle_readiness(self):
        """就绪探针,检查 vLLM 是否可用。"""
        vllm_ok = self._check_vllm()
        if vllm_ok:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ready", "vllm": "ok"}')
        else:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "not_ready", "vllm": "error"}')

    def _check_vllm(self) -> bool:
        """检查 vLLM 是否可用。"""
        VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
        try:
            resp = httpx.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _handle_metrics(self):
        """Prometheus 格式的指标。"""
        import json
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        metrics = METRICS.get_stats()
        self.wfile.write(METRICS.get_prometheus_format().encode())


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="竞赛推理服务")
    parser.add_argument("--port", type=int, default=None, help="HTTP 服务器端口 (默认从 CONTESTANT_PORT 环境变量读取)")
    parser.add_argument("--token", default=None, help="参赛 token")
    parser.add_argument("--name", default=None, help="队伍名称")
    parser.add_argument("--platform-url", default=None, help="平台 URL")
    parser.add_argument("--no-vllm", action="store_true", help="跳过 vLLM 启动 (假设已运行)")
    parser.add_argument("--model-path", default=None, help="模型路径")
    parser.add_argument("--config-path", default=None, help="配置文件路径")
    args = parser.parse_args()

    # 设置信号处理器
    setup_signal_handlers()

    # 命令行参数优先于环境变量
    if args.port:
        os.environ["CONTESTANT_PORT"] = str(args.port)
    if args.token:
        os.environ["TEAM_TOKEN"] = args.token
    if args.name:
        os.environ["TEAM_NAME"] = args.name
    if args.platform_url:
        os.environ["PLATFORM_URL"] = args.platform_url
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.config_path:
        os.environ["CONFIG_PATH"] = args.config_path

    # 验证配置
    try:
        validate_config()
    except ValueError as e:
        print(f"[Main] {e}")
        sys.exit(1)

    # 加载推理模块配置
    load_config()

    # 确定 HTTP 端口
    HTTP_PORT = int(os.environ.get("CONTESTANT_PORT", "9000"))

    # 启动 vLLM (除非使用 --no-vllm 跳过)
    if not args.no_vllm:
        proc = start_vllm_background()
        # 等待 vLLM 就绪后预热模型
        if proc:
            print("[Main] 正在预热模型 (构建 CUDA Graph)...")
            warmup_model()
    else:
        print("[Main] 跳过 vLLM 启动 (--no-vllm)")
        # 即使跳过 vLLM 启动,也尝试预热
        print("[Main] 尝试预热已有 vLLM...")
        warmup_model()

    # 在后台线程中启动 HTTP 服务器
    http_thread = threading.Thread(
        target=start_http_server,
        args=(HTTP_PORT,),
        daemon=True,
    )
    http_thread.start()
    print(f"[Main] HTTP 服务器已启动,端口 {HTTP_PORT}")

    # 等待 HTTP 服务器绑定端口
    time.sleep(0.5)

    # 在主线程中启动客户端循环 (会一直运行)
    print("[Main] 启动客户端循环...")

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[Main] 收到中断信号,正在关闭...")
    except Exception as e:
        print(f"[Main] 客户端循环错误: {e}")
        traceback.print_exc()
        raise
    finally:
        close_vllm_client()
        cleanup_vllm()
        print("[Main] 服务已关闭")


if __name__ == "__main__":
    main()
