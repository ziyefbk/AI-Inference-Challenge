"""
AI Inference Challenge - 竞赛推理服务主入口。

启动三个组件:
1. vLLM 推理引擎 (后台进程, 端口 8000+)
2. HTTP 健康检查服务器 (端口 9000)
3. 调度客户端循环 (后台线程)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.client.loop import main_loop
from src.inference import load_config, METRICS, close_vllm_client, warmup_model
from src.utils.logger import setup_logger, get_logger

setup_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger("main")

# 全局变量
_vllm_procs: list = []
_shutdown_requested = False

os.environ.setdefault("OMP_NUM_THREADS", "1")


# ── 配置验证 ─────────────────────────────────────────────────────────────

def validate_config() -> None:
    """验证必要配置。"""
    errors = []
    TOKEN = os.environ.get("TEAM_TOKEN", "")
    MODEL_PATH = os.environ.get("MODEL_PATH", "")
    PLATFORM_URL = os.environ.get("PLATFORM_URL", "")

    if not TOKEN:
        errors.append("TEAM_TOKEN 未设置")
    if not MODEL_PATH:
        errors.append("MODEL_PATH 未设置")
    elif not os.path.exists(MODEL_PATH):
        errors.append(f"MODEL_PATH 不存在: {MODEL_PATH}")
    if not PLATFORM_URL:
        errors.append("PLATFORM_URL 未设置")

    if errors:
        raise ValueError("配置错误:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info("配置验证通过")


# ── 信号处理 ─────────────────────────────────────────────────────────────

def setup_signal_handlers() -> None:
    """设置信号处理器。"""

    def _sigterm(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("收到 SIGTERM, 准备优雅关闭")
        cleanup_vllm()

    def _sigint(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("收到 Ctrl+C, 准备关闭")
        cleanup_vllm()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


# ── vLLM 进程管理 ───────────────────────────────────────────────────────

def cleanup_vllm() -> None:
    """清理所有 vLLM 进程。"""
    global _vllm_procs
    for i, port, proc in _vllm_procs:
        try:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            logger.error(f"清理 vLLM 进程 {i} 失败: {e}")
    _vllm_procs = []
    logger.info("vLLM 进程组已终止")


def start_vllm_background():
    """
    后台启动一个或多个 vLLM 推理引擎并等待其就绪。
    返回 (实例索引, 端口, 进程) 元组列表。
    """
    global _vllm_procs

    MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-0.5B")

    # 检测 GPU
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

    # 查找 Python
    contestant_python = "/tmp/contestant_env/bin/python3.12"
    if os.path.exists(contestant_python):
        PYTHON_BIN = contestant_python
    else:
        for py in ["python3.12", "python3.11", "python3"]:
            if shutil.which(py):
                PYTHON_BIN = py
                break
        else:
            PYTHON_BIN = "python3"

    num_instances = max(1, num_gpus)
    logger.info(f"检测到 {num_gpus} 张 GPU, 启动 {num_instances} 个 vLLM 实例")

    SPECULATIVE_MODEL = os.environ.get("SPECULATIVE_MODEL", "")
    os.environ["VLLM_NUM_INSTANCES"] = str(num_instances)

    all_procs = []
    for i in range(num_instances):
        port = 8000 + i
        gpu_id = i if i < num_gpus else 0

        vllm_cmd = [
            PYTHON_BIN, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--port", str(port),
            "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1",
            "--enable-prefix-caching",
            "--disable-log-stats",
            "--disable-uvicorn-access-log",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "8192",
            "--max-num-seqs", "256",
            "--ubatch-size", "256",
        ]

        if SPECULATIVE_MODEL:
            vllm_cmd.extend([
                "--speculative-model", SPECULATIVE_MODEL,
                "--num-speculative-tokens", os.environ.get("SPECULATIVE_DRAFT_TOKENS", "4"),
            ])
            logger.info(f"实例 {i} 启用投机解码: {SPECULATIVE_MODEL}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=env,
        )
        all_procs.append((i, port, proc))

    _vllm_procs = all_procs

    # 等待就绪
    start_time = time.time()
    max_wait = 55
    logger.info(f"等待 vLLM 实例就绪 (最多 {max_wait}s)...")

    ready_count = 0
    for i, port, proc in all_procs:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                logger.warning(f"vLLM 实例 {i} 在 {max_wait}s 内未就绪, 继续运行")
                break

            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                logger.error(f"vLLM 实例 {i} 进程异常退出: {proc.returncode}")
                if stdout:
                    logger.error(f"stdout: {stdout.decode()[:500]}")
                if stderr:
                    logger.error(f"stderr: {stderr.decode()[:500]}")
                break

            try:
                resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=2)
                if resp.status_code == 200:
                    logger.info(f"vLLM 实例 {i} 就绪 (端口 {port}), 耗时 {elapsed:.1f}s")
                    ready_count += 1
                    break
            except Exception:
                pass

            remaining = max_wait - elapsed
            time.sleep(1.0 if remaining > 30 else 0.5 if remaining > 10 else 0.2)

    logger.info(f"{ready_count}/{num_instances} 个 vLLM 实例已就绪")
    return all_procs


# ── HTTP 服务器 ──────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    """健康检查处理器。"""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._handle_health()
        elif self.path == "/health/live":
            self._send_json(200, {"status": "alive"})
        elif self.path == "/health/ready":
            self._handle_readiness()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        import json
        self.wfile.write(json.dumps(data).encode())

    def _handle_health(self):
        if self._check_vllm():
            self._send_json(200, {"status": "healthy", "vllm": "ok"})
        else:
            self._send_json(503, {"status": "unhealthy", "vllm": "error"})

    def _handle_readiness(self):
        if self._check_vllm():
            self._send_json(200, {"status": "ready", "vllm": "ok"})
        else:
            self._send_json(503, {"status": "not_ready", "vllm": "error"})

    def _check_vllm(self) -> bool:
        num_instances = int(os.environ.get("VLLM_NUM_INSTANCES", "1"))
        for i in range(num_instances):
            port = 8000 + i
            try:
                resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def _handle_metrics(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(METRICS.get_prometheus_format().encode())


def start_http_server(port: int) -> None:
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"HTTP 服务器已启动, 监听端口 {port}")
    server.serve_forever()


# ── 主函数 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="竞赛推理服务")
    parser.add_argument("--port", type=int, default=None, help="HTTP 服务器端口")
    parser.add_argument("--token", default=None, help="参赛 token")
    parser.add_argument("--name", default=None, help="队伍名称")
    parser.add_argument("--platform-url", default=None, help="平台 URL")
    parser.add_argument("--no-vllm", action="store_true", help="跳过 vLLM 启动")
    parser.add_argument("--model-path", default=None, help="模型路径")
    parser.add_argument("--config-path", default=None, help="配置文件路径")
    args = parser.parse_args()

    setup_signal_handlers()

    # 命令行参数覆盖环境变量
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

    try:
        validate_config()
    except ValueError as e:
        print(f"[Main] {e}")
        sys.exit(1)

    load_config()

    HTTP_PORT = int(os.environ.get("CONTESTANT_PORT", "9000"))

    if not args.no_vllm:
        procs = start_vllm_background()
        if procs:
            logger.info("正在预热模型 (构建 CUDA Graph)...")
            warmup_model()
    else:
        logger.info("跳过 vLLM 启动 (--no-vllm)")
        warmup_model()

    http_thread = threading.Thread(target=start_http_server, args=(HTTP_PORT,), daemon=True)
    http_thread.start()
    logger.info(f"HTTP 服务器已启动, 端口 {HTTP_PORT}")

    time.sleep(0.5)
    logger.info("启动客户端循环...")

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("收到中断信号, 正在关闭...")
    except Exception as e:
        logger.error(f"客户端循环错误: {e}")
        traceback.print_exc()
        raise
    finally:
        close_vllm_client()
        cleanup_vllm()
        logger.info("服务已关闭")


if __name__ == "__main__":
    main()
