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
import io
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.client.loop import main_loop
from src.inference import load_config, METRICS, close_vllm_client, warmup_model
from src.utils.logger import setup_logger, get_logger
from src.utils.process import cleanup_orphaned_enginecores

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

    def _sigint(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("收到 Ctrl+C, 准备关闭")

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


# ── vLLM 进程管理 ───────────────────────────────────────────────────────

def cleanup_vllm() -> None:
    """清理所有 vLLM 进程。

    优先通过 /shutdown 端点优雅关闭（会连带关闭 EngineCore 子进程），
    如果失败再退而用 SIGTERM/SIGKILL。
    """
    global _vllm_procs
    if not _vllm_procs:
        return
    for i, port, proc in _vllm_procs:
        try:
            resp = httpx.post(f"http://localhost:{port}/shutdown", timeout=10)
            logger.info(f"vLLM 实例 {i} (端口 {port}) 通过 /shutdown 端点关闭")
        except Exception as e:
            logger.warning(f"vLLM /shutdown 失败，回退到进程终止: {e}")
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
            except Exception as ce:
                logger.error(f"清理 vLLM 进程 {i} 失败: {ce}")

    _vllm_procs = []
    logger.info("vLLM 进程组已终止")


def _cleanup_orphaned_enginecores() -> None:
    cleanup_orphaned_enginecores(logger)


# 错误关键词分类，便于快速定位启动失败原因
_VLLM_ERROR_PATTERNS = {
    "CUDA OOM": ["out of memory", "OutOfMemoryError", "CUDA out of memory", "out of GPU memory"],
    "模型未找到": ["does not exist", "Model not found", "no such file", "FileNotFoundError"],
    "端口被占用": ["Address already in use", "port is already in use", "OSError: [Errno 98]"],
    "驱动/硬件": ["CUDA error", "NVIDIA driver", "no CUDA-capable device", "CUDA_VISIBLE_DEVICES"],
    "vLLM 参数错误": ["invalid argument", "unexpected keyword argument", "got an unexpected keyword"],
    "权重加载失败": ["Error in loading", "weight file", "safetensors", "Failed to load"],
}


def _classify_vllm_error(stderr_bytes: bytes) -> str:
    text = stderr_bytes.decode(errors="replace").lower()
    for category, keywords in _VLLM_ERROR_PATTERNS.items():
        if any(kw.lower() in text for kw in keywords):
            return category
    return "未知错误"


def _build_vllm_cmd(python_bin: str, model_path: str, port: int, tp_size: int, i: int) -> list:
    """构建单个 vLLM 实例的启动命令。"""
    cmd = [
        python_bin, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", "0.75",
        "--tensor-parallel-size", str(tp_size),
        "--enable-prefix-caching",
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
        "--enforce-eager",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "128",
    ]

    speculative = os.environ.get("SPECULATIVE_MODEL", "")
    if speculative:
        cmd.extend([
            "--speculative-model", speculative,
            "--num-speculative-tokens", os.environ.get("SPECULATIVE_DRAFT_TOKENS", "4"),
        ])
        logger.info(f"实例 {i} 启用投机解码: {speculative}")

    return cmd


def _write_vllm_crash_log(i: int, port: int, stdout: bytes, stderr: bytes) -> str:
    """写 vLLM 崩溃日志到临时文件，返回路径。"""
    import tempfile
    log_path = tempfile.mktemp(suffix=f"_vllm_crash_{port}.log", prefix="/tmp/")
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            if stdout:
                f.write("=== STDOUT ===\n")
                f.write(stdout.decode(errors="replace"))
            f.write("\n=== STDERR ===\n")
            if stderr:
                f.write(stderr.decode(errors="replace"))
    except Exception:
        log_path = f"/tmp/vllm_crash_{port}_(write_failed).log"
    return log_path


class _VLLMCapture:
    """
    实时捕获 vLLM 子进程的 stderr。
    用独立线程持续读管道，避免管道填满导致子进程阻塞或退出。
    崩溃后调用 get_output() 获取完整 stderr 用于分类错误。
    """

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._stderr_lines: List[str] = []
        self._done = threading.Event()
        self._t = threading.Thread(target=self._read_stderr, daemon=True)
        self._t.start()

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            self._done.set()
            return
        try:
            for line in io.TextIOWrapper(self._proc.stderr, errors="replace"):
                self._stderr_lines.append(line.rstrip())
        except Exception:
            pass
        self._done.set()

    def get_output(self) -> bytes:
        if self._proc.poll() is not None and self._proc.stderr is not None:
            try:
                remaining = self._proc.stderr.read()
                if remaining:
                    self._stderr_lines.append(remaining.decode(errors="replace").rstrip())
            except Exception:
                pass
        return "\n".join(self._stderr_lines).encode(errors="replace")

    def is_done(self) -> bool:
        return self._done.is_set()

    def join(self, timeout: float = 0.0) -> None:
        self._t.join(timeout)


def _wait_for_instance(
    i: int, port: int, proc,
    start_time: float,
    max_wait: float,
    restart_count: int,
    restart_max: int,
    python_bin: str, model_path: str, tp_size: int,
) -> bool:
    """
    等待单个 vLLM 实例就绪，支持崩溃后自动重启。
    用实时 stderr 捕获避免进程崩溃后丢失错误信息。
    HTTP 200 后发一个实际推理探测请求确认模型真正可用。
    返回该实例是否最终就绪。
    """
    capture = _VLLMCapture(proc)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_wait:
            logger.warning(
                f"vLLM 实例 {i} 在 {max_wait}s 内未就绪"
                f"{'（已重启' + str(restart_count) + '次）' if restart_count else ''}，跳过"
            )
            return False

        poll = proc.poll()
        if poll is not None:
            stderr_bytes = capture.get_output()
            category = _classify_vllm_error(stderr_bytes)
            logger.error(f"vLLM 实例 {i} 进程崩溃 (exit={poll}, 类型={category})")
            log_path = _write_vllm_crash_log(i, port, b"", stderr_bytes)
            logger.error(f"查看崩溃日志: cat {log_path}")

            if restart_count < restart_max:
                logger.info(f"尝试重启 vLLM 实例 {i} (第 {restart_count + 1}/{restart_max} 次)...")
                _cleanup_orphaned_enginecores()
                time.sleep(3)
                cmd = _build_vllm_cmd(python_bin, model_path, port, tp_size, i)
                env = os.environ.copy()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                    env=env,
                )
                capture = _VLLMCapture(proc)
                restart_count += 1
                continue
            else:
                logger.error(f"vLLM 实例 {i} 已重启 {restart_max} 次仍失败，放弃")
                return False

        try:
            resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=2)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
        except Exception:
            time.sleep(0.5)
            continue

        # HTTP 200 后，发一个实际推理探测确认模型真正可用
        # 用 /v1/completions 而非 /v1/chat/completions，因为后者可能不存在
        probe_ok = False
        for probe_attempt in range(10):
            if proc.poll() is not None:
                stderr_bytes = capture.get_output()
                category = _classify_vllm_error(stderr_bytes)
                logger.error(f"vLLM 实例 {i} 进程在推理探测时崩溃 (exit={proc.poll()}, 类型={category})")
                log_path = _write_vllm_crash_log(i, port, b"", stderr_bytes)
                logger.error(f"查看崩溃日志: cat {log_path}")
                break
            try:
                probe_resp = httpx.post(
                    f"http://localhost:{port}/v1/completions",
                    json={
                        "model": model_path,
                        "prompt": "hi",
                        "max_tokens": 2,
                        "temperature": 0.0,
                    },
                    timeout=15,
                )
                if 200 <= probe_resp.status_code < 300:
                    probe_ok = True
                    break
                elif probe_resp.status_code >= 500:
                    logger.warning(
                        f"vLLM 实例 {i} 推理探测返回 {probe_resp.status_code}，等待 2s 后重试 ({probe_attempt + 1}/10)"
                    )
            except Exception:
                pass
            time.sleep(2)

        if probe_ok:
            logger.info(
                f"vLLM 实例 {i} 推理探测通过 (端口 {port})"
                f"{'（第 ' + str(restart_count) + ' 次启动）' if restart_count else ''}"
            )
            return True

        # 探测失败（500 或崩溃），清理并重启
        stderr_bytes = capture.get_output()
        category = _classify_vllm_error(stderr_bytes)
        logger.error(f"vLLM 实例 {i} 推理探测失败 (HTTP 非 2xx 或崩溃, 类型={category})")
        log_path = _write_vllm_crash_log(i, port, b"", stderr_bytes)
        logger.error(f"查看崩溃日志: cat {log_path}")

        if restart_count < restart_max:
            logger.info(f"尝试重启 vLLM 实例 {i} (第 {restart_count + 1}/{restart_max} 次)...")
            _cleanup_orphaned_enginecores()
            time.sleep(3)
            cmd = _build_vllm_cmd(python_bin, model_path, port, tp_size, i)
            env = os.environ.copy()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=env,
            )
            capture = _VLLMCapture(proc)
            restart_count += 1
            continue
        else:
            logger.error(f"vLLM 实例 {i} 已重启 {restart_max} 次仍失败，放弃")
            return False


def start_vllm_background():
    """
    后台启动一个或多个 vLLM 推理引擎并等待其就绪。
    每个实例崩溃后最多自动重启 RESTART_MAX 次。
    返回 (实例索引, 端口, 进程) 元组列表，仅包含仍存活进程的条目。
    """
    global _vllm_procs

    _cleanup_orphaned_enginecores()

    model_path = os.environ["MODEL_PATH"]

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

    contestant_python = "/tmp/contestant_env/bin/python3.12"
    python_bin = contestant_python if os.path.exists(contestant_python) else "python3"
    for py in ["python3.12", "python3.11", "python3"]:
        if shutil.which(py):
            python_bin = py
            break

    num_instances = 1
    tp_size = num_gpus
    restart_max = 10
    max_wait = 90

    logger.info(
        f"检测到 {num_gpus} 张 GPU，启动 {num_instances} 个 vLLM 实例"
        f"（每个最多重启 {restart_max} 次，等待上限 {max_wait}s）"
    )

    all_procs = []
    for i in range(num_instances):
        port = 8000
        cmd = _build_vllm_cmd(python_bin, model_path, port, tp_size, i)
        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=env,
        )
        all_procs.append((i, port, proc))

    _vllm_procs = all_procs

    start_time = time.time()
    logger.info(f"等待 vLLM 实例就绪（最多 {max_wait}s）...")

    ready_procs = []
    for i, port, proc in all_procs:
        ready = _wait_for_instance(
            i, port, proc, start_time, max_wait,
            restart_count=0, restart_max=restart_max,
            python_bin=python_bin, model_path=model_path, tp_size=tp_size,
        )
        if ready:
            ready_procs.append((i, port, proc))

    _vllm_procs = ready_procs

    if not ready_procs:
        logger.error("没有任何 vLLM 实例就绪，拒绝继续启动。请检查日志中 vLLM 崩溃原因。")
        raise RuntimeError(
            "vLLM 启动失败（所有实例崩溃）。"
            "常见原因：CUDA OOM、模型路径错误、端口被占用。请运行: cat /tmp/vllm_crash_*.log"
        )

    logger.info(f"{len(ready_procs)}/{num_instances} 个 vLLM 实例已就绪")
    return ready_procs


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
        try:
            resp = httpx.get("http://localhost:8000/v1/models", timeout=2)
            return resp.status_code == 200
        except Exception:
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

    HTTP_PORT = int(os.environ["CONTESTANT_PORT"])

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
    finally:
        try:
            close_vllm_client()
        except Exception as e:
            logger.warning(f"关闭 vLLM 客户端时出错: {e}")
        try:
            cleanup_vllm()
        except Exception as e:
            logger.warning(f"清理 vLLM 时出错: {e}")
        logger.info("服务已关闭")


if __name__ == "__main__":
    main()
