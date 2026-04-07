"""
Main entry point for the contestant inference service.
Starts:
1. vLLM engine (background, port 8000)
2. HTTP server (port 9000, health check)
3. Scheduler client loop (background thread)
"""

import os
import sys
import asyncio
import signal
import threading
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.client import main_loop
from src.inference import load_config


# ── HTTP Server (port 9000) ────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    """Simple health check handler."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


def start_http_server(port: int):
    """Start the HTTP server on the given port."""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[Main] HTTP server listening on port {port}")
    server.serve_forever()


def start_vllm_background():
    """
    Start vLLM engine as a subprocess and wait for it to be ready.
    Returns the subprocess object.
    """
    import subprocess

    MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/model/Qwen3-32B")
    VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))

    # Find Python
    for py in ["python3.12", "python3.11", "python3"]:
        import shutil
        if shutil.which(py):
            PYTHON_BIN = py
            break
    else:
        PYTHON_BIN = "python3"

    vllm_cmd = [
        PYTHON_BIN, "-m", "vllm.entrypoints.api_server",
        "--model", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.9",
        "--enforce-eager",
    ]

    print(f"[Main] Starting vLLM: {' '.join(vllm_cmd)}")
    proc = subprocess.Popen(
        vllm_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for vLLM to be ready
    import httpx
    vllm_url = f"http://localhost:{VLLM_PORT}"
    for i in range(60):
        try:
            resp = httpx.get(f"{vllm_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                print("[Main] vLLM engine is ready!")
                return proc
        except Exception:
            pass
        time.sleep(2)

    print("[Main] WARNING: vLLM did not become ready in time, continuing anyway")
    return proc


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contestant Inference Service")
    parser.add_argument("--port", type=int, default=None, help="HTTP server port (default from CONTESTANT_PORT env)")
    parser.add_argument("--token", default=None, help="Contestant token")
    parser.add_argument("--name", default=None, help="Team name")
    parser.add_argument("--platform-url", default=None, help="Platform URL")
    parser.add_argument("--no-vllm", action="store_true", help="Skip vLLM startup (assumes already running)")
    parser.add_argument("--model-path", default=None, help="Model path")
    parser.add_argument("--config-path", default=None, help="Config file path")
    args = parser.parse_args()

    # Override env vars if provided
    if args.port:
        os.environ["CONTESTANT_PORT"] = str(args.port)
    if args.token:
        os.environ["CONTESTANT_TOKEN"] = args.token
    if args.name:
        os.environ["CONTESTANT_NAME"] = args.name
    if args.platform_url:
        os.environ["PLATFORM_URL"] = args.platform_url
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.config_path:
        os.environ["CONFIG_PATH"] = args.config_path

    # Load config for inference module
    load_config()

    # Determine HTTP port
    HTTP_PORT = int(os.environ.get("CONTESTANT_PORT", "9000"))

    # Start vLLM (unless skipped)
    if not args.no_vllm:
        start_vllm_background()
    else:
        print("[Main] Skipping vLLM startup (--no-vllm)")

    # Start HTTP server in background thread
    http_thread = threading.Thread(
        target=start_http_server,
        args=(HTTP_PORT,),
        daemon=True,
    )
    http_thread.start()
    print(f"[Main] HTTP server started on port {HTTP_PORT}")

    # Give HTTP server a moment to bind
    time.sleep(0.5)

    # Start client loop in main thread (it runs forever)
    print("[Main] Starting client loop...")
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[Main] Interrupted, shutting down...")
    except Exception as e:
        print(f"[Main] Client loop error: {e}")
        raise


if __name__ == "__main__":
    main()
