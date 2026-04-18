#!/usr/bin/env python3
"""
手动测试 vLLM completions API，模拟 generate_until 请求。
"""

import os
import sys
import time
import subprocess
import httpx

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-0.5B")
VLLM_PORT = 8000
BASE_URL = f"http://localhost:{VLLM_PORT}"


def start_vllm():
    print("=== 启动 vLLM ===")
    python_bin = "/tmp/contestant_env/bin/python3.12"
    if not os.path.exists(python_bin):
        python_bin = "python3"

    cmd = [
        python_bin, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size", "1",
        "--enable-prefix-caching",
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
        "--max-num-seqs", "256",
    ]
    print(f"CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_until_ready(timeout=60):
    print(f"=== 等待 vLLM 就绪 (最多 {timeout}s) ===")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                print(f"vLLM 就绪! 耗时 {time.time()-start:.1f}s")
                print(f"Models: {resp.json()}")
                return True
        except Exception as e:
            print(f"  等待中... ({e})")
        time.sleep(2)
    return False


def test_completions(prompt, **kwargs):
    print(f"\n=== 测试 completions ===")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Kwargs: {kwargs}")

    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": kwargs.get("max_tokens", 10),
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
    }
    if kwargs.get("logprobs", 0) > 0:
        payload["logprobs"] = kwargs["logprobs"]
        payload["echo"] = kwargs.get("echo", False)
    if kwargs.get("top_k", 0) > 0:
        payload["top_k"] = kwargs["top_k"]
    if kwargs.get("stop"):
        payload["stop"] = kwargs["stop"]
    if kwargs.get("repetition_penalty", 1.0) != 1.0:
        payload["repetition_penalty"] = kwargs["repetition_penalty"]
    if kwargs.get("best_of", 1) > 1:
        payload["best_of"] = kwargs["best_of"]
        payload["n"] = max(kwargs.get("n", 1), kwargs["best_of"])
    elif kwargs.get("n", 1) > 1:
        payload["n"] = kwargs["n"]

    print(f"Payload keys: {list(payload.keys())}")

    try:
        resp = httpx.post(f"{BASE_URL}/v1/completions", json=payload, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Body: {resp.text[:500]}")
        else:
            result = resp.json()
            print(f"Choices: {len(result.get('choices', []))}")
            for i, c in enumerate(result.get("choices", [])):
                text = c.get("text", "")
                print(f"  [{i}] text={text[:100]!r}")
        return resp
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_chat_completions():
    print(f"\n=== 测试 chat/completions ===")
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    try:
        resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Body: {resp.text[:500]}")
        else:
            result = resp.json()
            print(f"Result: {result}")
        return resp
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    proc = start_vllm()

    try:
        if not wait_until_ready(60):
            print("vLLM 启动失败!")
            sys.exit(1)

        # Test 1: 裸 completions（最小参数）
        test_completions("Hello world", max_tokens=5)

        # Test 2: 带 top_k
        test_completions("Hello world", max_tokens=5, top_k=50)

        # Test 3: 带 stop
        test_completions("Hello world", max_tokens=20, stop=["\n"])

        # Test 4: 带 repetition_penalty
        test_completions("Hello world", max_tokens=10, repetition_penalty=1.1)

        # Test 5: generate_until 默认参数（无 best_of）
        test_completions(
            "Answer: 2+2=",
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            stop=["\n\n"],
            repetition_penalty=1.0,
        )

        # Test 6: 带 best_of > 1
        test_completions(
            "Hello",
            max_tokens=5,
            temperature=0.7,
            top_p=0.95,
            best_of=2,
            n=2,
        )

        # Test 7: logprobs + echo（loglikelihood 用的）
        test_completions("Hello world", max_tokens=2, logprobs=1, echo=True)

        # Test 8: chat completions
        test_chat_completions()

    finally:
        print("\n=== 关闭 vLLM ===")
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), 9)
        else:
            proc.terminate()
