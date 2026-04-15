"""
Basic tests for the inference service.
Run with: python -m tests.test_basic
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import generate_text, compute_logprob, compute_rolling_logprob


def test_vllm_connection():
    """Test if vLLM is reachable."""
    try:
        import httpx
        resp = httpx.get("http://localhost:8000/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            print(f"[Test] vLLM connected. Models: {models}")
            return True
    except Exception as e:
        print(f"[Test] vLLM not reachable: {e}")
    return False


def test_generation():
    """Test text generation."""
    try:
        result = generate_text("The capital of France is", {"max_gen_toks": 20})
        print(f"[Test] Generated: {result}")
        return bool(result)
    except Exception as e:
        print(f"[Test] Generation failed: {e}")
        return False


def test_logprob():
    """Test logprob calculation."""
    try:
        logp = compute_logprob("The capital of France is", " Paris")
        print(f"[Test] Logprob: {logp}")
        return isinstance(logp, float)
    except Exception as e:
        print(f"[Test] Logprob failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Running basic tests...")
    print("=" * 50)

    results = {
        "vLLM Connection": test_vllm_connection(),
        "Generation": test_generation(),
        "Logprob": test_logprob(),
    }

    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 50)

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
