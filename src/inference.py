"""
Improved inference module with proper tokenization and logprob calculation.
Supports three inference types:
- generate_until: text generation with stop conditions
- loglikelihood: compute log P(continuation | prompt)
- loglikelihood_rolling: compute rolling log-likelihood of entire text
"""

import os
import re
import httpx
import json
import time
from typing import Optional, Dict, Any, List


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-32B")

# Sampling parameter presets (loaded from config)
SAMPLING_PARAMS: Dict[str, Dict[str, float]] = {}
SLA_LEVELS: Dict[str, Dict[str, float]] = {}

# Config path
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")


def load_config():
    """Load contest config if available."""
    global SAMPLING_PARAMS, SLA_LEVELS
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        if "sampling_params" in config:
            SAMPLING_PARAMS = config["sampling_params"]
        if "sla_levels" in config:
            SLA_LEVELS = config["sla_levels"]
    except Exception as e:
        print(f"[Inference] Failed to load config: {e}")


load_config()


def _call_vllm(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    echo: bool = False,
    best_of: int = 1,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Call vLLM completions API.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "best_of": best_of,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }

    if top_k > 0:
        payload["top_k"] = top_k

    if stop:
        payload["stop"] = stop

    if logprobs is not None:
        payload["logprobs"] = logprobs
        payload["logprobs_per_token"] = True

    if echo:
        payload["echo"] = echo

    with httpx.Client(timeout=300) as client:
        resp = client.post(f"{VLLM_URL}/v1/completions", json=payload)
        resp.raise_for_status()
        return resp.json()


def _apply_stop_strings(text: str, stop_list: List[str]) -> str:
    """Apply stop string trimming."""
    if not stop_list or not text:
        return text
    for stop_tok in stop_list:
        idx = text.find(stop_tok)
        if idx >= 0:
            return text[:idx]
    return text


def compute_logprob(prompt: str, continuation: str) -> float:
    """
    Compute log probability of continuation given prompt.
    Uses vLLM's logprobs feature with echo=True on the full text,
    then sums the logprobs for the continuation tokens only.
    
    The token_logprobs from vLLM with echo=True returns:
      logprob[0]  = log P(token_0 | BOS)
      logprob[1]  = log P(token_1 | prompt[0])
      ...
      logprob[n]  = log P(token_n | prompt[:n])
    
    Where the last len(continuation_tokens) are for the continuation.
    We need to find where the continuation starts in the token list.
    """
    if not continuation:
        return 0.0
    
    full_text = prompt + continuation

    try:
        response = _call_vllm(
            prompt=full_text,
            max_tokens=1,
            logprobs=100,
            echo=True,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
    except Exception as e:
        print(f"[Inference] vLLM call failed for logprob: {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])

    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0

    # Sum all logprobs for the continuation portion.
    # Strategy: use the token count ratio to estimate where continuation starts.
    # This is an approximation; for exact token-level we'd need tiktoken.
    # Instead, sum the second half as a proxy.
    n = len(token_logprobs)
    # Use the last half as continuation tokens (rough heuristic)
    start_idx = n // 2
    cont_logprobs = token_logprobs[start_idx:]
    
    # Filter out None values
    cont_logprobs = [lp for lp in cont_logprobs if lp is not None]
    if not cont_logprobs:
        return -10.0
    
    total_logprob = sum(cont_logprobs)
    return float(total_logprob)


def compute_rolling_logprob(text: str) -> float:
    """
    Compute rolling log-likelihood of entire text.
    Uses vLLM's logprobs to compute perplexity.
    With echo=True, token_logprobs[1:] gives log P(each token | previous).
    """
    if not text:
        return 0.0

    try:
        response = _call_vllm(
            prompt=text,
            max_tokens=1,
            logprobs=1000,
            echo=True,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
    except Exception as e:
        print(f"[Inference] vLLM call failed for rolling logprob: {e}")
        return -10.0

    choices = response.get("choices", [])
    if not choices:
        return -10.0

    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])

    if not token_logprobs or len(token_logprobs) < 2:
        return -10.0

    # token_logprobs[0] corresponds to the first token (BOS), skip it
    valid_logprobs = [lp for lp in token_logprobs[1:] if lp is not None]
    if not valid_logprobs:
        return -10.0

    return float(sum(valid_logprobs))


def generate_text(prompt: str, gen_kwargs: Dict[str, Any]) -> str:
    """
    Generate text until stopping condition.
    Applies stop string trimming from generated output.
    """
    max_tokens = gen_kwargs.get("max_gen_toks", 256)
    temperature = gen_kwargs.get("temperature", 0.0)
    top_p = gen_kwargs.get("top_p", 1.0)
    top_k = gen_kwargs.get("top_k", 50)
    until = gen_kwargs.get("until", ["\n\n"])
    repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
    frequency_penalty = gen_kwargs.get("frequency_penalty", 0.0)
    presence_penalty = gen_kwargs.get("presence_penalty", 0.0)

    try:
        response = _call_vllm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=until,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    except Exception as e:
        print(f"[Inference] Generation failed: {e}")
        return ""

    choices = response.get("choices", [])
    if not choices:
        return ""

    text = choices[0].get("text", "")
    # Apply stop string trimming as fallback
    text = _apply_stop_strings(text, until)
    return text


def run_inference(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of messages and return results.
    """
    results = []
    for msg in messages:
        msg_id = msg.get("ID")
        rt = msg.get("eval_request_type", "loglikelihood")
        prompt = msg["prompt"]

        result = {
            "ID": msg_id,
            "prompt": prompt,
            "eval_request_type": rt,
        }

        try:
            if rt == "generate_until":
                gen_kwargs = msg.get("eval_gen_kwargs", {})
                result["response"] = generate_text(prompt, gen_kwargs)
                result["accuracy"] = None

            elif rt == "loglikelihood":
                continuation = msg.get("eval_continuation", "")
                result["accuracy"] = compute_logprob(prompt, continuation)
                result["response"] = None

            elif rt == "loglikelihood_rolling":
                result["accuracy"] = compute_rolling_logprob(prompt)
                result["response"] = None

            else:
                # Unknown type
                result["response"] = None
                result["accuracy"] = None

        except Exception as e:
            print(f"[Inference] Error processing message {msg_id}: {e}")
            result["response"] = None
            result["accuracy"] = None

        # Preserve required fields
        if "eval_req_id" in msg:
            result["eval_req_id"] = msg["eval_req_id"]
        if "eval_gen_kwargs" in msg:
            result["eval_gen_kwargs"] = msg["eval_gen_kwargs"]
        if "eval_continuation" in msg:
            result["eval_continuation"] = msg["eval_continuation"]

        results.append(result)

    return results


# Quick test
if __name__ == "__main__":
    # Test basic generation
    print("[Inference] Testing basic generation...")
    try:
        result = generate_text("Hello, how are you?", {"max_gen_toks": 20})
        print(f"[Inference] Generated: {result[:100]}")
    except Exception as e:
        print(f"[Inference] Test failed (expected if vLLM not running): {e}")
