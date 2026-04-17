"""
GSM8K 本地推理测试脚本。

用法：
1. 确保 vLLM 服务已启动（端口 8000）
2. 设置环境变量 MODEL_PATH, MODEL_NAME 等
3. 运行: python src/test_gsm8k.py [--max-samples 100]

该脚本会：
- 加载本地缓存的 GSM8K 数据集
- 对每个问题调用推理服务
- 计算准确率
- 保存结果到 /tmp/gsm8k_results.jsonl
"""

import os
import sys
import json
import time
import argparse
import pyarrow as pa

# 设置缓存目录
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/huggingface_cache/datasets")

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import run_inference_async, generate_text, get_sla_strategy


# ── 数据集加载 ──────────────────────────────────────────────────────────────

def load_gsm8k_dataset():
    """加载本地缓存的 GSM8K 数据集。"""
    base = "/root/autodl-tmp/huggingface_cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"
    test_path = os.path.join(base, "gsm8k-test.arrow")
    train_path = os.path.join(base, "gsm8k-train.arrow")

    records = []

    # 读取 test
    with pa.memory_map(test_path, 'r') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
        for i in range(table.num_rows):
            records.append({
                "id": i,
                "split": "test",
                "question": table.column("question")[i].as_py(),
                "answer": table.column("answer")[i].as_py(),
            })

    # 读取 train
    with pa.memory_map(train_path, 'r') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
        for i in range(table.num_rows):
            records.append({
                "id": len(records),
                "split": "train",
                "question": table.column("question")[i].as_py(),
                "answer": table.column("answer")[i].as_py(),
            })

    print(f"[GSM8K] 加载完成: {len(records)} 条 (test + train)")
    return records


def extract_final_answer(answer_text: str) -> str:
    """从 answer 字段提取最终答案（#### 分隔符后的部分）。"""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def extract_model_answer(response: str) -> str:
    """从模型回复中提取答案（尝试找 #### 或最后一行数字）。"""
    if not response:
        return ""
    # 优先找 ####
    if "####" in response:
        return response.split("####")[-1].strip()
    # 找最后一行的数字
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].strip()
    return response.strip()


def normalize_answer(ans: str) -> str:
    """标准化答案：去除 $，去除逗号，转小写。"""
    return ans.replace("$", "").replace(",", "").replace(",", "").strip().lower()


# ── 推理 ────────────────────────────────────────────────────────────────────

def build_messages(question: str, answer: str):
    """将 GSM8K 样本构造成推理消息格式。"""
    return [
        {
            "ID": 0,
            "eval_request_type": "generate_until",
            "prompt": question,
            "eval_gen_kwargs": {"max_gen_toks": 512, "temperature": 0.3},
            "eval_continuation": "",
        }
    ]


# ── 主测试循环 ──────────────────────────────────────────────────────────────

async def run_test(dataset, max_samples: int = None, output_file: str = "/tmp/gsm8k_results.jsonl"):
    """运行本地推理测试。"""
    import asyncio

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"[GSM8K 测试] 开始测试 {len(dataset)} 条样本...")

    results = []
    correct = 0
    total = 0

    for i, item in enumerate(dataset):
        q = item["question"]
        expected_answer = extract_final_answer(item["answer"])

        print(f"[{i+1}/{len(dataset)}] 问题: {q[:60]}...")

        # 构造消息并推理
        messages = build_messages(q, item["answer"])

        try:
            start = time.time()
            inference_results = await run_inference_async(messages, sla_level="standard")
            elapsed = time.time() - start

            # 提取模型回答
            if inference_results:
                response = inference_results[0].get("response", "")
                model_answer = extract_model_answer(response)
            else:
                response = ""
                model_answer = ""

            # 判断正确性
            norm_expected = normalize_answer(expected_answer)
            norm_model = normalize_answer(model_answer)

            is_correct = norm_expected == norm_model
            if is_correct:
                correct += 1
            total += 1

            result = {
                "id": item["id"],
                "split": item["split"],
                "question": q,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "response": response[:500],
                "correct": is_correct,
                "elapsed_s": elapsed,
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"    {status} 期望: {expected_answer}, 模型: {model_answer} ({elapsed:.2f}s)")

        except Exception as e:
            print(f"    错误: {e}")
            results.append({
                "id": item["id"],
                "split": item["split"],
                "question": q,
                "expected_answer": expected_answer,
                "model_answer": "",
                "response": "",
                "correct": False,
                "error": str(e),
            })
            total += 1

        # 每 50 条保存一次
        if (i + 1) % 50 == 0:
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            acc = correct / total * 100 if total > 0 else 0
            print(f"\n[Checkpoint] {correct}/{total} 正确, 准确率: {acc:.1f}%\n")

    # 保存结果
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n=== GSM8K 测试完成 ===")
    print(f"样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {acc:.1f}%")
    print(f"结果已保存: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GSM8K 本地推理测试")
    parser.add_argument("--max-samples", type=int, default=None, help="最多测试多少条样本")
    parser.add_argument("--test-only", action="store_true", help="只用 test 集")
    parser.add_argument("--train-only", action="store_true", help="只用 train 集")
    parser.add_argument("--output", default="/tmp/gsm8k_results.jsonl", help="结果输出文件")
    args = parser.parse_args()

    # 加载数据集
    dataset = load_gsm8k_dataset()

    if args.test_only:
        dataset = [d for d in dataset if d["split"] == "test"]
    elif args.train_only:
        dataset = [d for d in dataset if d["split"] == "train"]

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    print(f"[配置] 测试样本数: {len(dataset)}")

    # 运行测试
    import asyncio
    asyncio.run(run_test(dataset, max_samples=args.max_samples, output_file=args.output))


if __name__ == "__main__":
    main()
