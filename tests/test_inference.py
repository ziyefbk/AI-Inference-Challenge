"""
推理模块测试。
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_extract_final_answer():
    """测试答案提取函数。"""
    from src.inference.task_types import extract_final_answer

    # GSM8K 格式
    assert extract_final_answer("Let's solve it.\n#### 42") == "42"
    assert extract_final_answer("Answer is\n####\n300") == "300"

    # \boxed 格式
    assert extract_final_answer("Therefore \\boxed{71}") == "71"
    assert extract_final_answer("The answer is \\boxed{14}.") == "14"

    # 等号格式
    assert extract_final_answer("= 71") == "71"
    assert extract_final_answer(r"cost is $300") == "300"

    # 纯数字
    assert extract_final_answer("The result is 25") == "25"

    # 空输入
    assert extract_final_answer("") is None
    assert extract_final_answer(None) is None  # type: ignore

    print("extract_final_answer 测试通过")


def test_validate_result():
    """测试结果验证函数。"""
    from src.inference import validate_result

    # generate_until 有效
    assert validate_result({"eval_request_type": "generate_until", "response": "The answer is 42."})

    # generate_until 过短
    assert not validate_result({"eval_request_type": "generate_until", "response": "Hi"})

    # generate_until 为空
    assert not validate_result({"eval_request_type": "generate_until", "response": ""})

    # generate_until 重复内容
    assert not validate_result({"eval_request_type": "generate_until", "response": "foo foo foo foo foo"})

    # loglikelihood 有效
    assert validate_result({"eval_request_type": "loglikelihood", "accuracy": -5.0})

    # loglikelihood None
    assert not validate_result({"eval_request_type": "loglikelihood", "accuracy": None})

    print("validate_result 测试通过")


def test_std_calculation():
    """测试标准差计算。"""
    from src.inference import _std

    assert _std([]) == 0.0
    assert _std([5.0]) == 0.0
    assert abs(_std([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.414) < 0.01

    print("_std 测试通过")


if __name__ == "__main__":
    test_extract_final_answer()
    test_validate_result()
    test_std_calculation()
    test_classify_task_length()
    test_priority_task_holder_buckets()
    print("推理模块测试通过")


def test_classify_task_length():
    """测试任务长度分桶逻辑。"""
    from src.client.task_holder import classify_task_length, PROMPT_LENGTH_SHORT_THRESHOLD

    # 空任务 → short
    assert classify_task_length({}) == 0

    # 短 prompt（远低于阈值）→ short
    short_task = {
        "messages": [
            {"prompt": "What is 2+2?", "eval_continuation": " 4"}
        ]
    }
    assert classify_task_length(short_task) == 0

    # prompt + continuation 总长度刚好等于阈值 → short
    threshold = PROMPT_LENGTH_SHORT_THRESHOLD
    task_at_threshold = {
        "messages": [{"prompt": "x" * (threshold - 4), "eval_continuation": " A"}]
    }
    assert classify_task_length(task_at_threshold) == 0

    # prompt + continuation 超过阈值 → long
    task_over_threshold = {
        "messages": [{"prompt": "x" * (threshold + 1), "eval_continuation": " A"}]
    }
    assert classify_task_length(task_over_threshold) == 1

    # 多条消息，取最大值判断；max > 阈值 → long
    mixed_task = {
        "messages": [
            {"prompt": "x" * 10, "eval_continuation": ""},
            {"prompt": "x" * (threshold + 100), "eval_continuation": ""},
        ]
    }
    assert classify_task_length(mixed_task) == 1

    # 多条消息，全在阈值内 → short
    all_short_task = {
        "messages": [
            {"prompt": "x" * 10, "eval_continuation": ""},
            {"prompt": "x" * 20, "eval_continuation": ""},
        ]
    }
    assert classify_task_length(all_short_task) == 0

    # continuation 为 None（不抛异常）
    none_cont_task = {"messages": [{"prompt": "Hello", "eval_continuation": None}]}
    assert classify_task_length(none_cont_task) == 0

    print("classify_task_length 测试通过")


def test_priority_task_holder_buckets():
    """测试 PriorityTaskHolder 分桶和短任务优先弹出逻辑。"""
    from src.client.task_holder import PriorityTaskHolder, classify_task_length

    holder = PriorityTaskHolder(max_held=20, timeout_s=60)

    short_task = {"overview": {"task_id": 1}, "messages": [{"prompt": "hi"}]}
    long_task = {
        "overview": {"task_id": 2},
        "messages": [{"prompt": "x" * 1000}],
    }

    # classify_task_length 结果符合预期
    assert classify_task_length(short_task) == 0
    assert classify_task_length(long_task) == 1

    async def run():
        # 添加两个任务
        assert await holder.add_task(short_task) is True
        assert await holder.add_task(long_task) is True

        # 连续弹出，应始终短任务优先
        item1 = await holder.pop_task()
        assert item1 is not None
        task1, _, bucket1 = item1
        assert bucket1 == 0  # short 先出

        item2 = await holder.pop_task()
        assert item2 is not None
        task2, _, bucket2 = item2
        assert bucket2 == 1  # long 后出

        # 两个都弹完后为空
        assert await holder.pop_task() is None

        # stats 包含分桶字段
        stats = await holder.get_stats()
        assert "short_held" in stats
        assert "long_held" in stats
        assert stats["held"] == 0

        print("PriorityTaskHolder 分桶测试通过")

    asyncio.run(run())
