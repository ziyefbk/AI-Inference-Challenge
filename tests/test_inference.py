"""
推理模块测试。
"""

import os
import sys

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
    print("推理模块测试通过")
