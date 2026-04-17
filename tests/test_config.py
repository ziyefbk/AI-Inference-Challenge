"""
配置加载测试。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config


def test_config_defaults():
    """测试默认配置值。"""
    c = Config.__new__(Config)
    c._data = {}
    c._apply_env_overrides = lambda: None

    assert c.team_name == "contestant"
    assert c.platform_url == "http://127.0.0.1:8003"
    assert c.vllm_timeout == 120.0
    assert c.max_concurrent_messages == 20


def test_config_yaml_loading():
    """测试 YAML 配置文件加载。"""
    yaml_content = """
team:
  token: "test_token_123"
  name: "test_team"

platform:
  url: "http://test.example.com"

model:
  path: "/models/test"
  name: "TestModel"

inference:
  max_concurrent_messages: 50

vllm:
  timeout: 60.0
  max_retries: 5

logging:
  level: "DEBUG"

sampling_params:
  Normal:
    temperature: 0.5
    top_p: 0.95

sla_levels:
  express:
    ttft_avg: 0.3
  fast:
    ttft_avg: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        os.environ["CONFIG_PATH"] = f.name

        c = Config.__new__(Config)
        c._data = {}
        c._load()

        assert c.team_token == "test_token_123"
        assert c.team_name == "test_team"
        assert c.platform_url == "http://test.example.com"
        assert c.model_path == "/models/test"
        assert c.model_name == "TestModel"
        assert c.vllm_timeout == 60.0
        assert c.max_concurrent_messages == 50

        os.unlink(f.name)


def test_config_get_nested():
    """测试嵌套键访问。"""
    c = Config.__new__(Config)
    c._data = {
        "a": {"b": {"c": 123}},
        "x": None,
    }

    assert c.get("a", "b", "c") == 123
    assert c.get("a", "b", "missing", default=-1) == -1
    assert c.get("x") is None
    assert c.get("missing", default="default") == "default"


if __name__ == "__main__":
    test_config_defaults()
    test_config_yaml_loading()
    test_config_get_nested()
    print("配置加载测试通过")
