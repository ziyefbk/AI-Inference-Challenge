# AI Inference Challenge

## 项目结构

```
AI-Inference-Challenge/
├── main.py              # 服务主入口：启动 vLLM + HTTP 服务器 + 客户端循环
├── src/
│   ├── client.py        # 平台交互：register → query → accept → inference → submit
│   ├── inference.py     # 推理引擎：三种任务类型的 vLLM 调用封装
│   ├── test_gsm8k.py    # GSM8K 本地测试脚本
│   └── utils/
│       ├── circuit.py   # 熔断器
│       ├── graceful.py  # 优雅关闭
│       ├── logger.py    # 日志
│       └── metrics.py   # 指标
├── requirements.txt
└── README.md
```

## 三种任务类型

| 字段 | 任务类型 | 调用 API | 计算内容 |
|------|---------|---------|---------|
| `response` | `generate_until` | Chat Completions | 文本生成，响应填入 `response` |
| `accuracy` | `loglikelihood` | Completions | log P(continuation \| prompt)，概率填入 `accuracy` |
| `accuracy` | `loglikelihood_rolling` | Completions | 整文档总 logprob，概率填入 `accuracy` |

- `loglikelihood` 的 `eval_continuation` 由平台下发，参赛者计算该 continuation 的概率
- `loglikelihood_rolling` 的 `eval_continuation` 为 null，直接对 prompt 自身算 perplexity
- 平台收集同题所有候选后取 argmax 与参考答案比对

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEAM_TOKEN` | 参赛 token | — |
| `TEAM_NAME` | 队伍名称 | contestant |
| `MODEL_PATH` | 模型路径 | — |
| `MODEL_NAME` | 模型名 | Qwen3-32B |
| `PLATFORM_URL` | 平台地址 | http://127.0.0.1:8003 |
| `CONTESTANT_PORT` | HTTP 服务端口 | 9000 |
| `VLLM_PORT` | vLLM 端口 | 8000 |
| `VLLM_NUM_INSTANCES` | vLLM 实例数（自动检测 GPU 数） | 1 |
| `MAX_CONCURRENT_MESSAGES` | 最大并发消息数 | 20 |
| `VLLM_MAX_RETRIES` | API 重试次数 | 3 |
| `VLLM_TIMEOUT` | API 超时（秒） | 120 |
| `WARMUP_ENABLED` | 是否预热模型 | true |
| `LOG_LEVEL` | 日志级别 | INFO |
| `SPECULATIVE_MODEL` | 投机解码小模型路径 | — |

## 启动方式

```bash
# 标准启动（自动检测 GPU 数，启动多实例 vLLM）
python main.py --token <TOKEN> --name <NAME> --platform-url <URL> --model-path <PATH>

# 跳过 vLLM 启动（假设已运行）
python main.py --no-vllm --token <TOKEN> --platform-url <URL>

# 本地 GSM8K 测试（无需平台）
MODEL_PATH=/path/to/model python src/test_gsm8k.py --max-samples 100
```

## 推理参数配置

通过 `SLA_STRATEGIES` 配置四档推理策略：

| 级别 | 适用场景 | max_gen_toks | temperature | top_p | 典型用途 |
|------|---------|-------------|-------------|-------|---------|
| express | deadline ≤ 1s | 128 | 0.0 | 1.0 | 极速选择题 |
| fast | deadline ≤ 5s | 256 | 0.1 | 0.95 | 短答案 |
| standard | deadline ≤ 30s | 512 | 0.3 | 0.95 | 常规推理 |
| high_quality | deadline > 30s | 1024 | 0.3 | 0.95 | 复杂推理 |

`generate_until` 任务的停止条件为 `####`（GSM8K 标准答案分隔符），模型在生成答案后停止。

## 监控文件

推理过程的详细记录保存在 `/tmp/task_monitor/` 目录：

```
/tmp/task_monitor/
├── queried_tasks.jsonl        # 平台查询到的任务概览
├── submitted_results.jsonl    # 提交结果（成功/失败）
└── inference_results.jsonl   # 每条消息的 prompt + response/accuracy
```

## 修改记录

### v2 — 2026-04-17

**P0 修复：generate_until 截断问题**
- 停止条件从 `\n\n` 改为 `####`
- 之前 34.6% 的响应在第一段换行时被截断，停在 `"Let's break down..."` 开头
- 现在响应会在答案分隔符处停止，确保包含完整答案

**P0 修复：tiktoken vs vLLM tokenizer 不对齐**
- `compute_logprob` 和 `compute_rolling_logprob` 不再依赖 tiktoken 估算 prompt 末尾位置
- 改用 vLLM completions API 返回的 `tokens` 列表精确定位
- 启发式规则：找到第一个有效负数 logprob 作为分界点

**P1 优化：采样参数调整**
- `express`: max_gen_toks 32 → 128
- `fast`: max_gen_toks 64 → 256
- `standard`: max_gen_toks 128 → 512, temperature 0.7 → 0.3
- `high_quality`: max_gen_toks 256 → 1024, temperature 0.8 → 0.3
- 所有档位 top_p 从 0.9 → 0.95，提高生成稳定性
- repetition_penalty 降低（1.1 → 1.05），避免过度惩罚
- frequency/presence_penalty 降低，避免影响答案生成

**P2 增强：答案提取后处理**
- 新增 `extract_final_answer()` 函数，从 `generate_until` 响应中提取最终答案
- 优先级：#### 分隔符 > 等号表达式 > 末尾数字
- 结果写入 `result["extracted_answer"]` 字段，供平台参考答案匹配
