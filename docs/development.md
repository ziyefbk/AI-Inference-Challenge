# 推理服务挑战赛 — 初赛开发文档

> 本文档面向参赛选手，说明代码架构、平台交互规范、计分规则，以及当前实现中发现的问题及修复优先级。

---

## 一、架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                        评测平台                               │
│              (http://10.0.0.1:8003)                         │
│         /register  /query  /ask  /submit  /reject          │
└──────────┬────────────────┬──────────────────┬────────────────┘
           │                │                  │
           │  httpx POST   │                  │
           ▼                ▼                  ▼
┌─────────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│   main.py           │ │   main.py       │ │   main.py          │
│  run_vllm_instances │ │  HTTP Server    │ │  main_loop()       │
│  (管理 vLLM 进程)   │ │ (健康检查)       │ │ (任务调度)          │
└────────┬────────────┘ └────────┬────────┘ └────────┬─────────┘
         │                       │                    │
         │  HTTP POST            │                    │
         ▼                       │                    ▼
┌─────────────────────┐          │         ┌────────────────────┐
│   vLLM Server       │          │         │  client/loop.py     │
│  localhost:8000+    │          │         │  (PrefetchWorker +  │
│                     │          │         │   InferenceWorkers)│
└─────────────────────┘          │         └────────┬───────────┘
                                  │                  │
                                  │         ┌────────▼───────────┐
                                  │         │  task_holder.py    │
                                  │         │  (优先队列调度)     │
                                  │         └────────┬───────────┘
                                  │                  │
                                  │         ┌────────▼───────────┐
                                  │         │  inference/         │
                                  │         │  vllm.py           │
                                  │         │  (vLLM 客户端封装)  │
                                  │         └────────┬───────────┘
                                  │                  │
                                  │         ┌────────▼───────────┐
                                  │         │  task_types/       │
                                  │         │  generate_until.py │
                                  │         │  loglikelihood.py  │
                                  │         └────────────────────┘
                                  │                  │
                                  └──────────────────┘
```

**组件职责：**

| 组件 | 文件 | 职责 |
|------|------|------|
| 入口 | `main.py` | 启动 vLLM 实例、HTTP 健康检查服务、主循环 |
| 任务调度 | `client/loop.py` | prefetch-worker + inference-worker 并发模型 |
| 任务持有 | `client/task_holder.py` | 基于优先队列的任务缓冲与过期管理 |
| 平台交互 | `client/platform.py` | /register /query /ask /submit /reject 封装 |
| 推理引擎 | `inference/__init__.py` | 消息并行处理、异常聚合 |
| vLLM 客户端 | `inference/vllm.py` | completions / chat_completions API，负载均衡 |
| 策略管理 | `inference/strategy.py` | SLA 策略加载、自适应降级 |
| 任务处理器 | `inference/task_types/*.py` | generate_until / loglikelihood / loglikelihood_rolling |

---

## 二、任务处理流程

```
query_task()
    │
    ▼
estimate_task_feasible()          ← 若预测超时 → reject_task() 并跳过
    │
    ▼
accept_task()                     ← 若竞争失败 → reject_task() 并跳过
    │
    ▼
task_holder.add_task()            ← 若 holder 满 → reject_task() 并跳过
    │
    ▼
InferenceWorker.pop_task() ────────── worker 从队列取任务（优先级顺序）
    │
    ▼
process_task()
    │
    ▼
run_inference_async()
    │
    ▼
asyncio.gather(_process_single_message for each msg)
    │
    ├── generate_until  → completions()  → vLLM /v1/completions
    ├── loglikelihood   → compute_logprob() → vLLM /v1/completions (echo=True)
    └── loglikelihood_rolling → compute_rolling_logprob() → vLLM /v1/completions
    │
    ▼
submit_results()
    │
    ▼
platform /submit  → 评测平台评分
```

**SLA 时限（`deadline_ms`）的含义：** 从 `accept_task` 返回时刻起算，平台允许的最大端到端延迟。`ttft_avg` 是该 SLA 级别的目标平均首 token 时间（见配置文件 `sla_levels`）。

---

## 三、平台交互协议

### 3.1 接口对照表

| 步骤 | 平台端点 | 代码位置 | 关键字段 |
|------|---------|---------|---------|
| 注册 | `POST /register` | `platform.register()` | `name`, `token` |
| 查询 | `POST /query` | `platform.query_task()` | `token` |
| 接受 | `POST /ask` | `platform.accept_task()` | `token`, `task_id`, `sla` |
| 拒绝 | `POST /reject` | `platform.reject_task()` | `token`, `task_id`, `reason` |
| 提交 | `POST /submit` | `platform.submit_results()` | `user`, `msg` |

### 3.2 submit 字段映射

`POST /submit` 请求体格式：

```json
{
  "user": {
    "name": "team_alpha",
    "token": "your_secret_token"
  },
  "msg": {
    "overview": { ... },              // 来自 accept_task 返回的 task.overview，原样透传
    "messages": [
      {
        "ID": 0,
        "prompt": "Andy plants 90 geraniums...",
        "response": " To determine the total...",  // generate_until 填充，loglikelihood 为 null
        "accuracy": null,                          // generate_until 为 null
        "eval_req_id": "w0_a1b2c3d4",             // 平台下发了则有，否则可省略
        "eval_request_type": "generate_until",
        "continuation": null,                     // 原样透传
        "extracted_answer": null                  // extract_final_answer() 结果，可省略
      },
      {
        "ID": 1,
        "prompt": "My kitchen floor has a total area...",
        "response": null,                         // loglikelihood 为 null
        "accuracy": -117.63,                      // loglikelihood/loglikelihood_rolling 填充
        "eval_req_id": "w0_b5c6d7e8",
        "eval_request_type": "loglikelihood",
        "continuation": " No"                     // 原样透传，平台下发的选项文本
      },
      {
        "ID": 2,
        "prompt": "Two track teams are competing...",
        "response": null,
        "accuracy": -367.39,                      // 整段文本所有 token 的 logprob sum
        "eval_request_type": "loglikelihood_rolling",
        "continuation": null
      }
    ],
    "sla_level": "Silver"                      // 必须，放在 msg 顶层
  }
}
```

**填充规则：**

| eval_request_type | `response` | `accuracy` | `continuation` | `extracted_answer` |
|-------------------|:----------:|:----------:|:--------------:|:-------------------:|
| `generate_until` | 模型生成的文本 | `null` | `null` | `\boxed{}` / `####` 后提取的答案 |
| `loglikelihood` | `null` | log P(continuation \| prompt) | 原样透传 | — |
| `loglikelihood_rolling` | `null` | 整段 token 的 logprob sum | `null` | — |

---

## 四、计分机制

```
R_i = w_task × w_sla × w_sp × C_i

其中:
  w_task   - 任务基础权重（generate_until 最高，loglikelihood 按候选数加权）
  w_sla    - SLA 难度乘数（Bronze=1.0，Bronze→Supreme 递增至 2.5）
  w_sp     - 采样参数难度乘数（Deterministic=1.0，ExtremePenalty=1.3，仅 generate_until）
  C_i      - 正确性得分（0~1.0）
```

**SLA 时限行为：**

| 状态 | 条件 | 得分 |
|------|------|------|
| SLA 内完成 | 端到端延迟满足 SLA 要求 | `R_i` |
| SLA 超时完成 | 超过 SLA 但在 600 秒内提交 | `0`（不扣分） |
| 未完成 | 600 秒内未提交 | `-2 × w_task × w_sla × w_sp` |

**loglikelihood 正确性：** 多候选消息的 accuracy 由平台做 argmax 判对——选手提交的多个候选 logprob 中最大值对应的选项与参考答案一致则得 1.0，否则 0.0。

---

## 五、环境变量与配置

### 5.1 平台环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PLATFORM_URL` | 评测平台地址 | `http://127.0.0.1:8003` |
| `TEAM_TOKEN` | 队伍认证 token | `""` |
| `TEAM_NAME` | 队伍名称 | `"contestant"` |
| `CONFIG_PATH` | 比赛配置文件路径（JSON） | `""` |
| `MODEL_PATH` | 模型权重目录 | `/root/autodl-tmp/models/Qwen2.5-0.5B` |

### 5.2 vLLM 推理配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_NUM_INSTANCES` | vLLM 实例数量 | `1` |
| `VLLM_URLS` | vLLM 服务地址列表（逗号分隔） | `http://localhost:8000` |
| `VLLM_TIMEOUT` | 单次请求超时（秒） | `120` |
| `VLLM_MAX_RETRIES` | 最大重试次数 | `3` |
| `VLLM_BACKOFF_FACTOR` | 退避系数 | `1.5` |
| `VLLM_MAX_BACKOFF` | 最大退避时间（秒） | `30` |
| `VLLM_MAX_CONNECTIONS` | 最大连接数 | `200` |
| `VLLM_MAX_KEEPALIVE` | 最大 keepalive 连接数 | `100` |
| `MODEL_MAX_LEN` | 模型 context window 上限（tokens） | `1000` |

### 5.3 调度配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NUM_WORKERS` | 推理 worker 数量 | `3` |
| `MAX_CONCURRENT_MESSAGES` | 最大并发消息数 | `20` |
| `PREFETCH_SIZE` | 预取任务数量 | `8` |
| `MAX_HELD_TASKS` | 任务持有器最大容量 | `64` |
| `TASK_ACCEPT_TIMEOUT` | 任务接受超时（秒） | `300` |
| `CLIENT_MAX_CONNECTIONS` | 平台客户端最大连接数 | `256` |
| `CLIENT_MAX_KEEPALIVE` | 平台客户端 keepalive 连接数 | `128` |

---

## 六、问题清单与修复优先级

### P0 — 必须修复（影响正确性或稳定性）

#### 1. `MODEL_MAX_LEN` 默认值过小

**文件：** `src/inference/vllm.py:34`

```python
MODEL_MAX_LEN = int(os.environ.get("MODEL_MAX_LEN", "1000"))
```

**问题：** 默认值 `1000` 远小于 Qwen2.5-0.5B 的实际 context window（32768），也小于配置文件 `strategy.py` 中 `max_model_len: 32768`。`generate_until` 的 `max_tokens` 被错误地限制在 1000，导致长文本生成任务无法完整输出。

**影响：** 所有 `generate_until` 任务的 `max_tokens` 上限被压至 1000，即使平台配置 `max_gen_toks=50000`，实际生成也会在 1000 处截断。

**修复：** 将默认值改为 `32000`（比赛环境 Qwen3-32B 应为 `32768` 或更高，可通过环境变量覆盖）：

```python
MODEL_MAX_LEN = int(os.environ.get("MODEL_MAX_LEN", "32000"))
```

---

#### 2. `SLA_LEVEL_TO_STRATEGY` 映射不完整

**文件：** `src/inference/strategy.py:201-207`

```python
SLA_LEVEL_TO_STRATEGY: Dict[str, str] = {
    "Bronze": "standard",
    "Silver": "standard",
    "Gold": "standard",
    "Diamond": "standard",
    "Platinum": "standard",
}
```

**问题：** 比赛规则定义了 8 个 SLA 级别（Bronze / Silver / Gold / Platinum / Diamond / Stellar / Glorious / Supreme），但映射表只覆盖了 5 个。`Stellar`、`Glorious`、`Supreme` 三个高优先级 SLA 完全缺失。当平台下发这些 SLA 时，`prepare_sla()` 返回 `"standard"` 策略，导致：

- 高难度任务的采样参数（`max_gen_toks`、`temperature` 等）被降级
- `w_sla` 难度乘数无法正确体现（Supreme=2.5 vs Bronze=1.0）

**影响：** 高 SLA 级别任务的收益显著降低，可能只有理论得分的约 1/2。

**修复：** 补全映射并按难度分级：

```python
SLA_LEVEL_TO_STRATEGY: Dict[str, str] = {
    "Bronze":   "standard",
    "Silver":   "standard",
    "Gold":     "standard",
    "Platinum": "standard",
    "Diamond":  "high_quality",
    "Stellar":  "high_quality",
    "Glorious": "express",
    "Supreme":  "express",
}
```

---

### P1 — 建议修复（影响性能或可维护性）

#### 3. `generate_until` 用词数估算 token 数

**文件：** `src/inference/task_types/generate_until.py:105`

```python
"prompt_tokens": len(prompt.split()),
```

**问题：** `len(prompt.split())` 是词数（word count），而非 token 数。中文和代码场景下差异极大（1 token ≈ 0.75 中文汉字 ≈ 0.25 英文单词）。`max_model_len - prompt_tokens` 的 clamp 逻辑因此失效，可能导致 `max_tokens` 超出模型 context window。

**影响：** 极少数超长 prompt 场景下可能触发 vLLM 400 错误。

**修复：** 使用真实 tokenizer 估算，或在 clamp 时使用保守系数（如 `prompt_tokens * 1.3`）：

```python
"prompt_tokens": len(prompt.split()) * 1.3,  # 词数 → token 保守估计
```

---

#### 4. warmup 中未使用的变量

**文件：** `src/inference/__init__.py:93`

```python
async def _do_warmup():
    import httpx
    from src.inference.vllm import chat_completions, RETRY_CONFIG, MODEL_PATH, get_vllm_urls
    urls = get_vllm_urls()          # ← 声明后未使用
    client = httpx.AsyncClient(...)
```

**问题：** `urls` 变量被获取后未使用，`chat_completions()` 内部会重新调用 `get_vllm_urls()`，这段逻辑是死代码。

**修复：** 删除该行。

---

#### 5. vLLM 未启动时 ConnectError 无明确提示

**文件：** `src/inference/vllm.py`

**问题：** 当 vLLM 服务未启动（或崩溃）时，`chat_completions` 和 `completions` 对所有异常（OSError、TimeoutException、HTTPStatusError）都执行等幂重试，耗尽重试次数后才抛出原始异常。用户看到的错误是隐晦的 `httpx.ConnectError: All connection attempts failed`，没有任何提示告知 vLLM 未启动。

**影响：** 调试困难，用户需要手动查看 traceback 才能判断是 vLLM 不可达还是其他错误。

**修复：** 在 OSError（ConnectError）全部重试耗尽后，追加一次健康检查。若 vLLM 仍不可达，抛出明确的 `VLLMUnavailableError`，提示启动命令：

```python
VLLMUnavailableError: vLLM 实例 http://localhost:8000 不可达（连接被拒绝）。
请确认 vLLM 进程已启动。启动命令: python -m vllm.entrypoints.openai.api_server ...
```

---

#### 6. loglikelihood 计算错误（已修复）

**文件：** `src/inference/task_types/loglikelihood.py`

**问题：** `compute_logprob` 和 `compute_rolling_logprob` 的计算逻辑与数学定义不符。

**旧版 `compute_logprob`：**
- 用 `max_tokens=len(continuation.split())`（词数，非 token 数）生成 continuation
- 额外生成的 padding tokens 被累加进 logprob
- **正确做法：** 将 `prompt + continuation` 整体作为 prompt，`max_tokens=0`（不生成），`echo=True`，精确提取 continuation token 数

**旧版 `compute_rolling_logprob`：**
- 用 `max_tokens=1` 生成额外 1 个 token，`lp_list[1:]` 累加了非文本的 token
- **正确做法：** `max_tokens=0`，`lp_list[1:]` 恰好等于文本的 N 个 token

**状态：** ✅ 已修复（2026-04-18）

---

### P2 — 可选修复（边界情况处理）

#### 5. `run_inference` 的事件循环检测不准确

**文件：** `src/inference/__init__.py:329-334`

```python
try:
    asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.submit(lambda: asyncio.run(_run())).result()
except RuntimeError:
    return asyncio.run(_run())
```

**问题：** `get_running_loop()` 在没有运行中的 loop 时抛出 `RuntimeError`，此时走 `asyncio.run()` 分支——逻辑正确。但若在有 loop 的线程中调用（如嵌套 async 调用），会走 ThreadPoolExecutor 分支，创建不必要的线程开销。当前调用路径（`loop.py` → `run_inference_async`）无此问题。

**影响：** 低，仅在同步入口被嵌套调用时有一定性能损耗。

**修复：** 若确认所有调用路径均走 `run_inference_async`，可删除此分支逻辑，保留 `asyncio.run(_run())`。

---

#### 6. 任务结果验证过于宽松

**文件：** `src/inference/__init__.py:171-180`

```python
if rt == "generate_until":
    if len(response) < 3:
        return False
    words = response.split()
    if len(words) >= 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False
```

**问题：** 验证逻辑会将低熵重复文本（如 `"aaaaa aaaaa aaaaa"）标记为无效，但不影响提交（只加 `_validation_failed` 元数据）。对于短回答（如答案只有一个词），`len(response) < 3` 也会拒绝。这些结果是错的，但不会阻止提交。

**影响：** 监控指标会记录验证失败，但不影响评分。酌情调整阈值。

---

#### 7. 重试逻辑对 400 不重试

**文件：** `src/inference/vllm.py:282-286`

```python
if sc == 400:
    body = _err_body.get("vllm") or e.response.content.decode(errors="replace")
    body_hint = f" | vLLM: {body[:300] if body else 'empty'}"
    logger.warning("vllm_400_detail", ...)
```

**问题：** vLLM 返回 400 时，body 信息只在日志中出现，不出现在 `submit` 的请求中。平台侧看不到推理失败原因。

**影响：** 调试时难以定位 400 根因。建议将 400 body 附加到结果字段或单独记录。

---

## 七、文件清单

```
AI-Inference-Challenge/
├── main.py                          # 入口：vLLM 启动 + HTTP 服务 + 主循环
├── run.sh                           # 启动脚本
├── setup.sh                         # 环境安装脚本
├── requirements.txt                 # Python 依赖
├── config.example.yaml              # 配置示例
├── docs/
│   ├── 问题分析报告.md               # 历史问题记录
│   └── development.md              # 本文档
├── src/
│   ├── config.py                    # 配置加载（Config 单例 + 环境变量覆盖）
│   ├── inference/
│   │   ├── __init__.py             # 推理引擎入口、预热、消息并行处理
│   │   ├── vllm.py                 # vLLM API 封装（completions/chat_completions）
│   │   ├── strategy.py              # SLA 策略管理
│   │   └── task_types/
│   │       ├── __init__.py
│   │       ├── generate_until.py    # 文本生成任务处理
│   │       └── loglikelihood.py     # logprob 计算（包含 rolling）
│   ├── client/
│   │   ├── platform.py             # 平台交互（注册/查询/接受/提交/熔断器）
│   │   ├── loop.py                 # 主循环（prefetch + workers）
│   │   ├── task_holder.py          # 优先队列任务持有器
│   │   └── monitor.py             # 指标监控
│   └── utils/
│       ├── logger.py               # 结构化日志
│       ├── metrics.py             # Prometheus 指标
│       ├── circuit.py             # 熔断器
│       └── graceful.py            # 优雅退出
└── tests/
    └── ...
```

---

## 八、快速开始

```bash
# 1. 安装依赖
bash setup.sh

# 2. 配置环境变量
export PLATFORM_URL=http://10.0.0.1:8003
export TEAM_TOKEN=your_token_here
export TEAM_NAME=your_team_name
export MODEL_PATH=/mnt/model/Qwen3-32B
export CONFIG_PATH=/mnt/config/contest.json

# 3. 启动
bash run.sh
```

> **注意：** 比赛环境中 `MODEL_PATH` 和 `CONFIG_PATH` 由平台通过环境变量传入，无需手动配置。
