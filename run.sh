#!/bin/bash
set -e

cd "$(dirname "$0")"

# 查找可用的 Python 3.x
for py in python3.12 python3.11 python3 python; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "未找到 Python"
    exit 1
fi

source /tmp/contestant_env/bin/activate

# 从配置加载环境变量默认值
MODEL_PATH=${MODEL_PATH:-"/mnt/model/Qwen3-32B"}
CONTESTANT_PORT=${CONTESTANT_PORT:-9000}
PLATFORM_URL=${PLATFORM_URL:-"http://10.0.0.1:8003"}
CONTESTANT_TOKEN=${CONTESTANT_TOKEN:-"your_secret_token"}
CONTESTANT_NAME=${CONTESTANT_NAME:-"team_alpha"}

export MODEL_PATH
export CONTESTANT_PORT
export PLATFORM_URL
export CONTESTANT_TOKEN
export CONTESTANT_NAME

# 通过 main.py 启动完整服务
exec $PYTHON_BIN main.py
