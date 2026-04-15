#!/bin/bash
set -e

cd "$(dirname "$0")"

# 查找可用的 Python
for py in python python3 python3.12; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done


source ./tmp/contestant_env/bin/activate

# 从配置加载环境变量默认值
MODEL_PATH=${MODEL_PATH:-"../models/Qwen2.5-0.5B"}
CONTESTANT_PORT=${CONTESTANT_PORT:-9000}
PLATFORM_URL=${PLATFORM_URL:-"http://10.0.0.1:8003"}
TEAM_TOKEN=${TEAM_TOKEN:-6ac07c35427ca35e69222b18b51e81b2}
TEAM_NAME=${TEAM_NAME:-"contestant"}

export MODEL_PATH
export CONTESTANT_PORT
export PLATFORM_URL
export TEAM_TOKEN
export TEAM_NAME

# 通过 main.py 启动完整服务
exec $PYTHON_BIN main.py
