#!/usr/bin/env bash
cd "$(dirname "$0")"

PYTHON_BIN=python3.12
VENV_DIR="/tmp/contestant_env"

source $VENV_DIR/bin/activate

MODEL_PATH=${MODEL_PATH:-"/mnt/model/Qwen3-32B"}
CONTESTANT_PORT=${CONTESTANT_PORT:-9000}
PLATFORM_URL=${PLATFORM_URL:-"http://127.0.0.1:8003"}
TEAM_TOKEN=${TEAM_TOKEN:-6ac07c35427ca35e69222b18b51e81b2}
TEAM_NAME=${TEAM_NAME:-"冤有头债有组"}
CONFIG_PATH=${CONFIG_PATH:-"/mnt/config/contest.json"}

export MODEL_PATH
export CONTESTANT_PORT
export PLATFORM_URL
export TEAM_TOKEN
export TEAM_NAME
export CONFIG_PATH
export OMP_NUM_THREADS=1

exec python main.py
