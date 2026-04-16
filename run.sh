cd "$(dirname "$0")"

PYTHON_BIN=python3.12
VENV_DIR="/tmp/contestant_env"

source $VENV_DIR/bin/activate

MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/models/Qwen2.5-0.5B"}
CONTESTANT_PORT=${CONTESTANT_PORT:-9000}
PLATFORM_URL=${PLATFORM_URL:-"http://127.0.0.1:8003"}
TEAM_TOKEN=${TEAM_TOKEN:-6ac07c35427ca35e69222b18b51e81b2}
TEAM_NAME=${TEAM_NAME:-"contestant"}

export MODEL_PATH
export CONTESTANT_PORT
export PLATFORM_URL
export TEAM_TOKEN
export TEAM_NAME
export OMP_NUM_THREADS=1

exec python main.py
