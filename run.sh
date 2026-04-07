#!/bin/bash
set -e

cd "$(dirname "$0")"

# Find available Python 3.x
for py in python3.12 python3.11 python3 python; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "No Python found"
    exit 1
fi

source /tmp/contestant_env/bin/activate

# Load env defaults from config if available
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

# Start the full service via main.py
exec $PYTHON_BIN main.py
