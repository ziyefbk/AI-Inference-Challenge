#!/bin/bash
set -e

# 查找可用的 Python 3.x
for py in python3.12 python3 python; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "未找到 Python,请安装 Python 3.12"
    exit 1
fi

echo "使用 Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 如需要则创建虚拟环境
if [ ! -d /tmp/contestant_env ]; then
    $PYTHON_BIN -m venv /tmp/contestant_env
fi

source /tmp/contestant_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
