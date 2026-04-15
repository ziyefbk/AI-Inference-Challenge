#!/bin/bash
set -e

# 查找可用的 Python
for py in python python3 python3.12; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done

echo "使用 Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 如需要则创建虚拟环境
if [ ! -d ./tmp/contestant_env ]; then
    $PYTHON_BIN -m venv ./tmp/contestant_env
fi

source ./tmp/contestant_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
