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

# 使用 conda 创建/更新环境
ENV_NAME="quant"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "更新 conda 环境: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f environment.yml 2>/dev/null || true
else
    echo "创建 conda 环境: $ENV_NAME"
    conda env create -n "$ENV_NAME" -f environment.yml
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
pip install --upgrade pip
pip install -r requirements.txt
