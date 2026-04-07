#!/bin/bash
set -e

# Find available Python 3.x
for py in python3.12 python3 python; do
    if command -v $py &> /dev/null; then
        PYTHON_BIN=$py
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "No Python found, please install Python 3.12"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Create venv if needed
if [ ! -d /tmp/contestant_env ]; then
    $PYTHON_BIN -m venv /tmp/contestant_env
fi

source /tmp/contestant_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
