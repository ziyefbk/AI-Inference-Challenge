#!/bin/bash

PYTHON_BIN=python3.12
VENV_DIR="/tmp/contestant_env"

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
