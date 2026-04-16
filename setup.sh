PYTHON_BIN=python3.12

VENV_DIR="/tmp/contestant_env"

$PYTHON_BIN -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
