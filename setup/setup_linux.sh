#!/usr/bin/env bash
# ===============================================================================
# 3D Compressible CFD Solver - Automated Linux / Ubuntu / WSL2 Setup
# ===============================================================================

set -e

echo "==============================================================================="
echo " 3D Compressible CFD Solver - Automated Linux / Ubuntu / WSL2 Device Setup"
echo "==============================================================================="
echo ""

# Ensure execution in repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/../cuda code.py" ]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    cd "$REPO_ROOT"
fi
echo "[INFO] Repository Root: $(pwd)"

# 1. Verify Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed!"
    echo "Install it via: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

PY_VER=$(python3 --version)
echo "[INFO] Detected: $PY_VER"

# 2. Check Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[INFO] Existing virtual environment found (.venv)."
fi

# 3. Activate Virtual Environment
source .venv/bin/activate

# 4. Install Dependencies
REQ_FILE="setup/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="requirements.txt"
fi

echo ""
echo "[INFO] Installing Python dependencies from $REQ_FILE..."
python -m pip install --upgrade pip --quiet
pip install -r "$REQ_FILE"

# 5. Default param.dat
if [ ! -f "param.dat" ]; then
    if [ -f "setup/param.dat.example" ]; then
        echo "[INFO] Copying default param.dat configuration from setup/param.dat.example..."
        cp setup/param.dat.example param.dat
    elif [ -f "param.dat.example" ]; then
        cp param.dat.example param.dat
    fi
fi

# 6. Check GFortran
if ! command -v gfortran &> /dev/null; then
    echo ""
    echo "[NOTE] GFortran not found in PATH."
    echo "       If you wish to compile the Fortran CPU solvers and post-processing tools:"
    echo "       Run: sudo apt update && sudo apt install -y gfortran build-essential"
fi

# 7. Run Verification Check
echo ""
echo "==============================================================================="
echo " Running System & CUDA Verification Diagnostic"
echo "==============================================================================="
if [ -f "setup/check_setup.py" ]; then
    python setup/check_setup.py
else
    python check_setup.py
fi

echo ""
echo "==============================================================================="
echo " Setup complete! In future terminal sessions, activate with:"
echo "     source .venv/bin/activate"
echo "==============================================================================="
