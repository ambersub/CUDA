@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo  3D Compressible CFD Solver - Automated Windows Device Setup
echo ===============================================================================
echo.

:: 1. Verify Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 (64-bit) from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo [INFO] Detected: %PY_VER%

:: 2. Create or verify Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Existing virtual environment found (.venv).
)

:: 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 4. Upgrade pip and install requirements
echo.
echo [INFO] Installing required Python libraries from requirements.txt...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARN] Pip encountered an issue. Attempting direct install of core CUDA packages...
    pip install numpy numba numba-cuda nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "cuda-bindings<13" pandas scipy matplotlib tqdm
)

:: 5. Copy param.dat template if missing
if not exist "param.dat" (
    if exist "param.dat.example" (
        echo [INFO] Initializing default param.dat configuration...
        copy param.dat.example param.dat >nul
    )
)

:: 6. Run comprehensive verification check
echo.
echo ===============================================================================
echo  Running System & CUDA Verification Diagnostic
echo ===============================================================================
python check_setup.py

echo.
echo ===============================================================================
echo  Setup Completed! To activate your environment in the future:
echo     .venv\Scripts\activate
echo ===============================================================================
pause
