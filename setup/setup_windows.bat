@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo  3D Compressible CFD Solver - Automated Windows Device Setup
echo ===============================================================================
echo.

:: Ensure execution always takes place in repository root
cd /d "%~dp0.."
echo [INFO] Working Directory: %CD%

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
if exist ".venv" (
    :: Test if existing .venv actually works or has invalid paths from another machine
    .venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Existing .venv appears corrupted or copied from another machine with different user paths.
        echo [INFO] Re-creating clean virtual environment for this machine...
        rmdir /s /q .venv
    )
)

if not exist ".venv" (
    echo [INFO] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Existing virtual environment found (.venv) and verified working.
)

:: 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 4. Upgrade pip and install requirements
echo.
echo [INFO] Installing required Python libraries...
if exist "setup\wheels" (
    echo [INFO] Offline wheel cache detected in setup\wheels. Installing offline...
    python -m pip install --no-index --find-links=setup\wheels -r setup\requirements.txt
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Offline wheel installation had warnings, trying standard installation...
        python -m pip install -r setup\requirements.txt
    )
) else if exist "setup\requirements.txt" (
    python -m pip install -r setup\requirements.txt
) else if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo [WARN] requirements.txt not found. Installing core CUDA packages...
    pip install numpy numba numba-cuda nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "cuda-bindings<13" pandas scipy matplotlib tqdm
)

:: 5. Copy param.dat template if missing
if not exist "param.dat" (
    if exist "setup\param.dat.example" (
        echo [INFO] Initializing default param.dat configuration...
        copy setup\param.dat.example param.dat >nul
    ) else if exist "param.dat.example" (
        copy param.dat.example param.dat >nul
    )
)

:: 6. Run comprehensive verification check
echo.
echo ===============================================================================
echo  Running System & CUDA Verification Diagnostic
echo ===============================================================================
if exist "setup\check_setup.py" (
    python setup\check_setup.py
) else (
    python check_setup.py
)

echo.
echo ===============================================================================
echo  Setup Completed! To activate your environment in the future:
echo     .venv\Scripts\activate
echo ===============================================================================
pause
