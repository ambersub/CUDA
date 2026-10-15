# ===============================================================================
# 3D Compressible CFD Solver - Automated PowerShell Device Setup
# ===============================================================================

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host " 3D Compressible CFD Solver - Automated PowerShell Device Setup" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10 - 3.12 (64-bit) from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

$pyVersion = & python --version 2>&1
Write-Host "[INFO] Detected: $pyVersion" -ForegroundColor Green

# 2. Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Could not create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Existing virtual environment found (.venv)." -ForegroundColor Green
}

# 3. Activate Virtual Environment
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# 4. Install Dependencies
Write-Host "`n[INFO] Upgrading pip and installing requirements.txt..." -ForegroundColor Cyan
& python -m pip install --upgrade pip --quiet
& pip install -r requirements.txt

# 5. Default param.dat
if (-not (Test-Path "param.dat") -and (Test-Path "param.dat.example")) {
    Write-Host "[INFO] Copying default param.dat configuration..." -ForegroundColor Yellow
    Copy-Item "param.dat.example" "param.dat"
}

# 6. Run verification
Write-Host "`n===============================================================================" -ForegroundColor Cyan
Write-Host " Running System & CUDA Verification Diagnostic" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
& python check_setup.py

Write-Host "`nSetup process completed." -ForegroundColor Green
Write-Host "To activate in the future: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
