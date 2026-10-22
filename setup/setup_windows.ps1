# ===============================================================================
# 3D Compressible CFD Solver - Automated PowerShell Device Setup
# ===============================================================================

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host " 3D Compressible CFD Solver - Automated PowerShell Device Setup" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure working directory is the repository root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (Test-Path (Join-Path $ScriptDir "..\cuda code.py")) {
    $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
} else {
    $RepoRoot = (Get-Location).Path
}
Set-Location $RepoRoot
Write-Host "[INFO] Repository Root: $RepoRoot" -ForegroundColor Green

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
if (Test-Path ".venv") {
    & .\.venv\Scripts\python.exe -c "import sys" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Existing .venv appears corrupted or copied from another machine with different user paths." -ForegroundColor Yellow
        Write-Host "[INFO] Re-creating clean virtual environment for this machine..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force ".venv"
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Could not create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Existing virtual environment found (.venv) and verified working." -ForegroundColor Green
}

# 3. Activate Virtual Environment
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# 4. Install Dependencies
$reqFile = if (Test-Path "setup\requirements.txt") { "setup\requirements.txt" } else { "requirements.txt" }
Write-Host "`n[INFO] Installing dependencies from $reqFile..." -ForegroundColor Cyan
if (Test-Path "setup\wheels") {
    Write-Host "[INFO] Offline wheel cache detected in setup\wheels. Installing packages..." -ForegroundColor Green
    & python -m pip install --no-index --find-links="setup\wheels" -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Offline install had issues, falling back to online PyPI..." -ForegroundColor Yellow
        & python -m pip install -r $reqFile
    }
} else {
    & python -m pip install -r $reqFile
}

# 5. Default param.dat
if (-not (Test-Path "param.dat")) {
    if (Test-Path "setup\param.dat.example") {
        Write-Host "[INFO] Copying default param.dat configuration from setup\param.dat.example..." -ForegroundColor Yellow
        Copy-Item "setup\param.dat.example" "param.dat"
    } elseif (Test-Path "param.dat.example") {
        Copy-Item "param.dat.example" "param.dat"
    }
}

# 6. Run verification
Write-Host "`n===============================================================================" -ForegroundColor Cyan
Write-Host " Running System & CUDA Verification Diagnostic" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
if (Test-Path "setup\check_setup.py") {
    & python setup\check_setup.py
} else {
    & python check_setup.py
}

Write-Host "`nSetup process completed." -ForegroundColor Green
Write-Host "To activate in the future: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
