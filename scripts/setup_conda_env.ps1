# Phased projectome env setup: conda (Python 3.10) + uv (packages).
#
#   .\scripts\setup_conda_env.ps1
#   conda activate projectome
#   python scripts\verify_projectome_env.py
#
$ErrorActionPreference = "Stop"

function Get-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }
    foreach ($p in @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )) {
        if (Test-Path $p) { return $p }
    }
    throw "uv not found. Install: winget install astral-sh.uv"
}

$Conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
if (-not (Test-Path $Conda)) {
    $Conda = Join-Path $env:USERPROFILE "Miniconda3\Scripts\conda.exe"
}
if (-not (Test-Path $Conda)) {
    throw "Miniconda not found. Install: winget install Anaconda.Miniconda3"
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$EnvName = "projectome"
$Exists = & $Conda env list | Select-String "^\s*$EnvName\s"

if (-not $Exists) {
    Write-Host "Step 1/3: Creating minimal env (Python 3.10 + pip)..."
    & $Conda env create -f environment-minimal.yml
} else {
    Write-Host "Step 1/3: Env '$EnvName' already exists."
}

$Py = Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path $env:USERPROFILE "Miniconda3\envs\$EnvName\python.exe"
}
if (-not (Test-Path $Py)) {
    throw "Python not found for env '$EnvName' at $Py"
}

$Uv = Get-Uv
Write-Host "Step 2/3: Installing packages with uv..."
Write-Host "  uv:     $Uv"
Write-Host "  python: $Py"
& $Uv pip install -r requirements.txt --python $Py

Write-Host "Step 3/3: Done."
Write-Host ""
Write-Host "Activate:     conda activate $EnvName"
Write-Host "Verify:       python scripts/verify_projectome_env.py"
Write-Host "Re-sync:      .\scripts\uv_sync.ps1"
Write-Host "Optional GUI: .\scripts\uv_sync.ps1 -Gui"
