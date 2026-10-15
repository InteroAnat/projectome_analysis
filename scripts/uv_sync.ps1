# Install/sync Python packages with uv (fast pip-compatible resolver).
# Requires: conda env `projectome` (Python 3.10) from environment-minimal.yml
#
# Full setup:
#   .\scripts\setup_conda_env.ps1
#
# Sync only (env already exists):
#   .\scripts\uv_sync.ps1
#   .\scripts\uv_sync.ps1 -Gui
#
param(
    [switch]$Gui
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

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

function Get-ProjectomePython {
    foreach ($base in @(
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "Miniconda3")
    )) {
        $py = Join-Path $base "envs\projectome\python.exe"
        if (Test-Path $py) { return $py }
    }
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        return (Join-Path $env:CONDA_PREFIX "python.exe")
    }
    throw "projectome python not found. Run scripts/setup_conda_env.ps1 first."
}

$Uv = Get-Uv
$Py = Get-ProjectomePython

Write-Host "uv:     $Uv"
Write-Host "python: $Py"
Write-Host "Installing requirements.txt ..."
& $Uv pip install -r requirements.txt --python $Py

if ($Gui) {
    Write-Host "Installing requirements-gui.txt ..."
    & $Uv pip install -r requirements-gui.txt --python $Py
}

Write-Host "Done. Verify: python scripts/verify_projectome_env.py"
