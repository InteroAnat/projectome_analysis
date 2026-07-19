# Download Liu et al. 2026 NeuroMorpho morphology bundle (~14 MB).
# Run from repo root or this directory:
#   pwsh external/liu2026/scripts/download_morph.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$morphDir = Join-Path $repoRoot "external\liu2026\morph"
$zipPath = Join-Path $morphDir "liu2026_morph.zip"
$url = "https://neuromorpho.org/dableFiles/liu_s/Supplementary/liu_s.zip"

New-Item -ItemType Directory -Force -Path $morphDir | Out-Null

if (Test-Path (Join-Path $morphDir "NeuroMorph_upload260215")) {
    Write-Host "Already extracted: $morphDir\NeuroMorph_upload260215"
    exit 0
}

Write-Host "Downloading Liu 2026 morphologies..."
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

Write-Host "Extracting to $morphDir ..."
Expand-Archive -Path $zipPath -DestinationPath $morphDir -Force

$venl = (Get-ChildItem -Path (Join-Path $morphDir "NeuroMorph_upload260215\PatchClamp_morph\VENL") -Filter "*.ASC" -ErrorAction SilentlyContinue).Count
$vens = (Get-ChildItem -Path (Join-Path $morphDir "NeuroMorph_upload260215\PatchClamp_morph\VENS") -Filter "*.ASC" -ErrorAction SilentlyContinue).Count
Write-Host "Done. VEN-L: $venl ASC | VEN-S: $vens ASC"
Write-Host "Update registry status: external/ven_validation/registry.tsv -> liu2026_aic_ven status=downloaded"
