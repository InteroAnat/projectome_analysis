# L3 launcher — pins Julia 1.11.5 + monkeyrec project + JULIA_CPU_THREADS=1
#
# Examples:
#   .\run_insula_flatmap.ps1 -Mode single -DryRun
#   .\run_insula_flatmap.ps1 -Mode multi -Force
#   .\run_insula_flatmap.ps1 -Mode single -Dated
#   .\run_insula_flatmap.ps1 -Help

[CmdletBinding()]
param(
    [ValidateSet('single', 'multi', 'lr_mirror')]
    [string]$Mode,

    [switch]$DryRun,
    [switch]$Force,
    [switch]$Dated,
    [switch]$Help,

    [int]$Niter = 30000,
    [string]$Soma,
    [string]$Sheet,
    [string]$Out,
    [string]$Cache,
    [string]$Atlas,
    [string]$Monkeyrec,

    [string]$Julia = 'julia',
    [string]$JuliaVersion = '1.11.5'
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Entry = Join-Path $ScriptDir 'run_insula_flatmap.jl'

if (-not $Monkeyrec) {
    $Monkeyrec = Join-Path $ProjectRoot 'references\analysis-code_gou_etal_2025\monkeyrec'
}

$jlArgs = @("+$JuliaVersion", "--project=$Monkeyrec", $Entry)

if ($Help) {
    $jlArgs += '--help'
} else {
    if (-not $Mode) {
        Write-Error 'Specify -Mode single|multi|lr_mirror (or -Help)'
    }
    $jlArgs += @('--mode', $Mode)
    if ($DryRun) { $jlArgs += '--dry-run' }
    if ($Force)  { $jlArgs += '--force' }
    if ($Dated)  { $jlArgs += '--dated' }
    if ($PSBoundParameters.ContainsKey('Niter')) { $jlArgs += @('--niter', "$Niter") }
    if ($Soma)      { $jlArgs += @('--soma', $Soma) }
    if ($Sheet)     { $jlArgs += @('--sheet', $Sheet) }
    if ($Out)       { $jlArgs += @('--out', $Out) }
    if ($Cache)     { $jlArgs += @('--cache', $Cache) }
    if ($Atlas)     { $jlArgs += @('--atlas', $Atlas) }
    if ($Monkeyrec) { $jlArgs += @('--monkeyrec', $Monkeyrec) }
}

$env:JULIA_CPU_THREADS = '1'
Write-Host "JULIA_CPU_THREADS=$env:JULIA_CPU_THREADS"
Write-Host "Running: $Julia $($jlArgs -join ' ')"
$ErrorActionPreference = 'Continue'
& $Julia @jlArgs 2>&1
$rc = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($rc -ne 0) { Write-Error "Julia exited with code $rc"; exit $rc }
exit 0
