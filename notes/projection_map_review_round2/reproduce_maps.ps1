param(
    [string]$Python = "C:/Users/laika_yan/miniconda3/envs/projectome/python.exe",
    [Parameter(Mandatory = $true)][string]$Output,
    [switch]$Rebuild
)
# Run from the checkout root. Stop on failed native commands as well as PS errors.
$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if ((Get-Location).Path -ne $repository) { throw "Run this procedure from $repository" }
if (Test-Path -LiteralPath $Output) { throw "Use a fresh output directory" }
New-Item -ItemType Directory -Path $Output | Out-Null
function Run-Python([string[]]$Arguments) {
    & $Python -X utf8 -B @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python procedure failed: $($Arguments[0])" }
}
$review = "notes/region_analysis_review_20261009"
$evolution = "group_analysis/evolution_20261008"
$reference = "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"
$brainMask = "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_brainmask.nii.gz"
$atlas = "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
$atlasKey = "atlas/ARM_key_all.txt"
$ledger = "$review/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv"
$runs = [ordered]@{
    main_endpoints = "$evolution/arm_mapping_20261009/main/endpoints"
    additional_endpoints = "$evolution/endpoint_maps/additional_candidates_20261009"
    main_axons = "$evolution/arm_mapping_20261009/main/axons"
    additional_axons = "$evolution/projection_maps/additional_candidates_20261009"
    end_branches = "$review/axon_end_branches_20261009/selected462_ARM"
}
if ($Rebuild) {
    $oldRuns = @{}
    foreach ($key in $runs.Keys) { $oldRuns[$key] = $runs[$key] }
    foreach ($name in @("main", "additional")) {
        $manifest = Join-Path $oldRuns["${name}_endpoints"] "input_manifest.csv"
        $runs["${name}_endpoints"] = Join-Path $Output "${name}_endpoints"
        $runs["${name}_axons"] = Join-Path $Output "${name}_axons"
        Run-Python @("group_analysis/scripts/build_endpoint_maps.py", "--manifest", $manifest, "--reference", $reference, "--atlas-path", $atlas, "--atlas-key", $atlasKey, "--hemisphere-mask", "atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz", "--brain-mask", $brainMask, "--output", $runs["${name}_endpoints"])
        Run-Python @("group_analysis/scripts/build_projection_maps.py", "--manifest", $manifest, "--reference", $reference, "--brain-mask", $brainMask, "--output", $runs["${name}_axons"])
        Run-Python @("group_analysis/scripts/review_endpoint_run.py", "--run", $runs["${name}_endpoints"], "--output", (Join-Path $Output "${name}_endpoint_readback"))
        Run-Python @("group_analysis/scripts/review_projection_run.py", "--run", $runs["${name}_axons"], "--output", (Join-Path $Output "${name}_axon_readback"))
    }
    $runs["end_branches"] = Join-Path $Output "end_branches"
    Run-Python @("group_analysis/scripts/build_axon_end_branch_maps.py", "--manifest", $ledger, "--reference", $reference, "--atlas", $atlas, "--atlas-key", $atlasKey, "--output", $runs["end_branches"])
    Run-Python @("$review/axon_end_branches_20261009/independent_end_branch_readback.py", "--run", $runs["end_branches"], "--receipt", (Join-Path $Output "independent_end_branch_readback.json"))
    Run-Python @("group_analysis/scripts/export_arm_projection_tables.py", "--endpoint-run", $runs["main_endpoints"], "--axon-run", $runs["main_axons"], "--additional-endpoint-run", $runs["additional_endpoints"], "--additional-axon-run", $runs["additional_axons"], "--manifest", $ledger, "--output", (Join-Path $Output "ARM_hierarchy_tables"))
}
$configuration = Join-Path $Output "runs.json"
$runs | ConvertTo-Json | Set-Content -LiteralPath $configuration -Encoding utf8
$inspection = Join-Path $Output "inspection"
Run-Python @("notes/projection_map_review_round2/inspect_projection_maps.py", "--runs-json", $configuration, "--output", $inspection)
Run-Python @("notes/projection_map_review_round2/render_endpoint_markers.py", "--run", $runs["main_endpoints"], "--inspection", (Join-Path $inspection "inspection_receipt.json"), "--output", (Join-Path $Output "endpoint_markers"))
