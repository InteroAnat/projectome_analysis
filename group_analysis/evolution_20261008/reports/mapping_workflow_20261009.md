# Reproduce the multi-monkey descriptive maps

**Historical reproduction:** these commands retain the original 436/26 source partitions and selection-based display labels. The current [primary ARM workflow](../arm_mapping_20261009/README.md) uses official source-region names and defaults to one group figure set. The [six-level workbook](../../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/README.md) combines all 462 identities without inventing additional biological cohorts.

Run from `D:\projectome_analysis` in the existing `projectome` environment. These commands reuse audited local reconstructions and dated evidence; they do not fetch sources, register brains, alter labels or promote cohorts. Full live inventory refresh is a separate operation. Required versions and all source hashes are in the saved run provenance.

Use a **fresh short output root**. The public entrypoints reject existing destinations. Source paths in the saved manifests are absolute; relocating the dataset requires a new source inventory rather than editing hashes. The only parameter below to change for an identical local repeat is `$repeat`.

```powershell
$python = 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe'
$base = 'group_analysis/evolution_20261008'
$audit = "$base/classification/coarse_insula_review_20261009"
$triage = "$audit/distance_priority_v2_20261009"
$nmt = 'atlas/NMT_v2.1_sym/NMT_v2.1_sym'
$reference = "$nmt/NMT_v2.1_sym_SS.nii.gz"
$mask = "$nmt/NMT_v2.1_sym_brainmask.nii.gz"
$repeat = "$base/repeat_20261009_01"
if (Test-Path -LiteralPath $repeat) { throw 'Choose a fresh repeat directory' }

# Stop after any failed Python entrypoint; no dependent step uses a failed result.
function Invoke-ProjectionStep {
    param([string[]]$StepArguments)
    & $python -X utf8 -B @StepArguments
    if ($LASTEXITCODE -ne 0) { throw "Projection step failed: $($StepArguments[0])" }
}

Invoke-ProjectionStep -StepArguments @(
    'group_analysis/scripts/prepare_inventory_projection_manifest.py',
    '--sources', "$audit/map_ready_sources.csv",
    '--inventory-provenance', "$audit/provenance.json",
    '--inventory-delivery', "$audit/delivery_provenance_20261009.json",
    '--animal-map', 'group_analysis/docs/dataset_status_manifest.csv',
    '--reference', $reference, '--output', "$repeat/main_inputs"
)
Invoke-ProjectionStep -StepArguments @(
    'group_analysis/scripts/prepare_review_candidate_manifest.py',
    '--sources', "$audit/map_ready_sources.csv",
    '--inventory-delivery', "$audit/delivery_provenance_20261009.json",
    '--triage', "$triage/priority_review_manifest_distance_corrected.csv",
    '--triage-provenance', "$triage/correction_provenance.json",
    '--main-manifest', "$repeat/main_inputs/projection_manifest.csv",
    '--animal-map', 'group_analysis/docs/dataset_status_manifest.csv',
    '--reference', $reference, '--output', "$repeat/preview_inputs"
)

foreach ($cohort in @('main', 'preview')) {
    $manifest = "$repeat/${cohort}_inputs/projection_manifest.csv"
    $endpoints = "$repeat/${cohort}_endpoints"
    $axons = "$repeat/${cohort}_axons"
    Invoke-ProjectionStep -StepArguments @(
        'group_analysis/scripts/build_endpoint_maps.py',
        '--manifest', $manifest, '--reference', $reference, '--brain-mask', $mask,
        '--atlas-path', "$nmt/ARM_in_NMT_v2.1_sym.nii.gz",
        '--atlas-key', 'atlas/ARM_key_all.txt',
        '--hemisphere-mask', "$nmt/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz",
        '--output', $endpoints
    )
    Invoke-ProjectionStep -StepArguments @(
        'group_analysis/scripts/review_endpoint_run.py',
        '--run', $endpoints, '--output', "${endpoints}_review"
    )
    Invoke-ProjectionStep -StepArguments @(
        'group_analysis/scripts/build_projection_maps.py',
        '--manifest', $manifest, '--reference', $reference,
        '--brain-mask', $mask, '--output', $axons
    )
    Invoke-ProjectionStep -StepArguments @(
        'group_analysis/scripts/review_projection_run.py',
        '--run', $axons, '--output', "${axons}_review"
    )
    foreach ($metric in @('endpoint-density', 'axon-density')) {
        $run = if ($metric -eq 'axon-density') { $axons } else { $endpoints }
        $reviewName = if ($metric -eq 'axon-density') {
            'projection_run_readback.json'
        } else { 'endpoint_run_readback.json' }
        foreach ($scope in @('animal', 'group')) {
            Invoke-ProjectionStep -StepArguments @(
                'group_analysis/scripts/render_projection_slices.py',
                '--run', $run, '--readback', "${run}_review/$reviewName",
                '--metric', $metric, '--scope', $scope,
                '--cut-policy', 'fixed', '--slice-voxels', '56', '200', '87',
                '--output', "$repeat/figures/${cohort}_${metric}_${scope}"
            )
        }
    }
}
Invoke-ProjectionStep -StepArguments @(
    'group_analysis/scripts/render_projection_slices.py',
    '--run', "$repeat/main_endpoints",
    '--readback', "$repeat/main_endpoints_review/endpoint_run_readback.json",
    '--metric', 'endpoint-occupancy', '--scope', 'animal',
    '--cut-policy', 'fixed', '--slice-voxels', '56', '200', '87',
    '--output', "$repeat/figures/main_endpoint-occupancy_animal"
)
```

Expected partitions: main 436 unique neurons / eight animals; preview 26 / five animals, with no UID or source-path overlap. Compare exact source identity/hash rows and numerical grids, not timestamp-containing provenance bytes. Independent reviews must report `passed`. Gzip/container or metadata bytes can differ on a new run while numerical content agrees. Endpoint denominators are 403 main and 26 preview; axon averages use all selected neurons. Read the selected/computable distinction before interpreting zeros.

The complete PowerShell block passes the PowerShell syntax parser. Its individual public preparation/build/review/render entrypoints were executed for the saved delivery; exact receipts are linked below. The dated [delivery checker](../validation/check_delivery_20261009.py) can recheck original saved sources/maps/figures without recomputing them: `python -X utf8 -B group_analysis/evolution_20261008/validation/check_delivery_20261009.py --output group_analysis/evolution_20261008/validation/delivery_readback_repeat01.json`. Choose a new receipt filename; this checker targets the original delivery rather than the new repeat directory.

To render saved maps without recomputing them, use `render_projection_slices.py` directly with their matching successful review: main endpoints use `endpoint_maps/multimonkey_coarse_20261009_review_v2/endpoint_run_readback.json`; the main axon and both preview runs use the corresponding `_review` directory. Use a fresh figure output directory. The renderer verifies matching run/input/output hashes before displaying data. Original MIPs and older layouts remain preserved; current sheets are indexed in the [main figure guide](../figures/multimonkey_coarse_20261009_layout-v2/README.md).

The [existing CLI receipts](../validation/additional_candidate_maps_cli_20261009.json) record the executed public build/review commands. [Main axon receipt](../validation/multimonkey_axon_cli_20261009.json), [main endpoint successful review](../validation/multimonkey_endpoint_review_v2_cli_20261009.json) and [final display receipt](../validation/matched_slices_layout_v2_cli_20261009.json) retain their exact arguments and outcomes. The first main endpoint review's blank-label failure is preserved and superseded by the focused fix and successful v2 review.

Software checks: `python -X utf8 -B -m unittest discover -s tests -v`. New manifests retain source, evidence, geometry and anatomical/terminal status fields. Published terminal-arbor detection, anatomical registration acceptance, statistical inference and paired MSTIM analysis are not implemented by this repeat command; their dependencies are listed in the [status report](evolution_status_20261009.md).
