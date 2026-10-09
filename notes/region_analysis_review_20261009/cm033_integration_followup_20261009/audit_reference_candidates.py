"""Bounded historical-reference and current selected-source readback."""

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

import nibabel as nib
import numpy as np

from audit_spm_bridge_inputs import EXTERNAL, EXPECTED, HERE, binding


def windows_path(path):
    if path.startswith('/mnt/d/'):
        return Path('D:/' + path[7:])
    if path.startswith('/mnt/c/'):
        return Path('C:/' + path[7:])
    return Path(path)


def main():
    output = HERE / 'historical_reference_search.json'
    if output.exists():
        raise FileExistsError(output)
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'code': binding(Path(__file__))}
    prior = json.loads((HERE / 'spm_bridge_input_audit.json').read_text(encoding='utf-8'))
    cached = np.asarray(prior['cached_zero_based_affine'])
    base = Path('D:/multimodal_fmri')
    archive = base / '_archive/cm033_redo_20260725_115901'
    pre = base / 'derivatives/preproc/sub-cm033/ses-20160120'
    result['bounded_search_roots'] = [str(archive), str(pre), str(base / 'derivatives/norm/sub-cm033'), str(base / 'derivatives/glm/sub-cm033/ses-20160120/glm-mstim-pos1-short-tr3-smooth2mm')]
    result['archive_selected_first_frame_checks'] = []
    for sid in EXPECTED:
        name = f'smoothed_sub-cm033_ses-20160120_task-mstim_acq-pos1_run-{sid}_desc-aligned_bold.nii'
        old_path, current_path = archive / 'preproc_func' / name, pre / 'func' / name
        old, current = nib.load(old_path), nib.load(current_path)
        a, b = np.asarray(old.dataobj[..., 0]), np.asarray(current.dataobj[..., 0])
        result['archive_selected_first_frame_checks'].append({
            'scanner_id': sid, 'archive': binding(old_path), 'current_source_binding': prior['runs'][list(EXPECTED).index(sid)]['source_binding'],
            'archive_affine_matches_cachedSPM': bool(np.allclose(old.affine, cached, rtol=0, atol=1e-6)),
            'first_frame_scaled_arrays_exact': bool(np.array_equal(a, b)),
            'first_frame_max_abs_difference': float(np.max(np.abs(a - b))),
            'first_frame_Pearson_r': float(np.corrcoef(a.flat, b.flat)[0, 1]),
            'scope': 'First-frame counterexample only; no historical payload equality claimed.',
        })
    current_ref = pre / 'anat/sub-cm033_ses-20160120_desc-refmean_epi.nii'
    current_values = nib.load(current_ref).get_fdata(dtype=np.float32)
    result['reference_candidates'] = []
    for path in [archive / 'preproc_anat/sub-cm033_ses-20160120_desc-refmean_epi.nii',
        archive / 'preproc_anat/sub-cm033_ses-20160120_desc-refmean_epi.nii.gz',
        archive / 'epi_QCed/ref_mean_epi.nii.gz', current_ref]:
        image = nib.load(path)
        values = image.get_fdata(dtype=np.float32)
        result['reference_candidates'].append({**binding(path), 'shape': list(image.shape), 'affine': image.affine.tolist(),
            'affine_matches_cachedSPM': bool(np.allclose(image.affine, cached, rtol=0, atol=1e-6)),
            'all_values_finite': bool(np.isfinite(values).all()),
            'full_scaled_array_matches_current_reference': bool(np.array_equal(values, current_values)),
            'max_abs_difference_from_current_reference': float(np.max(np.abs(values - current_values)))})
    result['archive_reference_geometry_gate'] = binding(archive / 'preproc_anat/refmean_vs_aligned_gate.json')
    result['archive_same_grid_is_not_current_estimation_payload_proof'] = True
    manifest = json.loads((EXTERNAL / 'source_manifest.json').read_text(encoding='utf-8'))
    result['current_rebuild_selected_sources'] = []
    for record in manifest['runs']:
        sid = record['scanner_id']
        if sid not in EXPECTED:
            continue
        source = windows_path(record['source'])
        actual = binding(source)
        if actual['sha256'] != record['source_sha256']:
            raise ValueError('D source no longer matches current source-manifest hash')
        image = nib.load(source)
        if image.shape[-1] != EXPECTED[sid] or not np.isclose(image.header.get_zooms()[3], 3):
            raise ValueError('Current selected source frame count/TR differ from saved model cohort')
        result['current_rebuild_selected_sources'].append({
            'scanner_id': sid, 'legacy_run': record['legacy_run'], 'task': record['task'],
            'source': actual, 'shape': list(image.shape), 'TR_seconds': float(image.header.get_zooms()[3]),
            'staged_path': record['staged'], 'source_hash_matches_manifest': True,
            'staged_payload_equality_producer_claim': record['payload_byte_identical'],
            'selected_scanner_identity_and_frame_count_match_old_model': True,
            'old_smoothed_vs_new_alignment_payload_equality_established': False,
        })
    if len(result['current_rebuild_selected_sources']) != 5:
        raise ValueError('Selected source coverage is not exactly five model scans')
    repair = json.loads(Path(prior['source_records'][3]['path']).read_text(encoding='utf-8'))
    result['Sept7_repair_receipt'] = {
        'binding': prior['source_records'][3], 'version': repair['version'],
        'selected_paths_present': all(any(Path(r['path']) == Path(record['source_binding']['path']) for r in repair['files']) for record in prior['runs']),
        'geometry_gate_passed': repair['gate_passed'],
        'per_selected_path_payload_hash_or_voxel_equality_check_recorded': False,
        'implementation': prior['source_records'][4],
        'implementation_scope': 'write_one loads scaled data, writes copied header/image/temp and replaces; session gate checks geometry. Separate smoke only checks first/last frames of small copied examples, not all installed selected model payloads.'}
    log = EXTERNAL / 'spatial_rebuild.log'
    result['external_log_tail'] = log.read_text(encoding='utf-8', errors='replace')[-1800:]
    process = subprocess.run(['wsl.exe', '-e', 'ps', '-p', '405', '-o', 'pid,ppid,etimes,stat,args'], capture_output=True, text=True)
    result['PID405_snapshot'] = {'returncode': process.returncode, 'stdout': process.stdout.strip(), 'stderr': process.stderr.strip()}
    result['completion_receipts'] = {name: binding(EXTERNAL / name) if (EXTERNAL / name).exists() else None for name in ['spatial_rebuild_completed.json', 'full_series_qc/full_series_checks.json', 'functional_normalization_qc/chain_checks.json']}
    result['ended_utc'] = datetime.now(timezone.utc).isoformat()
    result['scientific_anatomical_acceptance'] = False
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': binding(output), 'PID405': result['PID405_snapshot'], 'completion': result['completion_receipts']}, indent=2))


if __name__ == '__main__':
    main()
