"""Read-only CM033 SPM/source audit; writes only a new local JSON receipt."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import subprocess

import h5py
import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
EXTERNAL = Path('C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009')
MODEL = Path('D:/multimodal_fmri/derivatives/glm/sub-cm033/ses-20160120/glm-mstim-pos1-short-tr3-smooth2mm')
EXPECTED = {57: 160, 58: 160, 60: 384, 61: 384, 64: 384}


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path):
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def string(dataset):
    return ''.join(chr(int(value)) for value in dataset[()].flat)


def scalar(dataset):
    values = dataset[()].flatten()
    if len(values) != 1:
        raise ValueError('Expected scalar SPM field')
    return float(values[0])


def main():
    output = HERE / 'spm_bridge_input_audit.json'
    if output.exists():
        raise FileExistsError(output)
    result = {'started_utc': datetime.now(timezone.utc).isoformat()}
    result['code'] = binding(Path(__file__))
    result['SPM'] = binding(MODEL / 'SPM.mat')
    records = {}
    matrices = []
    with h5py.File(MODEL / 'SPM.mat', 'r') as handle:
        volumes = handle['SPM/xY/VY']
        result['TR_seconds'] = scalar(handle['SPM/xY/RT'])
        for index, reference in enumerate(volumes['fname'][()].flat):
            path = Path(string(handle[reference]))
            scanner_id = int(re.search(r'run-(\d+)_', path.name).group(1))
            if scanner_id not in EXPECTED:
                raise ValueError(f'Unexpected model run {scanner_id}')
            matrix = handle[volumes['mat'][()].flat[index]][()].T
            shift = np.eye(4)
            shift[:3, 3] = 1
            affine = matrix @ shift
            matrices.append(affine)
            n = handle[volumes['n'][()].flat[index]][()].flatten()
            info = handle[volumes['pinfo'][()].flat[index]][()].flatten()
            private = handle[volumes['private'][()].flat[index]]
            dat = private['dat']
            descriptor = {
                'filename': str(Path(string(dat['fname']))),
                'shape': dat['dim'][()].flatten().astype(int).tolist(),
                'nifti_datatype': scalar(dat['dtype']),
                'offset': scalar(dat['offset']),
                'slope': scalar(dat['scl_slope']),
                'intercept': scalar(dat['scl_inter']),
                'big_endian': scalar(dat['be']),
            }
            record = records.setdefault(scanner_id, {
                'path': str(path), 'cached_descriptor': descriptor,
                'volume_numbers': [], 'pinfo_offsets': [],
            })
            if record['cached_descriptor'] != descriptor:
                raise ValueError('Inconsistent cached file-array descriptors')
            if descriptor['filename'] != str(path) or len(n) != 2 or n[1] != 1:
                raise ValueError('Unexpected SPM file-array identity/volume indexing')
            if not np.allclose(info[:2], [descriptor['slope'], descriptor['intercept']], rtol=0, atol=1e-12):
                raise ValueError('Cached SPM scaling differs from private descriptor')
            record['volume_numbers'].append(int(n[0]))
            record['pinfo_offsets'].append(float(info[2]))
        affine = matrices[0]
        if len(matrices) != sum(EXPECTED.values()) or not all(np.array_equal(a, affine) for a in matrices):
            raise ValueError('Unexpected model volume count or varying cached geometry')
        result['all1472_cached_zero_based_affines_exactly_equal'] = True
        result['cached_zero_based_affine'] = affine.tolist()
        result['xVol_M_matches_VY_exactly'] = bool(np.array_equal(handle['SPM/xVol/M'][()].T @ shift, affine))

    result['runs'] = []
    mean_sum = np.zeros((128, 128, 18), dtype=np.float64)
    for scanner_id, expected_count in EXPECTED.items():
        record = records[scanner_id]
        path = Path(record['path'])
        image = nib.load(path)
        descriptor = record['cached_descriptor']
        if image.shape != tuple(descriptor['shape']) or image.shape[-1] != expected_count:
            raise ValueError('Current payload dimensions differ from cached SPM')
        if record.pop('volume_numbers') != list(range(1, expected_count + 1)):
            raise ValueError('Missing, repeated, or reordered SPM input volume')
        proxy = image.dataobj
        current = {
            'filename': str(path), 'shape': list(image.shape),
            'nifti_datatype': float(image.header['datatype']),
            'offset': float(proxy.offset), 'slope': float(proxy.slope),
            'intercept': float(proxy.inter),
            'big_endian': float(image.header.endianness == '>'),
        }
        structural_keys = ['filename', 'shape', 'nifti_datatype', 'offset', 'big_endian']
        if any(current[key] != descriptor[key] for key in structural_keys):
            raise ValueError('Current payload layout differs from cached SPM')
        record['current_descriptor'] = current
        record['cached_scaling_matches_current'] = bool(
            current['slope'] == descriptor['slope'] and current['intercept'] == descriptor['intercept']
        )
        bytes_per_frame = int(np.prod(image.shape[:3])) * image.get_data_dtype().itemsize
        offsets = record.pop('pinfo_offsets')
        expected_offsets = [proxy.offset + k * bytes_per_frame for k in range(expected_count)]
        if offsets != expected_offsets:
            raise ValueError('Cached SPM per-frame byte offsets do not match current payload')
        record['source_binding'] = binding(path)
        payload = hashlib.sha256()
        with path.open('rb') as stream:
            stream.seek(proxy.offset)
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                payload.update(chunk)
        record['current_payload_to_EOF_sha256'] = payload.hexdigest()
        record['current_affine'] = image.affine.tolist()
        record['current_header_matches_cached_geometry'] = bool(np.allclose(image.affine, affine, rtol=0, atol=1e-6))
        record['current_to_cached_world_coordinate_change'] = (affine @ np.linalg.inv(image.affine)).tolist()
        record['cached_to_current_world_coordinate_change'] = (image.affine @ np.linalg.inv(affine)).tolist()
        finite = True
        minimum, maximum = np.inf, -np.inf
        for k in range(expected_count):
            frame = np.asarray(proxy[..., k], dtype=np.float64)
            finite &= bool(np.isfinite(frame).all())
            minimum = min(minimum, float(np.nanmin(frame)))
            maximum = max(maximum, float(np.nanmax(frame)))
            mean_sum += frame
        record['all_current_scaled_voxels_finite'] = finite
        record['scaled_range'] = [minimum, maximum]
        record['cached_file_layout_and_frame_offsets_match_current'] = True
        record['cached_scale_must_not_be_applied_to_current_stored_floats'] = not record['cached_scaling_matches_current']
        record['scanner_id'] = scanner_id
        result['runs'].append(record)

    if not all(record['all_current_scaled_voxels_finite'] for record in result['runs']):
        raise ValueError('Nonfinite source values prevent a mean reference')
    mean_path = HERE / 'sub-cm033_ses-20160120_desc-currentPayload1472VolumeMean_space-cachedSPM_bold.nii.gz'
    if mean_path.exists():
        raise FileExistsError(mean_path)
    mean_values = (mean_sum / sum(EXPECTED.values())).astype(np.float32)
    mean_image = nib.Nifti1Image(mean_values, affine)
    mean_image.header.set_xyzt_units('mm')
    mean_image.set_qform(affine, code=2)
    mean_image.set_sform(affine, code=2)
    nib.save(mean_image, mean_path)
    readback = nib.load(mean_path)
    if not np.array_equal(readback.get_fdata(dtype=np.float32), mean_values):
        raise ValueError('Mean reference saved-array readback differs')
    result['prepared_reference'] = {
        **binding(mean_path), 'shape': list(readback.shape), 'affine': readback.affine.tolist(),
        'method': 'Unweighted mean over all 1472 current scaled model-input volumes; float64 accumulation then float32 storage; no array reversal or spatial interpolation.',
        'saved_array_readback_exact': True,
        'cached_geometry_matches_atol_1e_6': bool(np.allclose(readback.affine, affine, rtol=0, atol=1e-6)),
        'physical_laterality_validated': False,
    }

    result['statistics'] = []
    for name in ['con_0001.nii', 'spmT_0001.nii', 'mask.nii']:
        path = MODEL / 'maps' / name
        image = nib.load(path)
        values = image.get_fdata(dtype=np.float32)
        result['statistics'].append({**binding(path), 'shape': list(image.shape),
            'affine': image.affine.tolist(), 'matches_cached_zero_based_affine': bool(np.allclose(image.affine, affine, rtol=0, atol=1e-6)),
            'all_values_finite': bool(np.isfinite(values).all()), 'nonzero_voxels': int(np.count_nonzero(values))})
    result['source_records'] = []
    for path in [MODEL / 'glm_config_summary.txt', MODEL / 'contrast_qc.json', MODEL / 'contrasts.json',
        Path('D:/multimodal_fmri/derivatives/preproc/sub-cm033/ses-20160120/qc/nii_reorientation_20260907_135316.json'),
        Path('D:/multimodal_fmri/code/tools/preproc/nii_reorientation.py'),
        EXTERNAL / 'source_manifest.json', EXTERNAL / 'normalization_completed.json',
        EXTERNAL / 'normalization_qc/normalization_checks.json', EXTERNAL / 'run_spatial_rebuild.py',
        EXTERNAL / 'qc_functional_normalization.py']:
        result['source_records'].append(binding(path))
    result['current_external_receipts'] = {}
    for name in ['spatial_rebuild_completed.json', 'full_series_qc/full_series_checks.json', 'functional_normalization_qc/chain_checks.json']:
        path = EXTERNAL / name
        result['current_external_receipts'][name] = binding(path) if path.exists() else None
    process = subprocess.run(['wsl.exe', '-e', 'ps', '-p', '405', '-o', 'pid,ppid,etimes,stat,args'], capture_output=True, text=True)
    result['PID405_snapshot'] = {'returncode': process.returncode, 'stdout': process.stdout.strip(), 'stderr': process.stderr.strip()}
    result['ended_utc'] = datetime.now(timezone.utc).isoformat()
    result['scope'] = 'Current payload descriptor/frame-index/finite-value verification; no original July payload hash exists in inspected records; no registration or physical laterality acceptance.'
    result['original_estimation_payload_byte_identity_independently_proven'] = False
    result['scientific_anatomical_acceptance'] = False
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': str(output), 'sha256': digest(output), 'runs': len(result['runs']), 'volumes': len(matrices), 'PID405': result['PID405_snapshot'], 'external_receipts': result['current_external_receipts']}, indent=2))


if __name__ == '__main__':
    main()
