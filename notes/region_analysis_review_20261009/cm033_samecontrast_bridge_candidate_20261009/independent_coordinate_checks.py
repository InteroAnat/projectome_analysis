"""Saved-source validation using sequential named-space coordinate operations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent


def host(path):
    path = str(path)
    return Path(path[5] + ':' + path[6:]) if path.startswith('/mnt/') else Path(path)


def binding(path):
    path = host(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    output = HERE / 'independent_coordinate_checks.json'
    if output.exists():
        raise FileExistsError(output)
    provenance = HERE / 'bridge_candidate_provenance.json'
    data = json.loads(provenance.read_text())
    composition = json.loads((HERE / 'prospective_coordinate_composition.json').read_text())
    B = np.array(data['B_cached_to_declared_current_RAS_mm'])
    M = np.array(data['AFNI_pullback']['values'])
    P = np.array(composition['prospective_pullback_newRef_RAS_to_cached_RAS'])
    F = np.array(composition['prospective_pointmap_cached_RAS_to_newRef_RAS'])
    reference = nib.load(host(data['actual_scan39_reference']['path']))
    original = nib.load(host(data['cached_mean']['path']))
    work = nib.load(host(data['declared_current_mean']['path']))
    assert np.array_equal(np.asanyarray(original.dataobj), np.asanyarray(work.dataobj))
    assert np.array_equal(B @ original.affine, work.affine)
    rng = np.random.default_rng(98113)
    points = reference.affine @ np.column_stack([rng.uniform(0, 17, (256, 3)), np.ones(256)]).T
    sign = np.array([-1., -1., 1., 1.])[:, None]
    # Independent sequential point operations: new RAS -> new DICOM ->
    # declared DICOM -> declared RAS -> original cached RAS.
    declared_dicom = M @ (points * sign)
    cached = np.linalg.solve(B, declared_dicom * sign)
    assert np.allclose(P @ points, cached, atol=1e-10, rtol=0)
    assert np.allclose(F @ cached, points, atol=1e-10, rtol=0)
    D = np.diag(sign[:, 0])
    assert not np.allclose(D @ M @ D @ np.linalg.inv(B) @ points, cached)
    assert not np.allclose(np.linalg.inv(B) @ M @ points, cached)
    u, s, vt = np.linalg.svd(M[:3, :3])
    rotation = u @ vt
    degree = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    assert degree < 30 and abs(degree - data['residual_rotation_degrees_independent_SVD']) < 1e-10
    for item in [data[k] for k in ['cached_mean', 'declared_current_mean', 'actual_scan39_reference', 'aligned_mean', 'inverse_mean', 'AFNI_pullback', 'inverse_AFNI_matrix']]:
        assert binding(item['path'])['sha256'] == item['sha256']
    checks = {
        'exact_saved_scaled_array_equality': True,
        'explicit_B_same_voxel_coordinate_identity': True,
        'sequential_named_space_pullback_256_points': True,
        'forward_pointmap_256_inverse_roundtrips': True,
        'incorrect_order_and_missing_RAS_DICOM_conversion_detected': True,
        'independent_residual_gate_check': True,
        'seven_saved_source_artifact_hashes_match': True,
    }
    receipt = dict(status='pass', checked_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
                   residual_rotation_degrees=degree, total_historical_physical_rotation_claim=False,
                   code=binding(__file__), provenance=binding(provenance), composition=binding(HERE / 'prospective_coordinate_composition.json'))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'pass', 'checks': len(checks), 'receipt': binding(output)}))


if __name__ == '__main__':
    main()
