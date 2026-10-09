"""Diagnose stored-array order/intensity scaling; never change image geometry."""
from pathlib import Path
import hashlib
import itertools
import json

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    source = HERE / 'mapping_and_normalization_snapshot.json'
    destination = HERE / 'storage_order_intensity_correspondence.json'
    if destination.exists():
        raise FileExistsError(destination)
    snapshot = json.loads(source.read_text(encoding='utf-8'))
    results = []
    for row in snapshot['scanner_ID_preserved_source_content_checks']:
        raw = np.asanyarray(nib.load(row['preserved_scanner_source']).dataobj)
        local = np.asanyarray(nib.load(row['local_source']).dataobj)
        frames = [0, raw.shape[3] // 2, raw.shape[3] - 1]
        comparisons = []
        for flags in itertools.product([False, True], repeat=3):
            candidate = local
            for axis, reverse in enumerate(flags):
                if reverse:
                    candidate = np.flip(candidate, axis=axis)
            a = raw[::4, ::4, ::2, frames].astype(float).ravel()
            v = candidate[::4, ::4, ::2, frames].astype(float).ravel()
            design = np.column_stack([v, np.ones_like(v)])
            slope, intercept = np.linalg.lstsq(design, a, rcond=None)[0]
            residual = a - (slope * v + intercept)
            comparisons.append({
                'storage_array_axis_reversals_xyz': list(flags),
                'sample_correlation': float(np.corrcoef(a, v)[0, 1]),
                'raw_from_local_slope': float(slope),
                'raw_from_local_intercept': float(intercept),
                'maximum_absolute_sample_residual': float(np.abs(residual).max()),
                'sample_residual_RMSE': float(np.sqrt(np.mean(residual ** 2))),
                'raw_sample_standard_deviation': float(a.std()),
                'sample_count': len(a),
            })
        comparisons.sort(key=lambda item: item['sample_residual_RMSE'])
        results.append({
            'scanner_id': row['scanner_id'], 'frames': frames,
            'sample_stride_xyz': [4, 4, 2], 'best': comparisons[0],
            'next_best': comparisons[1], 'all_storage_order_tests': comparisons,
            'interpretation': 'Storage-order/intensity diagnostic only; not physical orientation, registration or anatomical acceptance. No source was flipped or rewritten.',
        })
        del raw, local
    destination.write_text(json.dumps({
        'scope': 'Independent sampled array-order and affine intensity-scale correspondence to preserved scanner-ID sources',
        'source_snapshot_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'results': results,
        'scientific_anatomical_acceptance': False, 'source_writes': False,
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{k: row[k] for k in ['scanner_id', 'best', 'next_best']}
                      for row in results], indent=2))


if __name__ == '__main__':
    main()
