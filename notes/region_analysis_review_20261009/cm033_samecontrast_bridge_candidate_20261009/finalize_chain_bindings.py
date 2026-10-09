"""Freeze complete source aliases/QC; preserve the producer execution snapshot."""
from datetime import datetime, timezone
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from independent_coordinate_checks import binding, host

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / 'statistic_chain'
    receipt = out / 'chain_provenance.json'
    snapshot = out / 'chain_provenance_execution_snapshot.json'
    if snapshot.exists():
        raise FileExistsError('Already frozen; preserve all receipts')
    raw = receipt.read_bytes()
    data = json.loads(raw)
    assert data['status'] == 'software_completed_provisional_cm033_statistic_chain'
    snapshot.write_bytes(raw)
    candidate = json.loads((HERE / 'bridge_candidate_provenance.json').read_text())
    additional = [HERE / 'bridge_candidate_provenance.json', host(candidate['AFNI_pullback']['path']),
                  out / 'desc-taskAOutputToCachedInputComposite.aff12.1D',
                  HERE / 'prospective_coordinate_composition.json', HERE / 'prospective_statistic_chain.json',
                  HERE / 'independent_coordinate_checks.json']
    for path in additional:
        item = binding(path)
        data['source_bindings'].append(item)
        data['source_sha256'][str(path).replace('\\', '/')] = item['sha256']
    reference = nib.load(host(data['inputs']['reference']['path']))
    mask_path = host(data['outputs']['analysis_coverage']['path'])
    mask = np.asanyarray(nib.load(mask_path).dataobj)
    assert set(np.unique(mask)) <= {0, 1}
    valid = mask == 1
    qc = dict(model_mask_NMT_voxels=int(valid.sum()), model_mask_NMT_volume_mm3=float(valid.sum() * abs(np.linalg.det(reference.affine[:3, :3]))), continuous={})
    for key in ['contrast', 'T']:
        row = data['outputs'][key]
        assert binding(row['path'])['sha256'] == row['sha256']
        image = nib.load(host(row['path']))
        values = np.asanyarray(image.dataobj)
        assert image.shape == reference.shape and np.array_equal(image.affine, reference.affine)
        assert np.isfinite(values).all()
        qc['continuous'][key] = dict(min_inside_mask=float(values[valid].min()), max_inside_mask=float(values[valid].max()),
                                    positive_inside_mask=int(np.count_nonzero(values[valid] > 0)), negative_inside_mask=int(np.count_nonzero(values[valid] < 0)),
                                    zero_inside_mask=int(np.count_nonzero(values[valid] == 0)),
                                    nonzero_outside_validity=int(np.count_nonzero(values[~valid])), finite=True)
    data.update(metadata_freeze_utc=datetime.now(timezone.utc).isoformat(), execution_snapshot=binding(snapshot),
                metadata_freeze_code=binding(__file__), saved_NMT_output_QC=qc,
                independent_saved_transform_reproduction='pending; separate reviewer must bind this frozen receipt',
                root_and_Codex_candidate_image_review='provisional gross correspondence only; local/peripheral/smoothing differences remain',
                source_path_aliases='Primary producer uses WSL absolute paths; additional immutable local evidence may use Windows absolute paths. Each key is an exact recorded source string.')
    receipt.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': data['status'], 'receipt': binding(receipt), 'QC': qc}, indent=2))


if __name__ == '__main__':
    main()
