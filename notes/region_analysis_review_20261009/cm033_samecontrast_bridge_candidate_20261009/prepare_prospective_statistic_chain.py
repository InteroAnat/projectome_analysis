"""Bind and check a prospective chain; no statistic/mask resampling."""
from datetime import datetime, timezone
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from independent_coordinate_checks import binding, host

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXT = Path('C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009')


def main():
    output = HERE / 'prospective_statistic_chain.json'
    if output.exists():
        raise FileExistsError(output)
    candidate_path = HERE / 'bridge_candidate_provenance.json'
    candidate = json.loads(candidate_path.read_text())
    assert candidate['residual_rotation_gate_pass']
    pre = EXT / 'bids/sub-cm033/ses-20160120/derivatives/preproc/anat'
    refmean = pre / 'sub-cm033_ses-20160120_desc-refmean_bold.nii'
    taskA = EXT / 'anatomy_candidates/sub-cm033_srcdate-unknown_desc-taskAHeaderRepair_brain.nii'
    native = pre / 'sub-cm033_ses-20160120_desc-brain.nii'
    matrix = pre / 'sub-cm033_ses-20160120_desc-epiToAnatAffine.aff12.1D'
    installed = pre / 'sub-cm033_ses-20160120_desc-refmeanInAnat.nii'
    reproduced = HERE / 'installed_chain_readback/sub-cm033_desc-installedScan39FitReproduced_space-taskA_bold.nii.gz'
    assert binding(matrix)['sha256'] == 'b110f6baef72a7c50af604d66ac2bb6264216b1af0fbb0bdb9cbb60d096c28b3'
    assert binding(installed)['sha256'] == 'dfe64ff6295cf71ffdefae57b772f7382c43926c76b6e7f9491b5795c6e4008d'
    assert binding(refmean)['sha256'] == candidate['actual_scan39_reference']['sha256']
    assert binding(taskA)['sha256'] == binding(native)['sha256'] == 'f0a95a6b7e666c3b58a5e80fd179178bdca1da6e9b8405cbba730c26cb49c576'
    a, b = nib.load(installed), nib.load(reproduced)
    assert a.shape == b.shape and np.array_equal(a.affine, b.affine)
    av, bv = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
    difference = float(np.max(np.abs(av.astype(float) - bv.astype(float))))
    error = np.abs(av.astype(float) - bv.astype(float))
    reproduction_metrics = dict(max_absolute_difference=difference, mean_absolute_difference=float(error.mean()),
                                rms_difference=float(np.sqrt((error * error).mean())), p99_absolute_difference=float(np.percentile(error, 99)),
                                max_difference_over_installed_max=float(difference / np.max(np.abs(av))),
                                correlation=float(np.corrcoef(av.ravel(), bv.ravel())[0, 1]),
                                bitwise_scaled_value_equal=bool(np.array_equal(av, bv)))
    Ma = np.eye(4)
    Ma[:3] = np.loadtxt(matrix, comments='#').reshape(3, 4)
    Ms = np.array(candidate['AFNI_pullback']['values'])
    B = np.array(candidate['B_cached_to_declared_current_RAS_mm'])
    D = np.diag([-1., -1., 1., 1.])
    Bd = D @ B @ D
    total = np.linalg.inv(Bd) @ Ms @ Ma
    points = np.column_stack([np.random.default_rng(66319).normal(size=(256, 3)) * 30, np.ones(256)]).T
    sequential = np.linalg.solve(Bd, Ms @ (Ma @ points))
    assert np.allclose(total @ points, sequential, atol=1e-10, rtol=0)
    assert np.allclose(np.linalg.inv(total) @ sequential, points, atol=1e-10, rtol=0)
    assert not np.allclose(Ma @ Ms @ np.linalg.inv(Bd) @ points, sequential)
    old_audit = json.loads((ROOT / 'notes/region_analysis_review_20261009/cm033_integration_followup_20261009/spm_bridge_input_audit.json').read_text())
    stats = []
    by_name = {Path(row['path']).name: row for row in old_audit['statistics']}
    model_mask = nib.load(by_name['mask.nii']['path'])
    mask_values = np.asanyarray(model_mask.dataobj)
    assert set(np.unique(mask_values)) <= {0, 1}
    valid = mask_values == 1
    for row in old_audit['statistics']:
        assert binding(row['path'])['sha256'] == row['sha256']
        image = nib.load(row['path'])
        values = np.asanyarray(image.dataobj)
        assert image.shape == model_mask.shape and np.array_equal(image.affine, model_mask.affine)
        assert np.array_equal(image.affine, candidate['cached_mean']['affine_mm'])
        assert np.isfinite(values[valid]).all()
        stats.append(dict(binding(row['path']), in_mask_voxels=int(valid.sum()),
                          in_mask_negative=int(np.count_nonzero(values[valid] < 0)),
                          in_mask_positive=int(np.count_nonzero(values[valid] > 0)),
                          in_mask_zero=int(np.count_nonzero(values[valid] == 0)),
                          outside_mask_nonfinite=int(np.count_nonzero(~np.isfinite(values[~valid])))))
    chain_path = EXT / 'functional_normalization_qc/chain_checks.json'
    chain = json.loads(chain_path.read_text())
    assert binding(chain_path)['sha256'] == '179332cf564dd85d36d2a4157c95dbaf6741ebf7c7e8b710bef7338aaef1fc7a'
    transforms = [binding(host(p)) for p in chain['ants_forward_transforms']]
    equality = ROOT / 'notes/region_analysis_review_20261009/mstim_integration_20261009/ancillary_source_readback.json'
    assert binding(equality)['sha256'] == '93ea67b91ce47f0f22b580a0e59d5271e0fe8378b152bf0bfdfdc13f4352f2cd'
    comparison = json.loads(equality.read_text())
    assert comparison['reference_grid_and_scaled_values_identical']
    nmt = ROOT / 'atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz'
    external_nmt = Path('D:/OPTO_fMRI_CM/Templates/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz')
    assert binding(nmt)['sha256'] == '9e37a94c4b9e5865aabb9fd3b51dcc3b2cb16f3daed39c967e68acf16cf92bee'
    assert binding(external_nmt)['sha256'] == 'a45d0041f0cc1d77e8daaf9ca79cf08a17e8d3088741b4411732c6a91ad47056'
    result = dict(status='prospective_chain_geometry_checks_pass_numerical_reproduction_limit_requires_review',
                  checked_utc=datetime.now(timezone.utc).isoformat(), code=binding(__file__), candidate=binding(candidate_path),
                  source_statistics_and_actual_mask=stats,
                  scan39_to_taskA_sources=[binding(p) for p in [refmean, taskA, native, matrix, installed, reproduced]],
                  installed_scan39_warp_exactly_reproduced=False, installed_mean_reapplication_metrics=reproduction_metrics,
                  reproduction_limit='Saved AFNI text matrix has limited decimal precision; original fitted output and reapplication are not bitwise equal. This is a plausible explanation, not independently proven cause.',
                  reproduction_parameters={'program': '3dAllineate', 'source': str(refmean), 'master': str(native),
                                           '1Dmatrix_apply': str(matrix), 'final': 'wsinc5', 'registration_refit': False},
                  M_scan39_to_taskA_AFNI_pullback=Ma.tolist(),
                  prospective_taskA_output_to_cached_input_DICOM_pullback=total.tolist(),
                  formula='inverse(B_DICOM) @ M_samecontrast @ M_scan39_to_taskA', synthetic_256_chain_checks_pass=True,
                  ANTs_ordered_transform_arguments=transforms, external_chain_receipt=binding(chain_path),
                  exact_template_equality_receipt=binding(equality), target_reference=binding(nmt), source_norm_reference=binding(external_nmt),
                  plan={'source_continuous': 'finite signed con/T inside actual SPM mask, zero-filled elsewhere; no absolute value or threshold',
                        'taskA_stage': 'one AFNI composite pullback; Linear con/T; NN actual model mask',
                        'NMT_stage': 'unchanged four-transform ANTs order; Linear con/T; NearestNeighbor actual model mask; explicit default-value 0',
                        'outside_model_support': 'separate warped binary mask; invalid zeros are fill, not evidence of no response',
                        'boundary_limitation': 'Linear interpolation can mix within-model values with zero fill at boundaries; no normalized-convolution correction is silently introduced',
                        'T_interpretation': 'interpolated descriptive statistic only; no new inferential threshold/p-value or GLM'},
                  physical_LR_or_anatomical_acceptance=False, original_estimation_payload_identity_proven=False,
                  statistic_or_mask_propagation_executed=False)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'source_values': stats, 'receipt': binding(output)}, indent=2))


if __name__ == '__main__':
    main()
