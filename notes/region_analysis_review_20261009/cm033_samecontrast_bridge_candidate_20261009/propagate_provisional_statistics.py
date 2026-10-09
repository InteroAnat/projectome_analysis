"""Authorized one saved-transform propagation; no registration fit or new GLM."""
from datetime import datetime, timezone
from decimal import Decimal
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

from run_declared_coordinate_bridge import bind, sha, wsl, DEB, HERE


def main():
    out = HERE / 'statistic_chain'
    if out.exists():
        raise FileExistsError('Preserve prior propagation; never silently overwrite')
    proposal_path = HERE / 'prospective_statistic_chain.json'
    assert sha(proposal_path) == '5d2e19779dada3351abe94772e35eea3338f597be1f8b8faed7d9d5165290dfe'
    proposal = json.loads(proposal_path.read_text())
    candidate = json.loads((HERE / 'bridge_candidate_provenance.json').read_text())
    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import loadpkl
    all_sources = [*proposal['source_statistics_and_actual_mask'], *proposal['scan39_to_taskA_sources'],
                   *proposal['ANTs_ordered_transform_arguments'], proposal['target_reference'],
                   proposal['source_norm_reference'], proposal['external_chain_receipt'], proposal['exact_template_equality_receipt']]
    for row in all_sources:
        assert sha(wsl(row['path'])) == row['sha256'], row['path']
    assert candidate['residual_rotation_gate_pass']
    ext = DEB / 'alignment_review/cm033_Dsource_20261009'
    pre = ext / 'bids/sub-cm033/ses-20160120/derivatives/preproc/anat'
    node = ext / 'bids/derivatives/work/sub-cm033/ses-20160120/nipype_epi_anat/epi_anat_wf/allineate_refmean_to_brain'
    node_inputs = loadpkl(node / '_inputs.pklz')
    assert Path(node_inputs['in_file']) == pre / 'sub-cm033_ses-20160120_desc-refmean_bold.nii'
    assert Path(node_inputs['reference']) == Path(node_inputs['master']) == pre / 'sub-cm033_ses-20160120_desc-brain.nii'
    assert node_inputs['final_interpolation'] == 'wsinc5' and node_inputs['warp_type'] == 'shift_rotate'
    original_command = (node / 'command.txt').read_text()
    assert '-final wsinc5' in original_command and str(node_inputs['master']) in original_command
    assert Path(node_inputs['in_file']).name in original_command
    ma_path = pre / 'sub-cm033_ses-20160120_desc-epiToAnatAffine.aff12.1D'
    tokens = [token for line in ma_path.read_text().splitlines() if not line.startswith('#') for token in line.split()]
    assert len(tokens) == 12
    rounding_half_units = np.array([float(Decimal(10) ** Decimal(Decimal(t).as_tuple().exponent) / 2) for t in tokens]).reshape(3, 4)
    corner_bounds = {}
    for label, path in [('source_scan39', node_inputs['in_file']), ('master_taskA', node_inputs['master'])]:
        image = nib.load(path)
        indices = np.array([list(c) + [1] for c in itertools.product(*[(0, n - 1) for n in image.shape[:3]])]).T
        D = np.diag([-1., -1., 1., 1.])
        points = D @ image.affine @ indices
        bounds = rounding_half_units @ np.abs(points)
        corner_bounds[label] = dict(component_max_bound_mm=np.max(bounds, axis=1).tolist(),
                                    max_Euclidean_bound_mm=float(np.max(np.linalg.norm(bounds, axis=0))))
    out.mkdir()
    (out / 'working_copy').mkdir()
    (out / 'native').mkdir()
    (out / 'nmt').mkdir()
    matrix = out / 'desc-taskAOutputToCachedInputComposite.aff12.1D'
    total = np.array(proposal['prospective_taskA_output_to_cached_input_DICOM_pullback'])
    np.savetxt(matrix, total[:3].reshape(1, 12), fmt='%.17g')
    assert np.array_equal(np.loadtxt(matrix).reshape(3, 4), total[:3])
    os.environ.update(PATH='/home/binbin/abin:/home/binbin/ants-2.5.1/bin:' + os.environ['PATH'],
                      OMP_NUM_THREADS='4', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4', AFNI_DONT_LOGFILE='YES')
    commands = []
    def run(args):
        args = [str(v) for v in args]
        print('COMMAND', json.dumps(args), flush=True)
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        with (out / 'commands.log').open('a') as stream:
            stream.write(json.dumps(args) + '\n' + result.stdout + result.stderr + '\n')
        commands.append(args)
        return result.stdout + result.stderr
    reference = wsl(proposal['target_reference']['path'])
    native_master = Path(node_inputs['master'])
    source_rows = {wsl(row['path']).name: row for row in proposal['source_statistics_and_actual_mask']}
    model_dir = wsl(source_rows['mask.nii']['path']).parent.parent
    contrasts = model_dir / 'contrasts.json'
    spm_contrast = next(row for row in json.loads(contrasts.read_text()) if row['idx'] == 1)
    assert spm_contrast['name'] == 'stimulus_gt_baseline' and spm_contrast['stat'] == 'T'
    all_sources += [bind(contrasts), bind(model_dir / 'SPM.mat'), bind(model_dir / 'contrast_qc.json')]
    mask_image = nib.load(wsl(source_rows['mask.nii']['path']))
    valid = np.asanyarray(mask_image.dataobj) == 1
    outputs = {}
    working = {}
    status_path = out / 'chain_provenance.json'
    record = dict(status='running_provisional_saved_transform_statistic_propagation', subject='cm033', spm_contrast=spm_contrast, code=bind(__file__),
                  started_utc=datetime.now(timezone.utc).isoformat(), PID=os.getpid(), python=sys.executable,
                  proposal=bind(proposal_path), candidate=bind(HERE / 'bridge_candidate_provenance.json'),
                  inputs={'reference': bind(reference), 'source_statistics': proposal['source_statistics_and_actual_mask']},
                  source_bindings=[dict(row, path=str(wsl(row['path']))) for row in all_sources],
                  source_sha256={str(wsl(row['path'])): row['sha256'] for row in all_sources},
                  transformations={'B_coordinate_recoding_RAS': candidate['B_cached_to_declared_current_RAS_mm'],
                                   'AFNI_composite_pullback': dict(bind(matrix), values=total.tolist()),
                                   'ANTs_ordered_transform_arguments': [dict(row, path=str(wsl(row['path']))) for row in proposal['ANTs_ordered_transform_arguments']]},
                  original_installed_fit_command=original_command, original_installed_fit_inputs=bind(node / '_inputs.pklz'),
                  original_installed_fit_command_binding=bind(node / 'command.txt'),
                  installed_mean_reapplication_metrics=proposal['installed_mean_reapplication_metrics'],
                  conditional_text_rounding_corner_error_bounds=corner_bounds,
                  rounding_bound_assumption='Each printed coefficient is rounded to its last displayed decimal place; a conservative implied uncertainty, not a measured true coordinate error.',
                  interpolation={'AFNI_continuous': 'Linear', 'AFNI_mask': 'NN', 'ANTs_continuous': 'Linear', 'ANTs_mask': 'NearestNeighbor', 'ANTs_default_value': 0},
                  constraints={'provisional_correspondence_only': True, 'physical_LR_accepted': False, 'historical_estimation_payload_proven': False,
                               'new_GLM': False, 'new_smoothing': False, 'renormalization': False, 'inferential_interpolated_T_claim': False,
                               'zero_outside_validity_is_fill_not_no_response': True,
                               'continuous_values_outside_warped_NN_mask_may_reflect_boundary_interpolation_and_are_invalid': True,
                               'residual_gate_does_not_validate_B_as_physical_rotation': True}, outputs=outputs)
    def save():
        status_path.write_text(json.dumps(record, indent=2, allow_nan=False))
    save()
    try:
        record['software'] = {'AFNI': run(['afni', '-ver']).strip(), 'ANTs': run(['antsRegistration', '--version']).strip(),
                              'antsApplyTransforms_executable': bind('/home/binbin/ants-2.5.1/bin/antsApplyTransforms')}
        for name, source_name, suffix, is_mask in [('contrast', 'con_0001.nii', 'con', False), ('T', 'spmT_0001.nii', 'spmT', False), ('analysis_coverage', 'mask.nii', 'modelmask', True)]:
            source = wsl(source_rows[source_name]['path'])
            image = nib.load(source)
            values = np.asanyarray(image.dataobj)
            assert np.array_equal(image.affine, mask_image.affine)
            assert np.isfinite(values[valid]).all()
            payload = valid.astype(np.uint8) if is_mask else np.where(valid, values, 0).astype(np.float64)
            work_path = out / 'working_copy' / f'sub-cm033_space-cachedSPM_desc-finiteWithinActualModelMask_{suffix}.nii.gz'
            work = nib.Nifti1Image(payload, image.affine, image.header.copy())
            work.set_data_dtype(np.uint8 if is_mask else np.float64)
            work.set_qform(image.affine, 2)
            work.set_sform(image.affine, 2)
            nib.save(work, work_path)
            saved_values = np.asanyarray(nib.load(work_path).dataobj)
            assert np.array_equal(saved_values[valid], values[valid])
            assert np.count_nonzero(saved_values[~valid]) == 0
            assert np.isfinite(saved_values).all()
            working[name] = dict(bind(work_path), source=bind(source), exact_scaled_values_inside_actual_model_mask=True,
                                 outside_fill=0, source_mask_voxels=int(valid.sum()))
            native_path = out / 'native' / f'sub-cm033_ses-20160120_space-taskA_desc-provisionalBridge_{suffix}.nii.gz'
            nmt_path = out / 'nmt' / f'sub-cm033_ses-20160120_space-NMT_desc-provisionalBridge_{suffix}.nii.gz'
            run(['3dAllineate', '-source', work_path, '-master', native_master, '-1Dmatrix_apply', matrix,
                 '-final', 'NN' if is_mask else 'linear', '-prefix', native_path])
            cmd = ['antsApplyTransforms', '-d', '3', '-i', native_path, '-r', reference, '-o', nmt_path,
                   '-n', 'NearestNeighbor' if is_mask else 'Linear', '--default-value', '0']
            for transform in proposal['ANTs_ordered_transform_arguments']:
                cmd += ['-t', wsl(transform['path'])]
            run(cmd)
            for path, master in [(native_path, native_master), (nmt_path, reference)]:
                produced, target = nib.load(path), nib.load(master)
                assert produced.shape == target.shape and np.array_equal(produced.affine, target.affine)
                checked = np.asanyarray(produced.dataobj)
                assert np.isfinite(checked).all()
                if is_mask:
                    assert set(np.unique(checked)) <= {0, 1}
            outputs[name] = {'path': str(nmt_path), 'sha256': sha(nmt_path), 'native_path': str(native_path),
                             'native_sha256': sha(native_path), 'finite_exact_grid': True}
            save()
        for row in all_sources:
            assert sha(wsl(row['path'])) == row['sha256']
        record.update(status='software_completed_provisional_cm033_statistic_chain', independent_saved_transform_reproduction='pending',
                      completed_utc=datetime.now(timezone.utc).isoformat(), working_copies=working, commands=commands)
        save()
        print(json.dumps({'status': record['status'], 'receipt': bind(status_path), 'outputs': outputs}, indent=2), flush=True)
    except Exception as error:
        record.update(status='failed_provisional_statistic_chain', error=str(error), working_copies=working, commands=commands)
        save()
        raise


if __name__ == '__main__':
    main()
