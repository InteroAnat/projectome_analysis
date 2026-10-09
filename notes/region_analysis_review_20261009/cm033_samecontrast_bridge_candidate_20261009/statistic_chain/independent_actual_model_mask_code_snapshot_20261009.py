"""Independent saved-transform CM033 statistic reproduction; never fit or infer."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def local(path):
    value = str(path).replace('\\', '/')
    if os.name == 'nt' and value.startswith('/mnt/'):
        value = value[5].upper() + ':' + value[6:]
    elif os.name != 'nt' and len(value) >= 3 and value[1:3] == ':/':
        value = '/mnt/' + value[0].lower() + value[2:]
    return Path(value)


def digest(path):
    value = hashlib.sha256()
    with local(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def affine_file(path):
    result = np.eye(4)
    result[:3] = np.loadtxt(local(path), comments='#').reshape(3, 4)
    return result


def exact_grid(image, master):
    require(image.shape == master.shape and np.array_equal(image.affine, master.affine),
            'Image/master exact grid mismatch')
    require(image.header.get_xyzt_units()[0] == 'mm', 'Image units must be millimetres')


def command(args, output_folder, recorded):
    args = [str(item) for item in args]
    process = subprocess.run(args, capture_output=True, text=True, check=False, cwd=output_folder)
    recorded.append({'argv': args, 'returncode': process.returncode})
    with (output_folder / 'commands.log').open('a') as stream:
        stream.write(json.dumps(args) + '\n' + process.stdout + process.stderr + '\n')
    require(process.returncode == 0, 'Independent saved-transform command failed; see commands.log')
    return process.stdout + process.stderr


def check_actual_model_mask():
    """Use installed Windows h5py; WSL transform runtime need not install it."""
    import h5py

    path = HERE / 'statistic_chain/chain_provenance.json'
    chain = json.loads(path.read_text())
    require(chain['status'] == 'software_completed_provisional_cm033_statistic_chain', 'Chain not frozen')
    proposal = json.loads(local(chain['proposal']['path']).read_text())
    source = next(row for row in proposal['source_statistics_and_actual_mask'] if local(row['path']).name == 'mask.nii')
    require(digest(source['path']) == source['sha256'], 'Original actual model mask changed')
    image = nib.load(local(source['path']))
    valid = np.asanyarray(image.dataobj) == 1
    audit_path = ROOT / 'notes/region_analysis_review_20261009/cm033_integration_followup_20261009/spm_bridge_input_audit.json'
    audit = json.loads(audit_path.read_text())
    require(digest(audit['SPM']['path']) == audit['SPM']['sha256'], 'Actual SPM changed')
    with h5py.File(local(audit['SPM']['path']), 'r') as model:
        xyz = np.asarray(model['SPM/xVol/XYZ']).T
        require(xyz.shape[0] == 3 and xyz.shape[1] == valid.sum(), 'SPM model-mask dimensions differ')
        require(np.equal(xyz, np.floor(xyz)).all(), 'SPM model pointset indices not integers')
        indices = xyz.astype(int) - 1
        reconstructed = np.zeros(valid.shape, dtype=bool)
        require((indices >= 0).all() and (indices < np.asarray(valid.shape)[:, None]).all(), 'SPM point outside grid')
        reconstructed[tuple(indices)] = True
        require(np.array_equal(reconstructed, valid), 'Actual mask differs from SPM xVol pointset')
        shift = np.eye(4)
        shift[:3, 3] = 1
        require(np.array_equal(np.asarray(model['SPM/xVol/M']).T @ shift, image.affine), 'SPM one-based spatial matrix differs from cached zero-based mask grid')
    result = {'status': 'independent_actual_SPM_model_mask_pointset_readback_passed',
              'chain_provenance_sha256': digest(path), 'SPM': audit['SPM'], 'mask': source,
              'model_mask_voxels': int(valid.sum()), 'SPM_xVol_XYZ_exactly_matches_mask': True,
              'SPM_one_based_M_matches_zero_based_mask_affine': True, 'code_sha256': digest(__file__)}
    destination = HERE / 'statistic_chain/independent_actual_model_mask_readback.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


def main():
    chain_dir = HERE / 'statistic_chain'
    chain_path = chain_dir / 'chain_provenance.json'
    chain = json.loads(chain_path.read_text())
    require(chain['status'] == 'software_completed_provisional_cm033_statistic_chain',
            'Only a completed frozen chain may be independently reproduced')
    require(chain.get('subject') == 'cm033', 'Require exact CM033 subject')
    chain_hash = digest(chain_path)
    bindings = dict(chain['source_sha256'])
    for path, expected in bindings.items():
        require(digest(path) == expected, 'Changed frozen source: ' + path)
    require(digest(chain['code']['path']) == chain['code']['sha256'], 'Frozen producer changed')
    proposal_path = local(chain['proposal']['path'])
    proposal = json.loads(proposal_path.read_text())
    require(digest(proposal_path) == chain['proposal']['sha256'], 'Prospective contract changed')
    candidate_path = local(chain['candidate']['path'])
    candidate = json.loads(candidate_path.read_text())
    require(digest(candidate_path) == chain['candidate']['sha256'], 'Bridge receipt changed')
    output_hashes = {name: row['sha256'] for name, row in chain['outputs'].items()}
    require(set(output_hashes) == {'contrast', 'T', 'analysis_coverage'}, 'Three required outputs absent')
    for name, row in chain['outputs'].items():
        require(digest(row['path']) == row['sha256'], 'Original final output changed: ' + name)
        require(digest(row['native_path']) == row['native_sha256'], 'Original taskA output changed: ' + name)

    original_sources = {local(row['path']).name: row for row in proposal['source_statistics_and_actual_mask']}
    mask_image = nib.load(local(original_sources['mask.nii']['path']))
    mask = np.asanyarray(mask_image.dataobj)
    require(set(np.unique(mask)) <= {0, 1}, 'Actual model mask must be binary')
    valid = mask == 1
    require(int(valid.sum()) == 68267, 'Actual cached model mask denominator changed')
    cached = nib.load(local(candidate['cached_mean']['path']))
    exact_grid(mask_image, cached)
    audit_path = ROOT / 'notes/region_analysis_review_20261009/cm033_integration_followup_20261009/spm_bridge_input_audit.json'
    audit = json.loads(audit_path.read_text())
    require(digest(audit['SPM']['path']) == audit['SPM']['sha256'], 'Original cached SPM model changed')
    model_review_path = chain_dir / 'independent_actual_model_mask_readback.json'
    model_review = json.loads(model_review_path.read_text())
    require(model_review['status'] == 'independent_actual_SPM_model_mask_pointset_readback_passed', 'Actual-model independent review absent')
    require(model_review['chain_provenance_sha256'] == chain_hash, 'Model-mask review binds a different chain')
    require(model_review['SPM']['sha256'] == audit['SPM']['sha256'] and model_review['mask']['sha256'] == original_sources['mask.nii']['sha256'], 'Model-mask review source bindings differ')

    D = np.diag([-1., -1., 1., 1.])
    cached_to_current = np.array(candidate['B_cached_to_declared_current_RAS_mm'])
    current_image = nib.load(local(candidate['declared_current_mean']['path']))
    require(np.array_equal(cached_to_current @ cached.affine, current_image.affine), 'B affine recoding identity differs')
    B_dicom = D @ cached_to_current @ D
    samecontrast_matrix = affine_file(candidate['AFNI_pullback']['path'])
    installed_row = next(row for row in proposal['scan39_to_taskA_sources']
                         if local(row['path']).name.endswith('desc-epiToAnatAffine.aff12.1D'))
    installed_matrix = affine_file(installed_row['path'])
    composite = np.linalg.solve(B_dicom, samecontrast_matrix @ installed_matrix)
    saved_composite = affine_file(chain['transformations']['AFNI_composite_pullback']['path'])
    require(np.allclose(composite, saved_composite, atol=1e-12, rtol=0), 'Saved composite differs from independent composition')
    points = np.vstack([np.random.default_rng(908163).normal(size=(3, 512)) * 40, np.ones(512)])
    # taskA output DICOM -> scan39 DICOM -> declared-current DICOM -> cached input DICOM.
    sequential = np.linalg.solve(B_dicom, samecontrast_matrix @ (installed_matrix @ points))
    require(np.allclose(saved_composite @ points, sequential, atol=1e-10, rtol=0), 'Sequential named-space composition mismatch')
    require(np.allclose(np.linalg.solve(saved_composite, sequential), points, atol=1e-10, rtol=0), 'Independent inverse roundtrip mismatch')
    require(not np.allclose(installed_matrix @ samecontrast_matrix @ np.linalg.inv(B_dicom) @ points, sequential), 'Wrong-order negative control failed')
    require(not np.allclose(np.linalg.solve(cached_to_current, samecontrast_matrix @ installed_matrix @ points), sequential), 'Missing RAS/DICOM negative control failed')

    reference_path = local(proposal['target_reference']['path'])
    old_reference_path = local(proposal['source_norm_reference']['path'])
    reference, old_reference = nib.load(reference_path), nib.load(old_reference_path)
    exact_grid(old_reference, reference)
    require(np.array_equal(np.asanyarray(reference.dataobj), np.asanyarray(old_reference.dataobj)), 'NMT2.0/2.1 reference scaled values differ')
    require(digest(reference_path) == proposal['target_reference']['sha256'], 'Pinned projectome NMT reference changed')
    require(digest(old_reference_path) == proposal['source_norm_reference']['sha256'], 'Original normalization NMT reference changed')
    transforms = proposal['ANTs_ordered_transform_arguments']
    require(len(transforms) == 4, 'Four ordered ANTs transforms required')
    require([row['sha256'] for row in transforms] == [row['sha256'] for row in chain['transformations']['ANTs_ordered_transform_arguments']], 'ANTs order/hash mismatch')
    for row in transforms:
        require(digest(row['path']) == row['sha256'], 'ANTs transform changed')
    native_master = local(next(row['path'] for row in proposal['scan39_to_taskA_sources']
                               if local(row['path']).name.endswith('desc-brain.nii')))
    native_reference = nib.load(native_master)
    require(digest(native_master) == next(row['sha256'] for row in proposal['scan39_to_taskA_sources']
                                         if local(row['path']).name.endswith('desc-brain.nii')), 'TaskA master changed')
    output = chain_dir / 'independent_reapplication_20261009'
    require(not output.exists(), 'Preserve any earlier independent verification')
    output.mkdir()
    (output / 'tmp').mkdir()
    os.environ.update(OMP_NUM_THREADS='4', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4',
                      AFNI_DONT_LOGFILE='YES', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(output / 'tmp'))
    independent_matrix = output / 'desc-independentTaskAOutputToCachedInput.aff12.1D'
    # Reproduce the exact saved coefficients after independently checking their
    # composition. Re-solving must not introduce a different rounding variant.
    np.savetxt(independent_matrix, saved_composite[:3].reshape(1, 12), fmt='%.17g')
    require(np.array_equal(affine_file(independent_matrix), saved_composite), 'Independent saved precision differs')
    commands = []
    afni_binary = Path('/home/binbin/abin/3dAllineate')
    ants_binary = Path('/home/binbin/ants-2.5.1/bin/antsApplyTransforms')
    command(['/home/binbin/abin/afni', '-ver'], output, commands)
    command([ants_binary, '--version'], output, commands)
    comparisons = {}
    signed_sources = {}
    for name, source_name, is_mask in [('contrast', 'con_0001.nii', False), ('T', 'spmT_0001.nii', False),
                                      ('analysis_coverage', 'mask.nii', True)]:
        source = local(original_sources[source_name]['path'])
        image = nib.load(source)
        exact_grid(image, mask_image)
        values = np.asanyarray(image.dataobj)
        require(np.isfinite(values[valid]).all(), 'Within-mask source values nonfinite')
        signed_sources[name] = {'within_mask': int(valid.sum()), 'negative': int(np.sum(values[valid] < 0)),
                                'positive': int(np.sum(values[valid] > 0)), 'zero': int(np.sum(values[valid] == 0)),
                                'outside_nonfinite': int(np.sum(~np.isfinite(values[~valid])))}
        payload = valid.astype(np.uint8) if is_mask else np.where(valid, values, 0).astype(np.float64)
        frozen_work = nib.load(local(chain['working_copies'][name]['path']))
        exact_grid(frozen_work, image)
        require(np.array_equal(payload, np.asanyarray(frozen_work.dataobj)), 'Frozen finite work copy differs from independently derived actual-mask values')
        require(np.isfinite(payload).all(), 'Independent payload nonfinite')
        work = output / (name + '_source_finite_actual_mask.nii.gz')
        work_image = nib.Nifti1Image(payload, image.affine, image.header.copy())
        work_image.set_data_dtype(np.uint8 if is_mask else np.float64)
        work_image.set_qform(image.affine, 2)
        work_image.set_sform(image.affine, 2)
        nib.save(work_image, work)
        require(np.array_equal(np.asanyarray(nib.load(work).dataobj), payload), 'Independent saved payload altered')
        native = output / (name + '_space-taskA_reapplied.nii.gz')
        final = output / (name + '_space-NMT_reapplied.nii.gz')
        command([afni_binary, '-source', work, '-master', native_master, '-1Dmatrix_apply', independent_matrix,
                 '-final', 'NN' if is_mask else 'linear', '-prefix', native], output, commands)
        args = [ants_binary, '-d', '3', '-i', native, '-r', reference_path, '-o', final,
                '-n', 'NearestNeighbor' if is_mask else 'Linear', '--default-value', '0']
        for transform in transforms:
            args += ['-t', local(transform['path'])]
        command(args, output, commands)
        comparisons[name] = {}
        for stage, actual_path, original_path, master in (
            ('taskA', native, local(chain['outputs'][name]['native_path']), native_reference),
            ('NMT', final, local(chain['outputs'][name]['path']), reference),
        ):
            actual, original = nib.load(actual_path), nib.load(original_path)
            exact_grid(actual, master)
            exact_grid(original, master)
            actual_values = np.asanyarray(actual.dataobj)
            original_values = np.asanyarray(original.dataobj)
            require(np.isfinite(actual_values).all(), 'Reapplied values nonfinite')
            require(np.array_equal(actual_values, original_values), 'Every-voxel saved-transform reproduction failed: ' + name + '/' + stage)
            if is_mask:
                require(set(np.unique(actual_values)) <= {0, 1}, 'Nearest-neighbour coverage not binary')
            comparisons[name][stage] = {'all_scaled_voxels_exactly_equal': True,
                'max_absolute_difference': 0.0, 'shape': list(actual.shape), 'voxels_checked': int(actual_values.size),
                'independent_path': str(actual_path), 'independent_sha256': digest(actual_path),
                'original_path': str(original_path), 'original_sha256': digest(original_path)}
        print('independent every-voxel reproduction passed: ' + name, flush=True)
    final_mask = np.asanyarray(nib.load(output / 'analysis_coverage_space-NMT_reapplied.nii.gz').dataobj) == 1
    coverage = {'model_covered_reference_voxels': int(final_mask.sum()), 'actual_model_mask_only': True,
                'binary_coverage_not_response_threshold': True}
    for name in ['contrast', 'T']:
        values = np.asanyarray(nib.load(output / (name + '_space-NMT_reapplied.nii.gz')).dataobj)
        coverage[name] = {'covered_negative': int(np.sum(values[final_mask] < 0)),
                          'covered_positive': int(np.sum(values[final_mask] > 0)),
                          'covered_zero': int(np.sum(values[final_mask] == 0)),
                          'outside_coverage_nonzero': int(np.count_nonzero(values[~final_mask]))}
    for path, expected in bindings.items():
        require(digest(path) == expected, 'Source changed during verification: ' + path)
    require(digest(chain_path) == chain_hash, 'Frozen chain receipt changed during verification')
    for name, row in chain['outputs'].items():
        require(digest(row['path']) == output_hashes[name], 'Frozen output changed during verification')
    result = {'status': 'independent_saved_cm033_statistic_chain_reapplication_passed',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'subject': 'cm033',
              'chain_provenance': {'path': str(chain_path), 'sha256': chain_hash},
              'source_sha256': bindings, 'output_sha256': output_hashes,
              'independent_code': {'path': str(Path(__file__)), 'sha256': digest(__file__)},
              'producer_functions_imported': False, 'registration_refit': False, 'new_GLM': False,
              'source_model_mask_matches_actual_SPM_xVol_pointset': True,
              'actual_model_mask_independent_review': {'path': str(model_review_path), 'sha256': digest(model_review_path)},
              'actual_SPM_model': audit['SPM'], 'source_signed_value_checks': signed_sources,
              'sequential_named_space_composition_512_points': True, 'inverse_and_wrong_order_negative_controls_pass': True,
              'independent_AFNI_pullback': composite.tolist(), 'every_voxel_comparisons': comparisons,
              'template_compatibility': {'exact_shape_affine_scaled_values_equal': True,
                    'projectome_reference': proposal['target_reference'], 'original_norm_reference': proposal['source_norm_reference'],
                    'scope': 'Only these pinned skull-stripped reference images; filenames/byte hashes and atlas/version lineage remain distinct.'},
              'coverage': coverage, 'saved_installed_mean_reproduction_limit_preserved': proposal['installed_mean_reapplication_metrics'],
              'installed_mean_reproduction_cause_unproven': True,
              'software': {'AFNI_binary_sha256': digest(afni_binary), 'ANTs_binary_sha256': digest(ants_binary)},
              'commands': commands, 'anatomical_acceptance': False, 'physical_LR_acceptance': False,
              'original_estimation_payload_identity_proven': False,
              'limitations': ['Independent reproduction validates saved-transform arithmetic, not anatomy or historical estimation-payload correspondence.',
                    'Interpolated T is descriptive; no new p-value or inferential threshold.',
                    'Zero fill outside actual model support is invalid response evidence; linear boundary mixtures outside NN coverage remain invalid.']}
    receipt = chain_dir / 'independent_statistic_chain_readback.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': str(receipt), 'receipt_sha256': digest(receipt), 'coverage': coverage}, indent=2))


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--check-model-only']:
        check_actual_model_mask()
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('Use no arguments for reproduction, or --check-model-only')
