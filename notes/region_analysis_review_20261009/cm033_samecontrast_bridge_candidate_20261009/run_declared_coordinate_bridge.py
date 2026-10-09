"""One B-recoded rigid EPI bridge; no statistical propagation or external edits."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEB = Path('/mnt/c/Users/laika_yan/Documents/ChatGPT/deb_fmri')
PRIOR = ROOT / 'notes/region_analysis_review_20261009/cm033_bridge_candidate_20261009'
EVIDENCE = PRIOR / 'coordinate_initialization_evidence.json'
STATUS = HERE / 'bridge_candidate_provenance.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wsl(path):
    path = str(path).replace('\\', '/')
    return Path('/mnt/' + path[0].lower() + path[2:]) if path[1:3] == ':/' else Path(path)


def bind(path):
    return {'path': str(path), 'sha256': sha(path)}


def main():
    if STATUS.exists() or (HERE / 'fit').exists():
        raise FileExistsError('Existing candidate; never start duplicate fit')
    assert sha(EVIDENCE) == '541e168db61d849154a69baf390bf9dd35189d70437a8366e7f3561e7918cf3f'
    evidence = json.loads(EVIDENCE.read_text())
    sources = []
    for row in [evidence['prepared_mean'], evidence['actual_new_reference'], *evidence['current_inputs'], *evidence['external_source_bindings']]:
        path = wsl(row['path'])
        expected = row['sha256']
        if path == DEB / 'alignment_review/cm033_Dsource_20261009/functional_normalization_qc/chain_checks.json':
            current_chain = json.loads(path.read_text())
            assert len(current_chain['runs']) == 19
            assert all(r['nmt_grid_matches'] and r['all_values_finite'] for r in current_chain['runs'])
            assert current_chain['scope'] == '3D full-mean chain diagnostics; no normalized 4D or GLM outputs'
            expected = '179332cf564dd85d36d2a4157c95dbaf6741ebf7c7e8b710bef7338aaef1fc7a'
        assert sha(path) == expected, path
        sources.append(dict(bind(path), prior_snapshot_sha256=row['sha256']))
    for name in ['fit', 'working_copy', 'coarse_qc', 'nipype_work', 'nipype_config', 'mpl_config', 'logs', 'tmp']:
        (HERE / name).mkdir()
    os.chdir(HERE)
    os.environ.update(PATH='/home/binbin/abin:' + os.environ['PATH'], OMP_NUM_THREADS='4',
                      NIPYPE_CONFIG_DIR=str(HERE / 'nipype_config'), MPLCONFIGDIR=str(HERE / 'mpl_config'),
                      MPLBACKEND='Agg', TMPDIR=str(HERE / 'tmp'), AFNI_DONT_LOGFILE='YES', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(DEB))
    import nibabel as nib
    import numpy as np
    import nipype
    from nipype import config, logging
    from nipype.utils.filemanip import loadpkl
    from fmri_pipeline.workflows.nipype_anat_epi import init_epi_anat_wf
    from fmri_pipeline.preproc.run_anat_epi_bids import validate_registration_rotation
    loaded = Path(inspect.getsourcefile(init_epi_anat_wf)).resolve()
    assert loaded == DEB / 'fmri_pipeline/workflows/nipype_anat_epi.py'
    assert sha(loaded) == 'b50e42118ad6c461eb3c4be0e3c5c40c531e9f83914b8ede3d49bc11b4982bc1'
    config.set('logging', 'log_directory', str(HERE / 'logs'))
    logging.update_logging(config)
    original = wsl(evidence['prepared_mean']['path'])
    fixed = wsl(evidence['actual_new_reference']['path'])
    image = nib.load(original)
    data = np.asanyarray(image.dataobj)
    B = np.array(evidence['B_cached_world_to_current_world_mm'])
    current_affine = np.array(evidence['current_inputs'][0]['affine_mm'])
    assert np.array_equal(B @ image.affine, current_affine)
    moving = HERE / 'working_copy/sub-cm033_desc-currentPayload1472Mean_space-declaredCurrentWorld_bold.nii.gz'
    work = nib.Nifti1Image(data, current_affine, image.header.copy())
    work.set_qform(current_affine, 2)
    work.set_sform(current_affine, 2)
    nib.save(work, moving)
    saved = nib.load(moving)
    assert np.array_equal(data, np.asanyarray(saved.dataobj))
    assert np.array_equal(saved.affine, current_affine)
    voxels = np.column_stack([np.random.default_rng(20261009).uniform(-10, 140, (64, 3)), np.ones(64)])
    assert np.allclose(B @ image.affine @ voxels.T, current_affine @ voxels.T, atol=1e-12, rtol=0)

    def info(path, master=None):
        img = nib.load(path)
        assert img.header.get_xyzt_units()[0] == 'mm'
        assert np.isfinite(np.asanyarray(img.dataobj)).all()
        if master is not None:
            target = nib.load(master)
            assert img.shape == target.shape and np.array_equal(img.affine, target.affine)
        return dict(bind(path), shape=list(img.shape), affine_mm=img.affine.tolist(), header_axcodes=list(nib.aff2axcodes(img.affine)), finite=True)

    def command(args):
        result = subprocess.run([str(v) for v in args], capture_output=True, text=True, check=True)
        with (HERE / 'afni_commands.log').open('a') as stream:
            stream.write(json.dumps([str(v) for v in args]) + '\n' + result.stdout + result.stderr + '\n')
        return result.stdout

    record = dict(status='running_declared_coordinate_samecontrast_candidate', started_utc=datetime.now(timezone.utc).isoformat(),
                  PID=os.getpid(), python=sys.executable, nipype_version=nipype.__version__, code=bind(__file__),
                  verified_sources=sources, input_evidence=bind(EVIDENCE), workflow_source=bind(loaded),
                  cached_mean=info(original), declared_current_mean=info(moving), actual_scan39_reference=info(fixed),
                  B_cached_to_declared_current_RAS_mm=B.tolist(), scaled_array_exact_equality=True,
                  no_array_flip_or_interpolation_before_fit=True, synthetic_coordinate_samples=64,
                  parameters=dict(warp_type='shift_rotate', cost='lpa', center_of_mass=True, final_interpolation='wsinc5', residual_rotation_gate_degrees=30),
                  B_is_coordinate_recoding_not_accepted_physical_rotation=True,
                  external_mutations=False, statistics_or_model_mask_warped=False, physical_laterality_accepted=False,
                  original_estimation_payload_identity_proven=False, third_fit_or_tuning=False)
    def status():
        STATUS.write_text(json.dumps(record, indent=2, allow_nan=False))
    status()
    try:
        record['AFNI_version'] = command(['afni', '-ver']).strip()
        record['AFNI_executable'] = bind('/home/binbin/abin/3dAllineate')
        workflow = init_epi_anat_wf(brain=fixed, t1w=None, refmean=moving, out_dir=HERE / 'fit', warp_type='shift_rotate',
                                   name='cm033_declaredCurrent_to_scan39_candidate', base_dir=HERE / 'nipype_work')
        record['fit_command'] = workflow.get_node('allineate_refmean_to_brain').interface.cmdline
        status()
        print('FIT', record['fit_command'], flush=True)
        workflow.run(plugin='Linear')
        runtime = loadpkl(HERE / 'nipype_work/cm033_declaredCurrent_to_scan39_candidate/allineate_refmean_to_brain/result_allineate_refmean_to_brain.pklz').runtime
        (HERE / 'fit_execution_log.txt').write_text('COMMAND\n' + runtime.cmdline + '\nSTDOUT\n' + runtime.stdout + '\nSTDERR\n' + runtime.stderr)
        aligned = HERE / 'fit/desc-refmeanInAnat.nii'
        matrix = HERE / 'fit/desc-epiToAnatAffine.aff12.1D'
        inverse = HERE / 'fit/desc-scan39ToDeclaredCurrentApply.aff12.1D'
        inverse.write_text(command(['cat_matvec', matrix, '-I']))
        M, Mi = np.eye(4), np.eye(4)
        M[:3] = np.loadtxt(matrix, comments='#').reshape(3, 4)
        Mi[:3] = np.loadtxt(inverse, comments='#').reshape(3, 4)
        assert np.allclose(M @ Mi, np.eye(4), atol=1e-4, rtol=0)
        u, singular, vh = np.linalg.svd(M[:3, :3])
        rotation = u @ vh
        assert np.linalg.det(rotation) > 0
        degrees = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
        try:
            public_degrees, public_gate = validate_registration_rotation(matrix), True
        except RuntimeError as error:
            public_degrees, public_gate = None, False
            record['public_gate_error'] = str(error)
        assert public_gate == (degrees <= 30)
        D = np.diag([-1., -1., 1., 1.])
        pullback_RAS = np.linalg.inv(B) @ D @ M @ D
        forward_RAS = D @ np.linalg.inv(M) @ D @ B
        new_world = nib.load(fixed).affine @ voxels.T
        cached_world = pullback_RAS @ new_world
        assert np.allclose(B @ cached_world, D @ M @ D @ new_world, atol=1e-10, rtol=0)
        assert np.allclose(forward_RAS @ cached_world, new_world, atol=1e-10, rtol=0)
        composition = dict(B_cached_to_declared_RAS=B.tolist(), fitted_AFNI_pullback_newRef_DICOM_to_declaredCurrent_DICOM=M.tolist(),
                           prospective_pullback_newRef_RAS_to_cached_RAS=pullback_RAS.tolist(),
                           prospective_pointmap_cached_RAS_to_newRef_RAS=forward_RAS.tolist(), synthetic_samples=64,
                           inverse_and_named_space_checks_pass=True, not_applied_to_statistics_or_model_mask=True,
                           meaning='B is declared coordinate recoding;30degree gate tests residual only, not total historical physical rotation.')
        (HERE / 'prospective_coordinate_composition.json').write_text(json.dumps(composition, indent=2))
        inverse_mean = HERE / 'coarse_qc/sub-cm033_desc-scan39InDeclaredCurrentWorld_bold.nii.gz'
        command(['3dAllineate', '-source', fixed, '-master', moving, '-1Dmatrix_apply', inverse, '-final', 'wsinc5', '-prefix', inverse_mean])
        record.update(status='software_completed_samecontrast_candidate_awaiting_image_review', completed_utc=datetime.now(timezone.utc).isoformat(),
                      fit_runtime_seconds=float(runtime.duration), fit_runtime_returncode=int(runtime.returncode),
                      aligned_mean=info(aligned, fixed), inverse_mean=info(inverse_mean, moving),
                      AFNI_pullback=dict(bind(matrix), values=M.tolist()), inverse_AFNI_matrix=dict(bind(inverse), values=Mi.tolist()),
                      residual_rotation_degrees_independent_SVD=degrees, public_rotation_degrees=public_degrees,
                      residual_rotation_gate_pass=public_gate, linear_singular_values=singular.tolist(),
                      matrix_pair_max_abs_identity_error=float(np.max(np.abs(M @ Mi - np.eye(4)))),
                      prospective_composition=bind(HERE / 'prospective_coordinate_composition.json'))
        for source in sources:
            assert sha(source['path']) == source['sha256']
        status()
        print(json.dumps({k: record[k] for k in ['status', 'PID', 'fit_runtime_seconds', 'residual_rotation_degrees_independent_SVD', 'residual_rotation_gate_pass']}, indent=2), flush=True)
    except Exception as error:
        record.update(status='failed_samecontrast_candidate', error=str(error), completed_utc=datetime.now(timezone.utc).isoformat())
        status()
        raise


if __name__ == '__main__':
    main()
