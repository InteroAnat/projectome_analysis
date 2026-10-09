"""One isolated public-DEB cached-SPM mean to frozen taskA spatial candidate.

External inputs/code are read only. No array flip, header repair, fresh-series
matrix substitution, GLM rerun or anatomical acceptance is performed here.
"""
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
DEB = Path("/mnt/c/Users/laika_yan/Documents/ChatGPT/deb_fmri")
MOVING = ROOT / "notes/region_analysis_review_20261009/cm033_integration_followup_20261009/sub-cm033_ses-20160120_desc-currentPayload1472VolumeMean_space-cachedSPM_bold.nii.gz"
FIXED = DEB / "alignment_review/cm033_Dsource_20261009/anatomy_candidates/sub-cm033_srcdate-unknown_desc-taskAHeaderRepair_brain.nii"
WORKFLOW_SOURCE = DEB / "fmri_pipeline/workflows/nipype_anat_epi.py"
EXPECTED = {MOVING: "b964c0424e872e397e273851809c19d7ac5875c866736ff4250faffbae5157d8",
            FIXED: "f0a95a6b7e666c3b58a5e80fd179178bdca1da6e9b8405cbba730c26cb49c576",
            WORKFLOW_SOURCE: "b50e42118ad6c461eb3c4be0e3c5c40c531e9f83914b8ede3d49bc11b4982bc1"}
FIT = HERE / "fit"
QC = HERE / "coarse_qc"
STATUS = HERE / "bridge_candidate_provenance.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_status(value):
    STATUS.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(command, *, capture=False):
    print("COMMAND", json.dumps([str(x) for x in command]), flush=True)
    result = subprocess.run([str(x) for x in command], cwd=HERE, check=True,
                            capture_output=True, text=True)
    with (HERE / "afni_commands.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps([str(x) for x in command])+"\n"+result.stdout+result.stderr+"\n")
    return result.stdout if capture else result


def main():
    if STATUS.exists() or FIT.exists() or QC.exists():
        raise FileExistsError("Existing bridge candidate; never start a duplicate fit")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source/code hash changed: {path}")
    for name in ["nipype_work", "nipype_config", "logs", "mpl_config", "tmp", "fit", "coarse_qc"]:
        (HERE / name).mkdir()
    os.chdir(HERE)
    os.environ["PATH"] = "/home/binbin/abin:"+os.environ["PATH"]
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NIPYPE_CONFIG_DIR"] = str(HERE / "nipype_config")
    os.environ["MPLCONFIGDIR"] = str(HERE / "mpl_config")
    os.environ["TMPDIR"] = str(HERE / "tmp")
    os.environ["AFNI_DONT_LOGFILE"] = "YES"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(DEB))
    import nibabel as nib
    import numpy as np
    import nipype
    from nipype import config, logging
    from fmri_pipeline.workflows.nipype_anat_epi import init_epi_anat_wf
    from fmri_pipeline.preproc.run_anat_epi_bids import validate_registration_rotation
    from fmri_pipeline.qc.overlay import write_anat_epi_overlay
    loaded = Path(inspect.getsourcefile(init_epi_anat_wf)).resolve()
    if loaded != WORKFLOW_SOURCE.resolve() or sha(loaded) != EXPECTED[WORKFLOW_SOURCE]:
        raise ValueError("Imported workflow does not match the verified public source")
    config.set("logging", "log_directory", str(HERE / "logs"))
    logging.update_logging(config)

    def image_info(path, reference=None):
        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        qform, qcode = image.header.get_qform(coded=True)
        sform, scode = image.header.get_sform(coded=True)
        if (data.ndim != 3 or not np.isfinite(data).all() or not np.isfinite(image.affine).all()
                or image.header.get_xyzt_units()[0] != "mm" or not (qcode or scode)):
            raise ValueError(f"Invalid finite/coded-mm 3D image: {path}")
        if qcode and scode and not np.allclose(qform, sform, atol=1e-4):
            raise ValueError(f"Conflicting coded image transforms: {path}")
        if reference is not None and (image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5)):
            raise ValueError(f"Saved image does not match requested master grid: {path}")
        return {"path": str(path), "sha256": sha(path), "shape": image.shape,
                "affine_mm": image.affine.tolist(), "header_axcodes": list(nib.aff2axcodes(image.affine)),
                "qform_code": int(qcode), "sform_code": int(scode), "finite": True,
                "minimum": float(data.min()), "maximum": float(data.max()),
                "nonzero_voxels": int(np.count_nonzero(data))}

    record = {"status": "running_projectome_only_spatial_bridge_candidate", "started_utc": datetime.now(timezone.utc).isoformat(),
              "PID": os.getpid(), "python": sys.executable, "nipype_version": nipype.__version__,
              "workflow_API": "init_epi_anat_wf(brain=frozen_taskA,t1w=None,refmean=cachedSPM_mean,out_dir=fit,warp_type=shift_rotate,base_dir=nipype_work)",
              "workflow_source": {"path": str(loaded), "sha256": sha(loaded), "API_signature": str(inspect.signature(init_epi_anat_wf))},
              "public_code_reference_commit": "73544f327c0f03b3c70700939a517049dcc1e0e7",
              "parameters": {"warp_type": "shift_rotate", "cost": "lpa", "center_of_mass": True, "final_interpolation": "wsinc5", "OMP_NUM_THREADS": 4},
              "moving": image_info(MOVING), "fixed": image_info(FIXED),
              "input_array_or_header_modifications": False, "fresh_EPI_matrix_substitution": False,
              "external_source_writes_or_job_control": False, "MATLAB_used": False,
              "historical_estimation_payload_identity": "unproven", "physical_laterality": "unaccepted",
              "anatomical_acceptance": False, "driver_sha256": sha(Path(__file__))}
    write_status(record)
    try:
        help_text = subprocess.run(["3dAllineate", "-help"], capture_output=True, text=True, check=True).stdout
        (HERE / "afni_3dAllineate_help.txt").write_text(help_text)
        afni_version = subprocess.run(["afni", "-ver"], capture_output=True, text=True, check=True)
        record["AFNI_version"] = afni_version.stdout.strip()
        record["AFNI_executable_sha256"] = sha(Path("/home/binbin/abin/3dAllineate"))
        workflow = init_epi_anat_wf(brain=FIXED, t1w=None, refmean=MOVING, out_dir=FIT,
                                   warp_type="shift_rotate", name="cm033_cachedSPM_to_taskA_candidate",
                                   base_dir=HERE / "nipype_work")
        node = workflow.get_node("allineate_refmean_to_brain")
        record["fit_command"] = node.interface.cmdline
        write_status(record)
        print("FIT", record["fit_command"], flush=True)
        workflow.run(plugin="Linear")
        aligned = FIT / "desc-refmeanInAnat.nii"
        matrix = FIT / "desc-epiToAnatAffine.aff12.1D"
        inverse = FIT / "desc-taskAToCachedEpiApply.aff12.1D"
        inverse.write_text(run(["cat_matvec", matrix, "-I"], capture=True))
        forward_values = np.loadtxt(matrix, comments="#").reshape(3, 4)
        inverse_values = np.loadtxt(inverse, comments="#").reshape(3, 4)
        forward4, inverse4 = np.eye(4), np.eye(4)
        forward4[:3], inverse4[:3] = forward_values, inverse_values
        if not np.isfinite(forward4).all() or not np.isfinite(inverse4).all() or not np.allclose(forward4 @ inverse4, np.eye(4), atol=1e-4):
            raise ValueError("Nonfinite or inconsistent fitted affine/inverse")
        try:
            rotation = validate_registration_rotation(matrix)
            rotation_gate = "passes_30_degree_software_review_gate"
        except RuntimeError as error:
            rotation = None
            rotation_gate = str(error)
        moving_image, fixed_image = nib.load(MOVING), nib.load(FIXED)
        support = QC / "sub-cm033_desc-taskAPositiveSupport_mask.nii.gz"
        data = np.asanyarray(fixed_image.dataobj)
        support_image = nib.Nifti1Image((data > 0).astype(np.uint8), fixed_image.affine, fixed_image.header.copy())
        support_image.set_data_dtype(np.uint8)
        support_image.set_sform(fixed_image.get_sform(), int(fixed_image.header["sform_code"]))
        support_image.set_qform(fixed_image.get_qform(), int(fixed_image.header["qform_code"]))
        nib.save(support_image, support)
        inverse_brain = QC / "sub-cm033_desc-taskAInCachedEpi_brain.nii.gz"
        inverse_support = QC / "sub-cm033_desc-taskAPositiveSupportInCachedEpi_mask.nii.gz"
        run(["3dAllineate", "-source", FIXED, "-master", MOVING, "-1Dmatrix_apply", inverse,
             "-final", "wsinc5", "-prefix", inverse_brain])
        run(["3dAllineate", "-source", support, "-master", MOVING, "-1Dmatrix_apply", inverse,
             "-final", "NN", "-prefix", inverse_support])
        forward_png = QC / "cm033_cachedSPM_mean_on_taskA_candidate.png"
        inverse_png = QC / "cm033_taskA_support_on_cachedSPM_mean_candidate.png"
        title = "CM033 isolated spatial bridge candidate; physical LR and historical payload unaccepted"
        write_anat_epi_overlay(aligned, FIXED, forward_png, mask=support, n_slices=6, title=title,
                               underlay_label="registered cached-SPM mean", contour_label="positive taskA intensity support (QC only)")
        write_anat_epi_overlay(MOVING, inverse_brain, inverse_png, mask=inverse_support, n_slices=6, title=title,
                               underlay_label="original cached-SPM mean", contour_label="inverse-warped taskA positive support (QC only)")
        record.update({"status": "software_completed_spatial_bridge_candidate_awaiting_image_review",
                       "completed_utc": datetime.now(timezone.utc).isoformat(),
                       "aligned_mean": image_info(aligned, fixed_image), "inverse_anatomy": image_info(inverse_brain, moving_image),
                       "inverse_support": image_info(inverse_support, moving_image),
                       "inverse_support_binary": bool(set(np.unique(np.asanyarray(nib.load(inverse_support).dataobj)).tolist()) <= {0., 1.}),
                       "matrix": {"path": str(matrix), "sha256": sha(matrix), "values": forward4.tolist(),
                                  "semantics": "AFNI application matrix maps output/base taskA DICOM coordinates to input/source cached-EPI coordinates (pullback), not a source-to-target point map"},
                       "inverse_matrix": {"path": str(inverse), "sha256": sha(inverse), "values": inverse4.tolist(),
                                          "semantics": "cat_matvec -I; AFNI apply matrix for taskA image onto cached-EPI grid"},
                       "linear_determinant": float(np.linalg.det(forward4[:3, :3])),
                       "linear_singular_values": np.linalg.svd(forward4[:3, :3], compute_uv=False).tolist(),
                       "rotation_degrees": rotation, "rotation_review_gate": rotation_gate,
                       "coarse_QC_pngs": [{"path": str(path), "sha256": sha(path)} for path in [forward_png, inverse_png]],
                       "QC_mask_policy": "positive-intensity support of frozen taskA, display-only; not an accepted tissue mask",
                       "artifacts": [{"path": str(path), "sha256": sha(path)} for path in sorted(FIT.iterdir()) if path.is_file()]})
        for path, expected in EXPECTED.items():
            if sha(path) != expected:
                raise RuntimeError(f"Frozen source changed during candidate fit: {path}")
        write_status(record)
        print(json.dumps({key: record[key] for key in ["status", "PID", "rotation_degrees", "linear_determinant", "rotation_review_gate"]}, indent=2), flush=True)
    except Exception as error:
        record.update(status="failed_projectome_only_bridge_candidate", error=str(error), completed_utc=datetime.now(timezone.utc).isoformat())
        write_status(record)
        raise


if __name__ == "__main__":
    main()
