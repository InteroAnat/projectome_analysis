"""No-fit saved-grid/rotation check and readable paired-intensity QC displays."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from run_declared_coordinate_bridge import sha, bind

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'mpl_config')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def panel(master, warped, output, title, master_label, warped_label):
    a, b = nib.load(master), nib.load(warped)
    assert a.shape == b.shape and np.array_equal(a.affine, b.affine)
    a, b = nib.as_closest_canonical(a), nib.as_closest_canonical(b)
    arrays = [np.asanyarray(x.dataobj) for x in [a, b]]
    ranges = []
    normalized = []
    for values in arrays:
        low, high = np.percentile(values[np.isfinite(values)], [2, 98])
        assert high > low
        normalized.append(np.clip((values - low) / (high - low), 0, 1))
        ranges.append([float(low), float(high)])
    slices = np.unique(np.rint(np.linspace(2, a.shape[2] - 3, 6)).astype(int))
    xx, yy = np.indices(a.shape[:2])
    checker = ((xx // 12 + yy // 12) % 2).astype(bool)
    fig, axes = plt.subplots(len(slices), 3, figsize=(11, 13), squeeze=False)
    for row, z in enumerate(slices):
        planes = [normalized[0][:, :, z], normalized[1][:, :, z],
                  np.where(checker, normalized[0][:, :, z], normalized[1][:, :, z])]
        world = a.affine @ np.array([0, 0, z, 1])
        for col, plane in enumerate(planes):
            axes[row, col].imshow(plane.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f'z index {z}\nheader Z={world[2]:.1f} mm', fontsize=9)
    for ax, text in zip(axes[0], [master_label, warped_label, '12-voxel checkerboard']):
        ax.set_title(text, fontsize=10)
    fig.suptitle(title, fontsize=12, y=.985)
    fig.text(.5, .014, 'Header-canonical display only; physical LR unaccepted. Each image uses its own 2–98% intensity range.\nCheckerboard reveals local correspondence; it is not anatomical validation.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.935, bottom=.06, left=.09, right=.985, wspace=.04, hspace=.05)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return dict(bind(output), selected_canonical_z_indices=slices.tolist(), image_intensity_display_ranges=ranges,
                source_bindings=[bind(master), bind(warped)], header_only_display_directions=True)


def main():
    receipt = HERE / 'candidate_saved_readback.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    provenance = HERE / 'bridge_candidate_provenance.json'
    record = json.loads(provenance.read_text())
    assert record['status'] == 'software_completed_samecontrast_candidate_awaiting_image_review'
    for key in ['cached_mean', 'declared_current_mean', 'actual_scan39_reference', 'aligned_mean', 'inverse_mean', 'AFNI_pullback', 'inverse_AFNI_matrix']:
        assert sha(record[key]['path']) == record[key]['sha256']
    M = np.array(record['AFNI_pullback']['values'])
    Mi = np.array(record['inverse_AFNI_matrix']['values'])
    u, singular, vh = np.linalg.svd(M[:3, :3])
    rotation = u @ vh
    assert np.linalg.det(rotation) > 0
    degrees = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    assert abs(degrees - record['residual_rotation_degrees_independent_SVD']) < 1e-10
    gate = degrees <= 30
    assert gate == record['residual_rotation_gate_pass']
    pairs = [(record['aligned_mean']['path'], record['actual_scan39_reference']['path']),
             (record['inverse_mean']['path'], record['declared_current_mean']['path'])]
    for path, master in pairs:
        a, b = nib.load(path), nib.load(master)
        assert a.shape == b.shape and np.array_equal(a.affine, b.affine)
        assert np.isfinite(np.asanyarray(a.dataobj)).all()
    assert np.array_equal(np.asanyarray(nib.load(record['cached_mean']['path']).dataobj),
                          np.asanyarray(nib.load(record['declared_current_mean']['path']).dataobj))
    gate_text = 'residual gate passed' if gate else 'residual gate failed'
    title = f'CM033 same-contrast candidate: {degrees:.3f} degrees, {gate_text}\nB is coordinate recoding; historical payload and physical LR unaccepted'
    displays = [panel(record['actual_scan39_reference']['path'], record['aligned_mean']['path'],
                      HERE / 'coarse_qc/cm033_forward_mean_intensity_correspondence.png', title,
                      'Actual scan-39 reference', 'Fitted recoded 1472-volume mean'),
                panel(record['declared_current_mean']['path'], record['inverse_mean']['path'],
                      HERE / 'coarse_qc/cm033_inverse_mean_intensity_correspondence.png', title,
                      'B-recoded 1472-volume mean', 'Inverse-fitted scan-39 reference')]
    output = dict(status='saved_software_readback_pass_image_review_pending', checked_utc=datetime.now(timezone.utc).isoformat(),
                  provenance=bind(provenance), code=bind(__file__), residual_rotation_degrees=degrees,
                  residual_30_degree_gate_pass=gate, matrix_inverse_max_error=float(np.max(np.abs(M @ Mi - np.eye(4)))),
                  exact_saved_grids_and_finite_values=True, exact_scaled_array_preservation=True,
                  displays=displays, statistics_or_masks_warped=False, third_fit_started=False, anatomical_acceptance=False)
    with receipt.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ['status', 'residual_rotation_degrees', 'residual_30_degree_gate_pass']}, indent=2))


if __name__ == '__main__':
    main()
