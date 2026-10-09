"""Additive endpoint-voxel markers on fixed MRI slices; original maps unchanged."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd

from inspect_projection_maps import digest, require


def slice_markers(values, axis, cut, affine):
    """One dot per positive voxel on exactly one plane, with no depth pooling."""
    plane = np.take(values, cut, axis=axis)
    first, second = np.nonzero(plane > 0)
    in_plane = [a for a in range(3) if a != axis]
    coordinates = np.zeros((len(first), 3))
    coordinates[:, axis] = cut
    coordinates[:, in_plane[0]], coordinates[:, in_plane[1]] = first, second
    mm = nib.affines.apply_affine(affine, coordinates)
    return mm[:, in_plane[0]], mm[:, in_plane[1]], plane[first, second]


def render(run_directory, inspection, output, cuts=(56, 200, 87)):
    run_directory, inspection, output = map(Path, (run_directory, inspection, output))
    require(not output.exists(), 'Preserve existing figures; use a fresh output')
    receipt = json.loads(inspection.read_text())
    run_path = run_directory / 'run_provenance.json'
    require(receipt['status'] == 'passed_full_voxel_relations_and_direct_graph_endpoints', 'Successful second inspection required')
    require(receipt['runs']['main_endpoints']['sha256'] == digest(run_path), 'Run differs from inspection')
    run = json.loads(run_path.read_text())
    affine = np.array(run['affine_mm'])
    require(np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))) and np.all(np.diag(affine[:3, :3]) > 0), 'Positive axis-aligned grid required')
    reference_binding, mask_binding = (run['inputs'][key] for key in ('reference', 'brain_mask'))
    for binding in (reference_binding, mask_binding, run['inputs']['atlas_key'], run['inputs']['manifest']):
        require(digest(binding['path']) == binding['sha256'], 'Bound display source changed')
    reference_image, mask_image = (nib.load(binding['path']) for binding in (reference_binding, mask_binding))
    reference, mask = reference_image.get_fdata(), mask_image.get_fdata() > 0
    require(mask.shape == reference.shape and np.allclose(mask_image.affine, affine) and np.allclose(reference_image.affine, affine), 'Background grid mismatch')
    require(np.isfinite(reference).all() and mask.any(), 'Invalid MRI window source')
    require(len(cuts) == 3 and all(0 <= cut < reference.shape[a] for a, cut in enumerate(cuts)), 'Cuts outside MRI')
    low, high = np.percentile(reference[mask], [1, 99.5])
    ledger = pd.read_csv(run_directory / 'input_manifest.csv', dtype=str, keep_default_na=False)
    sources = ledger[['Subregion', 'ARMFullName', 'ARMIndex', 'SourceARMStatus']].drop_duplicates().set_index('Subregion')
    require(not sources.index.duplicated().any(), 'Contradictory source labels')
    prepared, positives = [], []
    for entry in run['group_maps']:
        metadata = sources.loc[entry['Subregion']]
        if metadata.SourceARMStatus != 'mapped' or not entry['n_animals']:
            continue
        path = run_directory / entry['density_path']
        require(digest(path) == entry['density_sha256'], 'Endpoint map changed')
        image = nib.load(path)
        require(image.shape == reference.shape and np.allclose(image.affine, affine), 'Map grid mismatch')
        data = np.asarray(image.dataobj)
        require(np.isfinite(data).all() and np.all(data >= 0), 'Invalid endpoint values')
        planes = [slice_markers(data, a, cuts[a], affine) for a in (2, 1, 0)]
        positives.extend(np.log10(1 + p[2]) for p in planes if len(p[2]))
        prepared.append((entry, metadata, planes))
    require(bool(prepared), 'No mapped source groups')
    upper = float(np.percentile(np.concatenate(positives), 99.5)) if positives else 1.0
    output.mkdir(parents=True)
    figures, panel_counts = [], []
    for start in range(0, len(prepared), 4):
        batch = prepared[start:start + 4]
        fig, axes = plt.subplots(len(batch), 3, figsize=(14, 3.65 * len(batch)), squeeze=False, layout='constrained')
        for row, (entry, metadata, planes) in enumerate(batch):
            for col, axis in enumerate((2, 1, 0)):
                panel = axes[row, col]
                in_plane = [a for a in range(3) if a != axis]
                extent = [affine[a, 3] + (edge - .5) * affine[a, a] for a in in_plane for edge in (0, reference.shape[a])]
                panel.imshow(np.take(reference, cuts[axis], axis=axis).T, cmap='gray', norm=Normalize(low, high, clip=True), extent=extent, origin='lower', interpolation='nearest')
                x, y, density = planes[col]
                # Fixed-area glyphs show occupied voxel centres. A thin white
                # halo plus black rim improves contrast on dark and light MRI.
                panel.scatter(x, y, s=17, c='white', linewidths=0, zorder=2)
                plotted = panel.scatter(x, y, s=9, c=np.log10(1 + density), cmap='viridis', norm=Normalize(0, upper, clip=True), edgecolors='black', linewidths=.25, zorder=3)
                panel.set_xlim(extent[:2]); panel.set_ylim(extent[2:])
                title = f"{'XYZ'[axis]} voxel {cuts[axis]} ({affine[axis, 3] + cuts[axis] * affine[axis, axis]:.2f} template mm)"
                if col == 0:
                    title = textwrap.fill(metadata.ARMFullName.replace('_', ' '), 45) + f"\nARM {metadata.ARMIndex}; {entry['n_animals']} animals; eligible {entry['n_computable']}/{entry['n_selected']}\n" + title
                panel.set_title(title, fontsize=9)
                panel.set_xlabel(f"{'XYZ'[in_plane[0]]} (template mm)", fontsize=8)
                panel.set_ylabel(f"{'XYZ'[in_plane[1]]} (template mm)", fontsize=8)
                panel.tick_params(labelsize=7)
                panel_counts.append({'source': entry['Subregion'], 'axis': 'XYZ'[axis], 'cut': int(cuts[axis]), 'occupied_voxels_drawn': len(x), 'slice_density_sum': float(density.sum(dtype=np.float64))})
        fig.suptitle('Candidate axon-end locations on symmetric NMT v2.1 MRI\nOne fixed-size dot per occupied 0.25 mm voxel; colour = equal-animal endpoint density\nSingle voxel slices; no MIP or smoothing; biological terminal fields unverified', fontsize=11)
        fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=.7, label='log10(1 + candidate ends / template mm³ / eligible neuron)')
        path = output / f'space-NMTv2p1_desc-candidateAxonEndVoxelMarkers_page-{start // 4 + 1:02}.png'
        fig.savefig(path, dpi=160); plt.close(fig)
        figures.append({'path': path.name, 'sha256': digest(path)})
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'renderer_sha256': digest(__file__), 'inspection_sha256': digest(inspection),
              'run_provenance_sha256': digest(run_path), 'background': reference_binding, 'brain_mask': mask_binding, 'MRI_window_percentiles': [1, 99.5], 'MRI_intensity_limits': [float(low), float(high)],
              'cuts_XYZ_voxels': list(cuts), 'marker_area_points_squared': 9, 'halo_area_points_squared': 17, 'glyph_semantics': 'occupied endpoint voxel centre; not one dot per neuron, end, arbor or synapse',
              'colour': 'log10(1 + equal-animal candidate-end density)', 'shared_colour_upper': upper, 'colour_upper_percentile': 99.5, 'smoothing': None, 'MIP': False,
              'anatomical_acceptance': False, 'source_maps_modified': False, 'unknown_source_locations': 'retained in ledger/QC, excluded from named anatomy figures', 'panel_counts': panel_counts, 'figures': figures}
    (output / 'display_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('run', 'inspection', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--slice-voxels', type=int, nargs=3, default=[56, 200, 87])
    args = parser.parse_args()
    print(len(render(args.run, args.inspection, args.output, args.slice_voxels)['figures']), 'additive sheets')
