"""Display legacy retained-length strength on the six actual ARM target grids.

Fresh source-bound calculation preserves legacy all-compartment edges, proximal
label assignment, three-decimal edge rounding and region-wide graph-leaf gate.
The per-neuron log transform is retained. This is not an arbor segmentation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'main_scripts'))
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from swc_validation import parse_swc
from map_projections_by_soma_origin import NAMED_INSULA_INDICES

TABLES = ROOT / 'notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462'
DEFAULT_OUTPUT = ROOT / 'group_analysis/evolution_20261008/arm_target_strength_20261010'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retained_lengths(rows, atlas, scale):
    """One graph/edge calculation, followed by reductions against six volumes."""
    data = np.asarray(rows, dtype=float)
    ids = {int(row[0]): i for i, row in enumerate(rows)}
    parents = data[:, 6].astype(int)
    edges = np.flatnonzero(parents != -1)
    proximal = np.array([ids[int(parents[i])] for i in edges])
    xyz = data[:, 2:5] / np.asarray(scale)
    voxels = np.rint(xyz).astype(int)
    inside = ((voxels >= 0) & (voxels < np.array(atlas.shape[:3]))).all(axis=1)
    labels = np.zeros((len(rows), 6), dtype=int)
    labels[inside] = atlas[tuple(voxels[inside].T)]
    # Python round matches RegionAnalysisPerNeuron rather than changing its rule.
    lengths = np.array([round(float(np.sqrt(np.sum((xyz[p] - xyz[c]) ** 2))), 3)
                        for p, c in zip(proximal, edges)])
    parent_ids = set(parents[parents != -1])
    leaves = np.array([int(row[0]) not in parent_ids for row in rows])
    outputs = []
    for level in range(6):
        receiving = set(labels[leaves & inside, level]) - {0}
        edge_labels = labels[proximal, level]
        totals = np.bincount(edge_labels, weights=lengths, minlength=int(labels.max()) + 1)
        outputs.append({index: float(totals[index]) for index in sorted(receiving)
                        if totals[index] > 0})
    return outputs, dict(NodeCount=len(rows), EdgeCount=len(edges),
                        AllCompartmentLengthVoxel=float(lengths.sum()),
                        GraphLeafCount=int(leaves.sum()),
                        OutsideReferenceLeafCount=int((leaves & ~inside).sum()))


def summarize_by_animal(frame):
    """Zeros for selected neurons are measured zeros; missing animals stay absent."""
    animal = frame.groupby(['SourceARMIndex', 'AnimalID', 'Level', 'TargetID'], sort=True).agg(
        MeanRetainedLengthVoxel=('RetainedLengthVoxel', 'mean'),
        MeanLegacyStrength=('LegacyStrength', 'mean'),
        SelectedNeuronCount=('NeuronUID', 'size')).reset_index()
    group = animal.groupby(['SourceARMIndex', 'Level', 'TargetID'], sort=True).agg(
        MeanRetainedLengthVoxel=('MeanRetainedLengthVoxel', 'mean'),
        MeanLegacyStrength=('MeanLegacyStrength', 'mean'),
        ContributingAnimalCount=('AnimalID', 'size'),
        SelectedNeuronCount=('SelectedNeuronCount', 'sum')).reset_index()
    return animal, group


def build(output):
    require(not output.exists(), 'Preserve existing derivatives: use a fresh output directory')
    provenance_path = TABLES / 'export_provenance.json'
    prior = json.loads(provenance_path.read_text())
    require(prior['status'] == 'software_verified_descriptive_tables', 'Validated source export required')
    source_paths = {k: Path(v) for k, v in {
        'ledger': TABLES / 'neuron_summary.csv', 'targets': TABLES / 'targets.csv',
        'reference': next(p for p in prior['source_bindings'] if p.endswith('NMT_v2.1_sym_SS.nii.gz')),
        'atlas': next(p for p in prior['source_bindings'] if p.endswith('ARM_in_NMT_v2.1_sym.nii.gz')),
        'atlas_key': next(p for p in prior['source_bindings'] if p.endswith('ARM_key_all.txt')),
        'source_export': provenance_path,
        'legacy_measure_code': ROOT / 'main_scripts/region_analysis/neuron_analysis.py',
        'legacy_strength_code': ROOT / 'main_scripts/region_analysis/population.py',
        'parser': ROOT / 'main_scripts/swc_validation.py', 'producer': Path(__file__),
    }.items()}
    bindings = {k: dict(path=str(p), sha256=sha(p)) for k, p in source_paths.items()}
    for name in ('reference', 'atlas', 'atlas_key'):
        require(bindings[name]['sha256'] == prior[name + '_sha256'], 'Changed source binding: ' + name)
    ledger = pd.read_csv(source_paths['ledger'], dtype=str, keep_default_na=False)
    targets = pd.read_csv(source_paths['targets'], dtype=str, keep_default_na=False)
    require(len(ledger) == 462 and ledger.NeuronUID.is_unique, 'Exact unique 462-neuron ledger required')
    require(ledger.CoordinateFrame.eq('atlas_index_um').all(), 'Declared atlas-index coordinates required')
    ref = nib.load(source_paths['reference'])
    atlas_image = nib.load(source_paths['atlas'])
    require(atlas_image.shape == ref.shape + (1, 6) and np.array_equal(atlas_image.affine, ref.affine), 'ARM geometry differs')
    atlas = np.asarray(atlas_image.dataobj, dtype=np.int32)[..., 0, :]
    key = pd.read_csv(source_paths['atlas_key'], sep='\t', dtype=str).set_index('Index')
    label_targets = {}
    for row in targets.itertuples():
        if row.ARMIndex and int(float(row.ARMIndex)) > 0:
            index = int(float(row.ARMIndex))
            require(row.OfficialFullName == key.loc[str(index), 'Full_Name'], 'Full ARM name mismatch')
            label_targets[(int(row.Level), index)] = row.TargetID
    sparse, qc = [], []
    for position, record in enumerate(ledger.to_dict('records')):
        require(sha(record['SWCPath']) == record['SWCSHA256'], 'Changed reconstruction ' + record['NeuronUID'])
        scale = [float(v) for v in record['IndexScaleUm'].split(';')]
        require(scale == [250., 250., 250.], 'This legacy voxel-unit run requires the verified isotropic scale')
        rows = parse_swc(Path(record['SWCPath']).read_text(encoding='utf-8-sig'), record['NeuronUID'])
        measures, metrics = retained_lengths(rows, atlas, scale)
        qc.append(dict(NeuronUID=record['NeuronUID'], SWCSHA256=record['SWCSHA256'], **metrics))
        for level, lengths in enumerate(measures, 1):
            for index, length in lengths.items():
                require((level, index) in label_targets, 'Unresolved positive atlas label')
                sparse.append(dict(NeuronUID=record['NeuronUID'], AnimalID=record['AnimalID'],
                    SourceARMIndex=record['ARMIndex'], Level=level, TargetID=label_targets[(level, index)],
                    RetainedLengthVoxel=length, LegacyStrength=float(np.round(np.log10(length + 1), 4))))
        if position % 50 == 0:
            print(f'Calculated legacy regional measures: {position + 1}/462', flush=True)
    output.mkdir(parents=True)
    (output / 'nifti').mkdir()
    (output / 'figures').mkdir()
    pd.DataFrame(sparse).to_csv(output / 'per_neuron_legacy_regional_measures.csv', index=False)
    pd.DataFrame(qc).to_csv(output / 'per_neuron_graph_qc.csv', index=False)
    ledger.to_csv(output / 'neuron_membership.csv', index=False)
    targets.to_csv(output / 'targets.csv', index=False)
    # Complete rectangular matrices include all measured zeros and source denominators.
    selected = ledger[ledger.ARMIndex.astype(int).isin(NAMED_INSULA_INDICES)]
    blocks = []
    sparse_frame = pd.DataFrame(sparse).set_index(['NeuronUID', 'TargetID'])
    for level in range(1, 7):
        target_level = targets[targets.Level.eq(str(level)) & targets.ARMIndex.ne('') & targets.ARMIndex.astype(str).ne('0.0')]
        target_level = target_level[target_level.ARMIndex.astype(float).gt(0)]
        rectangular = pd.MultiIndex.from_product([selected.NeuronUID, target_level.TargetID], names=['NeuronUID', 'TargetID'])
        measures = sparse_frame[['RetainedLengthVoxel', 'LegacyStrength']].reindex(rectangular, fill_value=0).reset_index()
        measures = measures.merge(selected[['NeuronUID', 'AnimalID', 'ARMIndex']], on='NeuronUID', validate='many_to_one').rename(columns={'ARMIndex': 'SourceARMIndex'})
        measures['Level'] = level
        blocks.append(measures)
    animal, group = summarize_by_animal(pd.concat(blocks, ignore_index=True))
    group = group.merge(targets, on=['TargetID'], validate='many_to_one', suffixes=('', '_metadata')).drop(columns='Level_metadata')
    group['SourceARMFullName'] = group.SourceARMIndex.map(lambda x: key.loc[str(x), 'Full_Name'])
    animal.to_csv(output / 'per_animal_source_target_strength.csv', index=False)
    group.to_csv(output / 'source_target_strength.csv', index=False)
    # Six legacy-style neuron x target matrices; columns carry official full labels.
    for level, block in enumerate(blocks, 1):
        names = targets.set_index('TargetID').OfficialFullName.to_dict()
        for measure, suffix in [('RetainedLengthVoxel', 'retained_length_voxel'), ('LegacyStrength', 'legacy_strength')]:
            matrix = block.pivot(index='NeuronUID', columns='TargetID', values=measure).rename(columns=names)
            require(matrix.columns.is_unique, 'Target full labels collide')
            matrix.to_csv(output / f'neuron_target_L{level}_{suffix}.csv')
    maximum = float(group.MeanLegacyStrength.max())
    artifacts = []
    reference = ref.get_fdata(dtype=np.float32)
    largest_label = int(atlas.max())
    for source in NAMED_INSULA_INDICES:
        rows = selected[selected.ARMIndex.eq(str(source))]
        full_name = key.loc[str(source), 'Full_Name']
        fig, axes = plt.subplots(6, 3, figsize=(13, 19), layout='constrained')
        fig.suptitle('ARM target-parcel projection strength\nFrom somata assigned to ' + full_name.replace('_', ' ') +
                     f'\n{len(rows)} reconstructions; {rows.AnimalID.nunique()} contributing monkeys', fontsize=13)
        for level in range(1, 7):
            values = group[group.SourceARMIndex.eq(str(source)) & group.Level.eq(level)]
            lookup = np.full(largest_label + 1, np.nan, dtype=np.float32)
            for row in values.itertuples():
                lookup[int(float(row.ARMIndex))] = row.MeanLegacyStrength
            data = lookup[atlas[..., level - 1]]
            filename = f'from-ARM6-{source}_space-NMTv2.1sym_target-ARM{level}_desc-meanLegacyStrength_map.nii.gz'
            path = output / 'nifti' / filename
            image = nib.Nifti1Image(data, ref.affine)
            image.header.set_xyzt_units('mm')
            image.set_sform(ref.affine, code=2)
            image.set_qform(None, code=0)
            image.header['descrip'] = f'ARM{level} parcel mean legacy log10(voxel length+1); source ARM6 {source}'
            nib.save(image, path)
            saved = nib.load(path)
            require(np.array_equal(np.asarray(saved.dataobj), data, equal_nan=True) and np.array_equal(saved.affine, ref.affine), 'NIfTI readback differs')
            artifacts.append(dict(SourceARMIndex=source, SourceARMFullName=full_name, TargetLevel=level,
                MapPath=str(path.resolve()), SHA256=sha(path), SelectedNeuronCount=len(rows),
                ContributingAnimalCount=rows.AnimalID.nunique()))
            for col, (axis, cut) in enumerate([(2, 87), (1, 200), (0, 56)]):
                ax = axes[level - 1, col]
                plane = [i for i in range(3) if i != axis]
                extent = [ref.affine[i, 3] + (v - .5) * ref.affine[i, i] for i in plane for v in (0, ref.shape[i])]
                ax.imshow(np.take(reference, cut, axis=axis).T, cmap='gray', vmin=75, vmax=920, origin='lower', extent=extent)
                shown = np.take(data, cut, axis=axis).T
                overlay = ax.imshow(np.ma.masked_where(~np.isfinite(shown) | (shown <= 0), shown),
                    cmap='magma', norm=Normalize(0, maximum), origin='lower', extent=extent, interpolation='nearest', alpha=.85)
                ax.set_title(f'Target ARM level {level}; {"XYZ"[axis]}={ref.affine[axis,3] + cut * .25:.2f} mm', fontsize=9)
                ax.set_xlabel('XYZ'[plane[0]] + ' (declared NMT mm)', fontsize=7)
                ax.set_ylabel('XYZ'[plane[1]] + ' (declared NMT mm)', fontsize=7)
                ax.tick_params(labelsize=7)
        fig.colorbar(overlay, ax=axes, shrink=.5, label='Equal-monkey mean of per-neuron log10(retained voxel length + 1)')
        fig.savefig(output / 'figures' / f'from-ARM6-{source}_target-ARM1to6_legacy_strength.png', dpi=130)
        plt.close(fig)
    pd.DataFrame(artifacts).to_csv(output / 'map_index.csv', index=False)
    files = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
    receipt = dict(status='software_checked_descriptive_legacy_strength_maps', created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=bindings, source_hashes=dict(zip(ledger.NeuronUID, ledger.SWCSHA256)), artifacts=files,
        definition='all-compartment proximal edge lengths; Python round(edge,3); retain positive target regions containing any graph leaf; per-neuron round(log10(length+1),4)',
        unit='atlas-index voxel length before log; no claim of native tissue length',
        hierarchy='same legacy rule recalculated independently at six actual ARM volumes; not historical parent pooling',
        group_policy='mean per-neuron logged strength within selected animal/source; equal contributing animal means; absent source/animal excluded',
        display=dict(common_limits=[0,maximum], slices_XYZ=[56,200,87], MRI_limits=[75,920], smoothing=False, MIP=False, clipping=False),
        selected_neurons=462, named_origin_neurons=301, scientific_anatomical_acceptance=False,
        interpretation='Uniform target-parcel summary, not voxel-level measurement; graph leaves are not verified terminal fields; zero is reconstructed evidence only',
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, nibabel=nib.__version__, matplotlib=matplotlib.__version__))
    (output / 'run_provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], maps=len(artifacts), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
