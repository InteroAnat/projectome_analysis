"""Independent legacy API checks and complete regional-map value readback."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'main_scripts'))
from swc_validation import parse_swc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(output, receipt):
    if receipt.exists():
        raise ValueError('Use a fresh receipt path')
    run_path = output / 'run_provenance.json'
    run = json.loads(run_path.read_text())
    for binding in run['inputs'].values():
        assert sha(binding['path']) == binding['sha256'], 'Changed run input'
    for relative, digest in run['artifacts'].items():
        assert sha(output / relative) == digest, 'Changed artifact'
    ledger = pd.read_csv(output / 'neuron_membership.csv', dtype=str, keep_default_na=False)
    for record in ledger.itertuples():
        assert sha(record.SWCPath) == record.SWCSHA256 == run['source_hashes'][record.NeuronUID]
    spec = importlib.util.spec_from_file_location('independent_legacy_api', run['inputs']['legacy_measure_code']['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atlas_img = nib.load(run['inputs']['atlas']['path'])
    atlas = np.asarray(atlas_img.dataobj)[..., 0, :]
    ref = nib.load(run['inputs']['reference']['path'])
    key = pd.read_csv(run['inputs']['atlas_key']['path'], sep='\t')
    measures = pd.read_csv(output / 'per_neuron_legacy_regional_measures.csv')
    targets = pd.read_csv(output / 'targets.csv').set_index('TargetID')
    selected = ledger[ledger.ARMIndex.isin(['41','42','43','228','229','541','542','728','729'])]
    examples = set(selected.groupby('ARMIndex').first().NeuronUID)
    examples.update(ledger.groupby('AnimalID').first().NeuronUID)
    examples.add('251637::115.swc')
    examples.add(ledger.loc[ledger.Endpoint_node_count.astype(int).idxmax(), 'NeuronUID'])
    examples.add(ledger[ledger.EndpointEligible.eq('False')].iloc[0].NeuronUID)
    checked_examples = []
    for record in ledger[ledger.NeuronUID.isin(examples)].itertuples():
        raw = parse_swc(Path(record.SWCPath).read_text(encoding='utf-8-sig'))
        scale = np.array([float(x) for x in record.IndexScaleUm.split(';')])
        nodes = {row[0]: SimpleNamespace(id=row[0], x_nii=row[2]/scale[0], y_nii=row[3]/scale[1], z_nii=row[4]/scale[2]) for row in raw}
        parents = {row[6] for row in raw if row[6] != -1}
        neuron = SimpleNamespace(root=nodes[next(row[0] for row in raw if row[6] == -1)],
            terminal_nodes=[nodes[row[0]] for row in raw if row[0] not in parents],
            branches=[[nodes[row[6]], nodes[row[0]]] for row in raw if row[6] != -1])
        for level in range(1, 7):
            legacy = module.RegionAnalysisPerNeuron(neuron, atlas[..., level-1], key)
            legacy.run()
            observed = measures[(measures.NeuronUID.eq(record.NeuronUID)) & measures.Level.eq(level)]
            expected = {name: value for name, value in legacy.mapped_brain_region_lengths.items() if value > 0}
            actual = {targets.loc[row.TargetID].Abbreviation: row.RetainedLengthVoxel for row in observed.itertuples()}
            assert expected.keys() == actual.keys(), 'Legacy API target/gate mismatch'
            for target in expected:
                assert np.isclose(expected[target], actual[target], rtol=1e-12, atol=1e-9), 'Legacy API length mismatch'
            assert np.array_equal(observed.LegacyStrength, np.round(np.log10(observed.RetainedLengthVoxel + 1), 4))
        checked_examples.append(record.NeuronUID)
    # Independent rectangular expansion and explicit animal aggregation.
    source_table = pd.read_csv(output / 'source_target_strength.csv', dtype={'SourceARMIndex':str})
    map_index = pd.read_csv(output / 'map_index.csv', dtype={'SourceARMIndex':str})
    assert len(map_index) == 54
    map_checks = []
    strength_matrix = measures.pivot(index='NeuronUID', columns='TargetID', values='LegacyStrength').reindex(ledger.NeuronUID).fillna(0)
    length_matrix = measures.pivot(index='NeuronUID', columns='TargetID', values='RetainedLengthVoxel').reindex(ledger.NeuronUID).fillna(0)
    for row in map_index.itertuples():
        members = selected[selected.ARMIndex.eq(row.SourceARMIndex)]
        labels = atlas[..., row.TargetLevel-1]
        saved_img = nib.load(row.MapPath)
        saved = np.asarray(saved_img.dataobj)
        assert np.array_equal(saved_img.affine, ref.affine) and saved.shape == ref.shape
        assert np.isnan(saved[labels == 0]).all(), 'Background must remain unassigned'
        expected_ids = set(targets[(targets.Level == row.TargetLevel) & (targets.ARMIndex > 0)].index)
        group = source_table[source_table.SourceARMIndex.eq(row.SourceARMIndex) & source_table.Level.eq(row.TargetLevel)]
        assert set(group.TargetID) == expected_ids, 'Dropped target parcel'
        expected_voxel_values = np.full(int(labels.max()) + 1, np.nan, dtype=np.float32)
        animal_memberships = [members_of_animal for _, members_of_animal in members.groupby('AnimalID')]
        balanced_strength = pd.concat([strength_matrix.reindex(m.NeuronUID).mean() for m in animal_memberships], axis=1).mean(axis=1)
        balanced_length = pd.concat([length_matrix.reindex(m.NeuronUID).mean() for m in animal_memberships], axis=1).mean(axis=1)
        for target in group.itertuples():
            strength = float(balanced_strength.get(target.TargetID, 0.))
            assert np.isclose(strength, target.MeanLegacyStrength, atol=1e-12)
            assert np.isclose(balanced_length.get(target.TargetID, 0.), target.MeanRetainedLengthVoxel, atol=1e-9)
            assert target.ContributingAnimalCount == members.AnimalID.nunique()
            assert target.SelectedNeuronCount == len(members)
            assert int(target.ARMIndex) < len(expected_voxel_values), 'Invalid target label'
            expected_voxel_values[int(target.ARMIndex)] = strength
        assert np.array_equal(saved, expected_voxel_values[labels], equal_nan=True), 'Full parcel voxel readback differs'
        map_checks.append(dict(source=row.SourceARMIndex, target_level=row.TargetLevel, target_parcels=len(group), sha256=sha(row.MapPath)))
    result = dict(status='passed_independent_legacy_API_and_full_parcel_readback', checked_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), run_sha256=sha(run_path), checked_source_hashes=462,
        legacy_API_neuron_examples=checked_examples, actual_hierarchy_levels=6,
        maps=map_checks, scientific_anatomical_acceptance=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], source_hashes=462, legacy_API_examples=len(checked_examples), maps=len(map_checks))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'group_analysis/evolution_20261008/arm_target_strength_20261010')
    parser.add_argument('--receipt', type=Path, default=Path(__file__).with_name('legacy_parcel_independent_readback.json'))
    args = parser.parse_args()
    check(args.output, args.receipt)
