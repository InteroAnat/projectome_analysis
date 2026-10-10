"""Trace unregistered dataset 250432 and the sole added 251637 selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVO = ROOT / 'group_analysis/evolution_20261008'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh investigation destination')
    paths = {
        'review': EVO / 'classification/coarse_insula_review_20261009/distance_priority_v2_20261009/all_neuron_review_manifest_distance_corrected.csv',
        'catalog': EVO / 'inventory/live_inventory_20261008/live_snapshots/20261008T152413Z/catalog_sample_info.json',
        'sample_inventory': EVO / 'inventory/live_inventory_20261008/sample_inventory.csv',
        'older353': ROOT / 'group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx',
        'current462': ROOT / 'notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv',
        'registry': ROOT / 'group_analysis/docs/dataset_status_manifest.csv',
        'experiment_summary': ROOT / 'group_analysis/docs/monkey_experiment_data_summary.xlsx',
        'atlas': ROOT / 'atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz',
        'key': ROOT / 'atlas/ARM_key_all.txt',
    }
    review = pd.read_csv(paths['review'], dtype=str, keep_default_na=False)
    catalog = json.loads(paths['catalog'].read_text(encoding='utf-8'))
    def metadata_rows(value):
        if isinstance(value, dict):
            if value.get('fMOST_id') == '250432':
                yield value
            for child in value.values():
                yield from metadata_rows(child)
        elif isinstance(value, list):
            for child in value:
                yield from metadata_rows(child)
    metadata = list(metadata_rows(catalog))
    assert len(metadata) == 1 and metadata[0]['sample_id'] == 'ION2508004'
    assert metadata[0]['injection_region'] == '' and metadata[0]['tracing_cell_number'] == 128
    sample = review[review['sample'].eq('250432')].copy()
    assert len(sample) == 234 and sample.neuron_id.nunique() == 234 and sample.registry_animal.eq('').all()
    atlas = np.asarray(nib.load(paths['atlas']).dataobj)[..., 0, 5]
    key = pd.read_csv(paths['key'], sep='\t', dtype=str).set_index('Index')
    direct_rows = []
    for row in sample.to_dict('records'):
        path = ROOT / row['map_selected_source']
        assert sha(path) == row['map_selected_sha256']
        graph = np.loadtxt(path, comments='#', ndmin=2)
        roots = graph[graph[:, 6] == -1]
        assert len(roots) == 1 and np.isfinite(graph).all()
        point = roots[0, 2:5] / 250
        voxel = np.rint(point).astype(int)
        assert ((voxel >= 0) & (voxel < atlas.shape)).all()
        index = int(atlas[tuple(voxel)])
        name = key.loc[str(index)].Full_Name if index else 'Atlas label 0 (unassigned)'
        direct_rows.append({'SampleID': '250432', 'NeuronID': row['neuron_id'], 'verified_monkey_id': '',
                            'portal_region': row['portal_region'], 'portal_INS_flag': row['atlas_INS'],
                            'spatial_candidate_flag': row['potential_INS_candidate'], 'selection_group': row['exclusive_map_group'],
                            'ARMIndex': index, 'OfficialARMFullName': name, 'source_root_index_xyz': json.dumps(point.tolist()),
                            'source_sha256': row['map_selected_sha256'], 'source_path': str(path.resolve()),
                            'native_pair_available': row['native_atlas_pair_available'], 'visual_manifest_present': row['visual_manifest_present'],
                            'visual_context_coverage_status': row['visual_context_coverage_status'],
                            'animal_aggregation_exclusion_reason': row['animal_aggregation_exclusion_reason']})
    direct = pd.DataFrame(direct_rows)
    assert direct.ARMIndex.isin([228, 229, 728, 729, 41, 42, 43, 541, 542, 543]).sum() == 8
    assert sum(sample.potential_INS_candidate.eq('True')) == 29
    old = pd.read_excel(paths['older353'], sheet_name='Summary', dtype=str)
    current = pd.read_csv(paths['current462'], dtype=str, keep_default_na=False)
    added = current[current.SampleID.eq('251637') & ~current.NeuronUID.isin(old.NeuronUID)]
    assert added.NeuronUID.tolist() == ['251637::115.swc']
    row = review[review['sample'].eq('251637') & review.neuron_id.eq('115.swc')].iloc[0]
    own_path = ROOT / row.map_selected_source
    assert sha(own_path) == row.map_selected_sha256
    anchor = review[review.uid.eq(row.nearest_INS_anchor_uid)].iloc[0]
    own_root = np.loadtxt(own_path, comments='#', ndmin=2)
    anchor_graph = np.loadtxt(ROOT / anchor.map_selected_source, comments='#', ndmin=2)
    own_root = own_root[own_root[:, 6] == -1][0, 2:5]
    anchor_root = anchor_graph[anchor_graph[:, 6] == -1][0, 2:5]
    distance = float(np.linalg.norm(own_root - anchor_root) / 1000)
    # The priority table uses portal coordinate precision; SWCs store rounded
    # export values. Verify their agreement within the recorded sub-micron
    # precision difference, while reporting the two distances separately.
    assert np.isclose(distance, float(row.nearest_same_exact_sample_INS_anchor_mm), rtol=0, atol=3e-6)
    assert row.Henry_coarse_INS_visual_evidence == 'False' and row.own_atlas_id == '0'
    fields = ['sample', 'neuron_id', 'uid', 'registry_animal', 'screen_reasons', 'bbox_hits', 'Henry_coarse_INS_visual_evidence',
              'own_index_xyz', 'own_rounded_voxel_xyz', 'own_atlas_label', 'own_brainmask_value',
              'own_root_current_rint_zero_center_tissue_class', 'own_root_published_ceil_edge_one_based_to_zero_tissue_class',
              'own_root_published_ceil_edge_one_based_to_zero_label', 'nearest_INS_anchor_uid', 'nearest_same_exact_sample_INS_anchor_mm',
              'nearest_INS_anchor_evidence_basis', 'anchor_distance_status', 'native_atlas_pair_available', 'visual_manifest_present',
              'map_selected_source', 'map_selected_sha256', 'coarse_review_state']
    output.mkdir(parents=True)
    direct.to_csv(output / 'dataset250432_all234_source_inventory.csv', index=False)
    direct[direct.spatial_candidate_flag.eq('True')].to_csv(output / 'dataset250432_selected29_candidates.csv', index=False)
    pd.DataFrame([{field: row[field] for field in fields}]).to_csv(output / 'dataset251637_added115_evidence.csv', index=False)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'producer_sha256': sha(__file__),
              'sources': {name: {'path': str(p.resolve()), 'sha256': sha(p)} for name, p in paths.items()},
              'dataset250432': {'portal_sample_alias': 'ION2508004', 'listed_reconstructions': 234, 'metadata_tracing_count': 128,
                               'fresh_source_hash_and_direct_ARM_lookups': 234, 'atlas_insula': 8, 'spatial_screen_total_including_INS': 29,
                               'other_spatial_candidates': 21, 'atlas_INS_distribution': {'CR_granular_insula': 6, 'CL_agranular_and_dysgranular_insula': 2},
                               'registered_monkey_id': None, 'injection_region_metadata': 'empty',
                               'injection_structure_metadata': 'M1; F5; area 3a/b; claustrum; putamen; lateral dorsal amygdala; dorsal endopiriform; ventral endopiriform',
                               'injection_claim_validation': 'Metadata only; no explicit INS injection evidence recovered',
                               'nominal5um_copied_source': 'absent at checked root in dated October 8 snapshot; global availability unknown',
                               'pooling': 'excluded from animal means until a verified animal identity/independence record exists; sources retained'},
              'dataset251637_single_addition': {'NeuronUID': '251637::115.swc', 'classification': 'spatial candidate; not an additional accepted INS neuron',
                                               'direct_ARM_current': 'label 0', 'alternative_origin_policy_ARM': 'CR_agranular_and_dysgranular_insula',
                                               'Henry_evidence': False, 'bbox_hits': row.bbox_hits, 'other_Henry_anchor': row.nearest_INS_anchor_uid,
                                               'fresh_source_root_anchor_distance_mm': distance, 'original_candidate_axon_ends': 31,
                                               'saved_portal_coordinate_anchor_distance_mm': float(row.nearest_same_exact_sample_INS_anchor_mm),
                                               'native_pair_available': False, 'recorded_visual_manifest_present': False,
                                               'addition_meaning': 'New inclusion relative to older353 ledger; not proof of newly traced or newly reconstructed neuron'},
              'anatomical_acceptance': False,
              'artifacts': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'source_investigation_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('dataset')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
