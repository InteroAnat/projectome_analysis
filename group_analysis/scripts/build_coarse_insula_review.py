"""Read-only coarse visual-review and reproducible NMT graph source inventory.

No label corrections, registration acceptance, image downloads or canonical writes.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from insula_label_set import build_insula_label_set, normalize_label

POLICIES = ('current_rint_zero_center', 'published_ceil_edge_one_based_to_zero')
BASE = ROOT / 'group_analysis/evolution_20261008/atlas_locations/soma_audit_20261009'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def truth(value):
    return str(value).lower() == 'true'


def identity(row):
    return (row.get('sample', row.get('SampleID')), row.get('neuron_id', row.get('NeuronID')))


def numeric_id(nid):
    match = re.fullmatch(r'(\d+)\.swc', nid)
    return int(match[1]) if match else None


def validate_tissue_reference_geometry(segmentation, reference):
    if segmentation.shape != reference.shape or not np.array_equal(segmentation.affine, reference.affine):
        raise ValueError('Segmentation and reference must have identical shape and affine')


def tissue_labels(image):
    """Recover official class codes from embedded AFNI atlas table, not guesses."""
    for ext in image.header.extensions:
        content = ext.get_content()
        text = content.decode() if isinstance(content, bytes) else str(content)
        outer = ET.fromstring(text)
        for item in outer:
            if item.attrib.get('atr_name') == 'ATLAS_LABEL_TABLE':
                inner = ET.fromstring(item.text.strip().strip('"').strip())
                names = {int(x.attrib['VAL']): x.attrib['STRUCT'] for x in inner}
                if names:
                    return names, hashlib.sha256(text.encode()).hexdigest()
    raise ValueError('No official embedded segmentation class table')


def tissue_at(seg, voxel, names):
    if not voxel:
        return None, 'missing_unassessed'
    idx = np.asarray(json.loads(voxel) if isinstance(voxel, str) else voxel, dtype=int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(seg.shape)):
        return None, 'out_of_bounds'
    code = int(seg[tuple(idx)])
    return code, names.get(code, 'background_0' if code == 0 else f'unmapped_{code}')


def tissue_flag(a, b, names):
    ca, cb = names.get(a), names.get(b)
    if a is None or b is None:
        return 'missing_or_out_of_bounds_unassessed'
    if ca == cb == 'WM':
        return 'persistent_WM'
    if {ca, cb} & {'WM'} and {ca, cb} & {'GM', 'scGM'}:
        return 'origin_sensitive_GM_WM_transition'
    if ca == 'WM' or cb == 'WM':
        return 'origin_sensitive_WM_other'
    if ca == cb and ca in ('GM', 'scGM'):
        return 'persistent_' + ca
    return ca + '_to_' + cb if ca and cb else 'background_or_unmapped'


def coarse_state(human_ins, human_adjacent, tissue_status, candidate):
    if human_ins:
        return 'reviewed_INS'
    if human_adjacent:
        return 'reviewed_adjacent_cortex'
    if tissue_status in ('persistent_WM', 'origin_sensitive_GM_WM_transition', 'origin_sensitive_WM_other'):
        return 'WM_or_boundary_uncertain'
    return 'candidate-only' if candidate else 'unresolved'


def established_anchor_basis(row, insula):
    """Evidence for retrieval anchors, never alternative-policy candidates."""
    if truth(row.get('excluded')):
        return []
    basis = []
    manual = json.loads(row.get('original_manual_rows') or '[]')
    if truth(row.get('Henry_coarse_INS_visual_evidence')) or any(normalize_label(x.get('area', x.get('Area', ''))) in insula for x in manual):
        basis.append('Henry_coarse_INS_visual_annotation')
    if truth(row.get('atlas_INS')):
        basis.append('original_portal_INS_label')
    return basis


def established_anchors(rows, insula):
    return [{**r, 'anchor_evidence_basis': established_anchor_basis(r, insula)} for r in rows if established_anchor_basis(r, insula)]


def nearest_other_anchor(row, anchors):
    if not row.get('portal_coordinate_index_xyz'):
        return None, '', '', []
    xyz = np.asarray(json.loads(row['portal_coordinate_index_xyz']), dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None, '', '', []
    candidates = []
    for anchor in anchors:
        if identity(anchor)[0] != identity(row)[0] or identity(anchor) == identity(row) or not anchor.get('portal_coordinate_index_xyz'):
            continue
        other = np.asarray(json.loads(anchor['portal_coordinate_index_xyz']), dtype=float)
        if other.shape == (3,) and np.isfinite(other).all():
            candidates.append((float(np.linalg.norm(xyz - other) * .25), anchor))
    if not candidates:
        return None, '', '', []
    distance, anchor = min(candidates, key=lambda x: (x[0], x[1]['neuron_id']))
    return distance, anchor['neuron_id'], anchor['sample'] + '|' + anchor['neuron_id'], anchor['anchor_evidence_basis']


def corrected_priority_fields(row, anchors):
    distance, nearest, uid, basis = nearest_other_anchor(row, anchors)
    n = numeric_id(row['neuron_id'])
    neighbors = [a for a in anchors if a['sample'] == row['sample'] and n is not None and numeric_id(a['neuron_id']) is not None and 0 < abs(n - numeric_id(a['neuron_id'])) <= 2]
    labels = [row.get('portal_coordinate_' + p + '_label', '') for p in POLICIES]
    source_base = normalize_label(row.get('portal_region', ''))
    uncertain = not source_base or source_base.startswith('UNKNOWN') or source_base == 'PRCO' or any(x == 'Unknown_0' or normalize_label(x) == 'PRCO' for x in labels)
    sensitivity = any(truth(row.get('portal_coordinate_' + p + '_insula_candidate')) for p in POLICIES)
    human = truth(row.get('Henry_coarse_INS_visual_evidence'))
    candidate = truth(row.get('potential_INS_candidate')) or sensitivity or human or (uncertain and ((distance is not None and distance <= 2) or bool(neighbors)))
    assets = json.loads(row.get('existing_visual_assets') or '{}')
    tissue = row.get('portal_coordinate_tissue_policy_status', '')
    score = (100 if uncertain and candidate else 0) + (50 if sensitivity and uncertain else 0) + (25 if assets and candidate else 0) + (15 if tissue in ('persistent_WM', 'origin_sensitive_GM_WM_transition') and candidate else 0) + (10 if neighbors and uncertain else 0)
    return {'nearest_same_exact_sample_INS_anchor_mm': distance, 'nearest_INS_anchor_id': nearest,
            'nearest_INS_anchor_uid': uid, 'nearest_INS_anchor_evidence_basis': basis,
            'anchor_distance_status': 'other_established_anchor_found' if uid else 'missing_coordinate_or_no_other_established_anchor',
            'neighboring_numeric_INS_ID_retrieval_cues': [a['neuron_id'] for a in neighbors],
            'numeric_neighbor_anchor_evidence': [{'uid': a['sample'] + '|' + a['neuron_id'], 'basis': a['anchor_evidence_basis']} for a in neighbors],
            'ID_adjacency_anatomical_evidence': False, 'alternative_policy_INS_sensitivity_candidate': sensitivity,
            'alternative_policy_candidate_used_as_established_anchor': False,
            'review_candidate': candidate, 'priority_score': score,
            'coarse_review_state': coarse_state(human, False, tissue, candidate)}


def distance_priority_supplement(input_dir, out):
    """Correct retrieval fields from immutable prior rows without reading graphs."""
    input_dir, out = Path(input_dir).resolve(), Path(out).resolve()
    out.relative_to(ROOT / 'group_analysis/evolution_20261008/classification')
    if out.exists():
        raise FileExistsError(out)
    delivery_path = input_dir / 'delivery_provenance_20261009.json'
    delivery = json.loads(delivery_path.read_text())
    protected = {name: digest(input_dir / name) for name in ('all_neuron_review_manifest.csv', 'priority_review_manifest.csv', 'map_ready_sources.csv', 'provenance.json')}
    for name, value in protected.items():
        if delivery['artifacts'][name] != value:
            raise ValueError('Prior delivery hash mismatch: ' + name)
    rows = read_csv(input_dir / 'all_neuron_review_manifest.csv')
    insula, _ = build_insula_label_set(str(ROOT / 'atlas/ARM_key_all.txt'))
    anchors = established_anchors(rows, insula)
    corrected, changes = [], Counter()
    for row in rows:
        fields = corrected_priority_fields(row, anchors)
        result = {**row, **fields, 'prior_nearest_INS_distance_mm': row['nearest_same_exact_sample_INS_anchor_mm'],
                  'prior_nearest_INS_anchor_id': row['nearest_INS_anchor_id'], 'prior_priority_score': row['priority_score'],
                  'prior_review_candidate': row['review_candidate'], 'prior_row_source_sha256': protected['all_neuron_review_manifest.csv']}
        result['review_queue_membership_changed'] = fields['review_candidate'] != truth(row['review_candidate'])
        if result['review_queue_membership_changed']:
            changes['added' if fields['review_candidate'] else 'removed'] += 1
        if row['nearest_INS_anchor_id'] == row['neuron_id']:
            changes['prior_self_anchor_rows'] += 1
        corrected.append(result)
    priority = sorted([r for r in corrected if r['review_candidate']], key=lambda r: (-r['priority_score'], r['nearest_same_exact_sample_INS_anchor_mm'] if r['nearest_same_exact_sample_INS_anchor_mm'] is not None else float('inf'), r['sample'], r['neuron_id']))
    for name, value in protected.items():
        if digest(input_dir / name) != value:
            raise RuntimeError('Protected input changed')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'all_neuron_review_manifest_distance_corrected.csv', corrected)
    write_csv(out / 'priority_review_manifest_distance_corrected.csv', priority)
    report = {'observed_at': datetime.now().astimezone().isoformat(), 'generator_sha256': digest(Path(__file__)),
              'input_delivery': {'path': delivery_path.relative_to(ROOT).as_posix(), 'sha256': digest(delivery_path)},
              'protected_prior_file_sha256': protected, 'exact_identity_rows': len(corrected),
              'established_anchor_count': len(anchors), 'prior_queue_rows': sum(truth(r['review_candidate']) for r in rows),
              'corrected_queue_rows': len(priority), 'changes': dict(changes),
              'map_membership_changed': False, 'map_group_flags_changed': False,
              'anchor_policy': 'Nearest OTHER exact UID in same sample/channel; only exact Henry coarse INS visual rows or original portal INS labels. Alternative-policy INS is sensitivity only, never anchor or neighbor-anchor basis.',
              'coordinate_metric': 'Stored portal XYZ/250 indices times0.25mm; relative retrieval distance, not accepted native registration.',
              'source_graph_recomputations': 0, 'network_requests': 0, 'prior_delivery_writes': False,
              'scientific_acceptance': False,
              'outputs': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    with (out / 'correction_provenance.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with (out / 'README.md').open('x', encoding='utf-8') as f:
        f.write('# Corrected coarse review distance supplement\n\nThe prior nearest-anchor field admitted alternative-origin candidates and could select the same neuron, creating circular zero distances. This supplement requires another exact UID and uses only Henry coarse INS visual annotations or original portal INS labels as retrieval anchors. Alternative-origin candidates remain sensitivity fields; numeric-neighbor cues use the same established anchor set and remain retrieval cues only.\n\nOriginal hash-bound delivery and map inputs are preserved. Only review distances, cues, priorities and queue membership are recalculated. Existing graph verification is retained as prior hash-bound evidence; no graph computations were rerun. Original portal labels, manual labels, tissue policies and map evidence groups are carried forward unchanged. See correction_provenance.json for queue changes.\n')
    print(json.dumps(report), flush=True)


def graph(path):
    a = np.loadtxt(path, comments='#', ndmin=2)
    if a.shape[1] != 7 or not len(a) or not np.isfinite(a).all():
        raise ValueError('SWC must contain finite seven-column nodes')
    a = a[np.argsort(a[:, 0])]
    ids, parents = a[:, 0], a[:, 6]
    if not np.equal(ids, np.floor(ids)).all() or not np.equal(parents, np.floor(parents)).all() or len(np.unique(ids)) != len(a):
        raise ValueError('Invalid or duplicate node IDs')
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1 or (a[:, 5] < 0).any():
        raise ValueError('Invalid root count or radius')
    lookup = {int(v): i for i, v in enumerate(ids)}
    children = [[] for _ in ids]
    for i, p in enumerate(parents):
        if p == -1:
            continue
        if int(p) not in lookup:
            raise ValueError('Missing graph parent')
        children[lookup[int(p)]].append(i)
    seen, stack = set(), [int(roots[0])]
    while stack:
        i = stack.pop()
        if i in seen:
            raise ValueError('Graph cycle')
        seen.add(i)
        stack.extend(children[i])
    if len(seen) != len(a):
        raise ValueError('Disconnected or cyclic graph')
    return a


def graph_equivalent(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--distance-supplement-from', type=Path)
    args = ap.parse_args()
    if args.distance_supplement_from:
        distance_priority_supplement(args.distance_supplement_from, args.out_dir)
        return
    out = args.out_dir.resolve()
    out.relative_to(ROOT / 'group_analysis/evolution_20261008/classification')
    out.mkdir(parents=True, exist_ok=True)
    targets = ('all_neuron_review_manifest.csv', 'priority_review_manifest.csv', 'map_ready_sources.csv', 'graph_source_readback.csv', 'provenance.json', 'README.md')
    if any((out / x).exists() for x in targets):
        raise FileExistsError('Output files already exist; choose a distinct version directory')
    paths = {'atlas_audit': BASE / 'validated_atlas_lookup/neuron_soma_locations.csv',
             'sources': BASE / 'validated_atlas_lookup/source_inventory.csv',
             'policies': BASE / 'coordinate_origin_sensitivity/per_neuron_origin_sensitivity.csv',
             'visual_table': ROOT / 'group_analysis/visual_review_20261002/tables/correction_table.csv',
             'raw_graph_readback': ROOT / 'notes/bulk_visual_review_20261002/all_raw_swc_graph_readback_20261004.json',
             'Henry': ROOT / 'R_analysis/tables/somainfo_Henry_2026.04.03.xlsx',
             'animal_registry': ROOT / 'group_analysis/docs/dataset_status_manifest.csv',
             'segmentation': ROOT / 'atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_segmentation.nii.gz',
             'reference': ROOT / 'atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz',
             'label_key': ROOT / 'atlas/ARM_key_all.txt'}
    rows = read_csv(paths['atlas_audit'])
    assert len(rows) == 8746 and len({identity(r) for r in rows}) == 8746
    policies = {identity(r): r for r in read_csv(paths['policies'])}
    visual = {identity(r): r for r in read_csv(paths['visual_table'])}
    raw_records = {r['uid']: r for r in json.loads(paths['raw_graph_readback'].read_text())['records']}
    sources = defaultdict(list)
    for source in read_csv(paths['sources']):
        sources[identity(source)].append(source)
    insula, _ = build_insula_label_set(str(paths['label_key']))
    registry = {r['fmost_id']: r for r in read_csv(paths['animal_registry']) if r.get('fmost_id')}
    seg_img, ref_img = nib.load(paths['segmentation']), nib.load(paths['reference'])
    validate_tissue_reference_geometry(seg_img, ref_img)
    names, label_hash = tissue_labels(seg_img)
    seg = np.asanyarray(seg_img.dataobj)
    by_sample = defaultdict(list)
    manual = {}
    for row in rows:
        annotations = json.loads(row.get('original_manual_rows') or '[]')
        manual[identity(row)] = annotations
        by_sample[row['sample']].append(row)
    anchors = {}
    for sample, sample_rows in by_sample.items():
        anchors[sample] = established_anchors(sample_rows, insula)
    graph_results, maps, output = [], [], []
    for num, row in enumerate(rows):
        key, sample, nid = identity(row), row['sample'], row['neuron_id']
        policy, vr = policies[key], visual.get(key, {})
        result = dict(row)
        result.update({k: v for k, v in policy.items() if k.startswith('own_root_') or k.startswith('portal_coordinate_')})
        result['Henry_annotation_rows'] = manual[key]
        human_ins = any(normalize_label(x.get('area', x.get('Area', ''))) in insula for x in manual[key])
        reviewed = normalize_label(row.get('reviewed_region', ''))
        human_ins = human_ins or (reviewed in insula and truth(row.get('reviewed_source_hash_verified')))
        result['Henry_coarse_INS_visual_evidence'] = human_ins and bool(manual[key])
        result['human_fine_label_status'] = 'historical_visual_annotation_not_histological_parcel_validation' if manual[key] else row.get('reviewed_status', 'unreviewed')
        selected = anchors[sample]
        distance, nearest, anchor_uid, anchor_basis = nearest_other_anchor(row, selected)
        n = numeric_id(nid)
        adjacent_ids = [r['neuron_id'] for r in selected if n is not None and numeric_id(r['neuron_id']) is not None and 0 < abs(n - numeric_id(r['neuron_id'])) <= 2]
        result.update(nearest_same_exact_sample_INS_anchor_mm=distance, nearest_INS_anchor_id=nearest,
                      nearest_INS_anchor_uid=anchor_uid, nearest_INS_anchor_evidence_basis=anchor_basis,
                      neighboring_numeric_INS_ID_retrieval_cues=adjacent_ids,
                      ID_adjacency_anatomical_evidence=False)
        origin_labels = [policy.get('portal_coordinate_' + p + '_label', '') for p in POLICIES]
        source_base = normalize_label(row.get('portal_region', ''))
        uncertain_label = not source_base or source_base.startswith('UNKNOWN') or source_base == 'PRCO' or any(x == 'Unknown_0' or normalize_label(x) == 'PRCO' for x in origin_labels)
        candidate = truth(row['potential_INS_candidate']) or any(truth(policy.get('portal_coordinate_' + p + '_insula_candidate')) for p in POLICIES) or human_ins or (uncertain_label and ((distance is not None and distance <= 2) or bool(adjacent_ids)))
        result['review_candidate'] = candidate
        for scope in ('own_root', 'portal_coordinate'):
            codes = []
            for p in POLICIES:
                code, label = tissue_at(seg, policy.get(scope + '_' + p + '_voxel'), names)
                result[scope + '_' + p + '_tissue_code'] = code
                result[scope + '_' + p + '_tissue_class'] = label
                codes.append(code)
            result[scope + '_tissue_policy_status'] = tissue_flag(*codes, names)
            result[scope + '_atlas_background_with_GM'] = any(origin_labels[i] == 'Unknown_0' and names.get(codes[i]) in ('GM', 'scGM') for i in range(2)) if scope == 'portal_coordinate' else any(policy.get(scope + '_' + POLICIES[i] + '_label') == 'Unknown_0' and names.get(codes[i]) in ('GM', 'scGM') for i in range(2))
        result['coarse_review_state'] = coarse_state(human_ins, False, result['portal_coordinate_tissue_policy_status'], candidate)
        result['fine_label_original_preserved'] = True
        result['automatic_correction'] = False
        result['visual_manifest_present'] = bool(vr)
        result['visual_channel'] = vr.get('channel', '')
        result['visual_channel_note'] = vr.get('channel_note', '')
        for field in ('production_status', 'coverage_status', 'context_coverage_status', 'context_pending_reason', 'context_coverage_fraction', 'context_loaded_z', 'context_missing_z', 'soma_missing', 'human_region', 'human_subregion', 'human_layer', 'reviewer', 'review_date', 'human_notes'):
            result['visual_' + field] = vr.get(field, '')
        assets = {}
        for field in ('soma_marked.png', 'soma_ortho.png', 'context_marked.png', 'context_coverage.png', 'section_locator.png'):
            path = ROOT / 'group_analysis/visual_review_20261002' / sample / Path(nid).stem / field
            if path.is_file():
                assets[field] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}
        prov = ROOT / 'group_analysis/visual_review_20261002' / sample / Path(nid).stem / 'provenance.json'
        if prov.is_file():
            assets['provenance.json'] = {'path': prov.relative_to(ROOT).as_posix(), 'sha256': digest(prov)}
        result['existing_visual_assets'] = assets
        result['source_error_details_preserved_in'] = 'group_analysis/visual_review_20261002/tables/correction_table.csv' if vr.get('context_source_errors') else ''
        result['known_visual_QC_warning'] = 'Historical source blank area/seams: unsuitable standalone anatomical proof; notes/bulk_visual_review_20261002/completion_audit_20261003.md' if key == ('252384', '003.swc') else ''
        raw = ROOT / 'group_analysis/visual_review_20261002/swc_raw' / sample / nid
        rr = raw_records.get(sample + '|' + nid, {})
        result['native_raw_swc_path'] = raw.relative_to(ROOT).as_posix() if raw.is_file() else ''
        result['native_raw_swc_sha256'] = digest(raw) if raw.is_file() else ''
        raw_verified = bool(rr.get('raw_sha256')) and rr['raw_sha256'] == result['native_raw_swc_sha256']
        result['native_root_xyz_um'] = rr.get('root_xyz_um') if raw_verified else None
        result['native_graph_status'] = 'prior_full_graph_read_exact_hash_reverified' if raw_verified and rr.get('outcome') == 'graph_valid' else 'missing_or_unassessed'
        case_native = {'112.swc': 'raw_112.swc', '114.swc': 'raw_114.swc', '111.swc': 'control_raw_111.swc', '422.swc': 'control_raw_422.swc'}
        if not raw.is_file() and sample == '251637' and nid in case_native:
            # Native role/identity are explicitly documented in the case readback
            # and same-sample control receipts, independently of these filenames.
            native_case = ROOT / 'notes/region_analysis_review_20261004/case_112_114_20261004'
            case_path = native_case / case_native[nid]
            receipt = native_case / ('same_sample_native_controls.json' if nid in ('111.swc', '422.swc') else 'source_readback.json')
            if receipt.is_file() and case_path.is_file():
                a = graph(case_path)
                raw = case_path
                result['native_raw_swc_path'] = raw.relative_to(ROOT).as_posix()
                result['native_raw_swc_sha256'] = digest(raw)
                result['native_root_xyz_um'] = a[a[:, 6] == -1][0, 2:5].tolist()
                result['native_graph_status'] = 'fresh_full_graph_read_explicit_case_native_role_and_identity_receipt'
                result['native_source_identity_receipt'] = {'path': receipt.relative_to(ROOT).as_posix(), 'sha256': digest(receipt)}
        result['native_atlas_pair_available'] = raw.is_file() and bool(sources[key])
        result['native_to_NMT_transform_acceptance'] = 'pending_original_transform_receipts'
        ss = sorted(sources[key], key=lambda s: s['source_path'])
        selected_source, graph_status, canonical, all_valid = None, 'missing_unassessed', None, True
        by_hash = {}
        for source in ss:
            path = ROOT / source['source_path']
            record = {'sample': sample, 'neuron_id': nid, 'path': source['source_path'], 'expected_sha256': source['sha256']}
            try:
                record['sha256'] = digest(path)
                if record['sha256'] != source['sha256']:
                    raise ValueError('Source hash changed')
                if record['sha256'] not in by_hash:
                    by_hash[record['sha256']] = graph(path)
                a = by_hash[record['sha256']]
                record['node_count'] = len(a)
                record['numeric_graph_sha256'] = hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
                if canonical is None:
                    canonical, selected_source = a, source
                if not graph_equivalent(a, canonical):
                    raise ValueError('Different numeric node graph; no source precedence established')
                record['status'] = 'graph_valid_equal_numeric_nodes'
            except Exception as exc:
                record['status'], record['error'] = 'unassessed_or_conflicting', str(exc)
                all_valid = False
            graph_results.append(record)
        if ss:
            graph_status = 'numeric_graph_verified_all_copies' if all_valid else 'graph_ambiguous_or_invalid_excluded'
        result['map_source_graph_status'] = graph_status
        group_base = 'HumanINS' if human_ins else 'AtlasINS' if truth(row['atlas_INS']) else 'Candidate' if truth(row['potential_INS_candidate']) else 'G_exploratory' if truth(row['atlas_G']) else 'Other'
        result['exclusive_map_group'] = group_base + '_' + (row.get('own_mask_side') or row.get('portal_coordinate_mask_side') or 'Unknown')
        result['registry_animal'] = registry.get(sample, {}).get('animal', '')
        result['animal_mapping_status'] = 'exact_registry_sample_match' if result['registry_animal'] else 'missing_registry_animal_mapping'
        result['animal_aggregation_eligible'] = bool(result['registry_animal'])
        result['animal_aggregation_exclusion_reason'] = '' if result['registry_animal'] else 'Exact sample/channel absent from animal registry; no animal ID inferred'
        result['map_ready'] = bool(ss) and all_valid and not truth(row['excluded'])
        result['map_selected_source'] = selected_source['source_path'] if result['map_ready'] else ''
        result['map_selected_sha256'] = selected_source['sha256'] if result['map_ready'] else ''
        result['map_source_selection_rule'] = 'lexicographically_first_path_after_all_distinct_byte_copies_verified_exact_equal_sorted_seven_column_numeric_graphs' if result['map_ready'] else ''
        result['priority_score'] = (100 if uncertain_label and candidate else 0) + (50 if any(truth(policy.get('portal_coordinate_' + p + '_insula_candidate')) for p in POLICIES) and uncertain_label else 0) + (25 if assets and candidate else 0) + (15 if result['portal_coordinate_tissue_policy_status'] in ('persistent_WM', 'origin_sensitive_GM_WM_transition') and candidate else 0) + (10 if adjacent_ids and uncertain_label else 0)
        if result['map_ready']:
            maps.append({k: result[k] for k in ('sample', 'neuron_id', 'uid', 'portal_region', 'atlas_INS', 'atlas_G', 'potential_INS_candidate', 'excluded', 'Henry_coarse_INS_visual_evidence', 'Henry_annotation_rows', 'reviewed_region', 'reviewed_status', 'coarse_review_state', 'exclusive_map_group', 'registry_animal', 'animal_mapping_status', 'animal_aggregation_eligible', 'animal_aggregation_exclusion_reason', 'map_selected_source', 'map_selected_sha256', 'map_source_selection_rule', 'map_source_graph_status', 'own_root_xyz_um', 'own_atlas_label', 'own_mask_side', 'portal_coordinate_tissue_policy_status', 'native_atlas_pair_available')} | {'graph_node_count': len(canonical), 'coordinate_encoding': 'XYZ/250=NMT index; origin unresolved; compare current rint and published ceil-1', 'scientific_registration_acceptance': 'pending'})
        output.append(result)
        if num % 500 == 0:
            print(json.dumps({'processed': num, 'map_sources': len(maps)}), flush=True)
    priority = sorted([r for r in output if r['review_candidate']], key=lambda r: (-r['priority_score'], r['nearest_same_exact_sample_INS_anchor_mm'] if r['nearest_same_exact_sample_INS_anchor_mm'] is not None else float('inf'), r['sample'], r['neuron_id']))
    write_csv(out / targets[0], output)
    write_csv(out / targets[1], priority)
    write_csv(out / targets[2], maps)
    write_csv(out / targets[3], graph_results)
    summary = {'observed_at': datetime.now().astimezone().isoformat(), 'inputs': {k: {'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for k, p in paths.items()},
               'generator': {'path': Path(__file__).relative_to(ROOT).as_posix(), 'sha256': digest(Path(__file__))},
               'official_segmentation_classes': names, 'embedded_label_table_sha256': label_hash,
               'exact_neurons': len(output), 'review_queue': len(priority), 'map_ready': len(maps), 'source_file_graph_readbacks': len(graph_results),
               'graph_status_counts': dict(Counter(r['map_source_graph_status'] for r in output)),
               'coarse_states': dict(Counter(r['coarse_review_state'] for r in output)),
               'tissue_policy_counts': dict(Counter(r['portal_coordinate_tissue_policy_status'] for r in output)),
               'map_sources_by_exact_sample': dict(Counter(r['sample'] for r in maps)),
               'exclusive_map_groups': dict(Counter(r['exclusive_map_group'] for r in maps)),
               'registry_eligible_exclusive_groups': dict(Counter(r['exclusive_map_group'] for r in maps if r['animal_aggregation_eligible'])),
               'map_sources_by_evidence': {k: sum(truth(r[k]) for r in maps) for k in ('atlas_INS', 'atlas_G', 'potential_INS_candidate', 'Henry_coarse_INS_visual_evidence')},
               'network_requests': 0, 'canonical_edits': False, 'anatomical_acceptance': False,
               'review_rules': {'radius_mm': 2, 'numeric_ID_window': 2, 'ID_cue_is_anatomy': False, 'same_sample_and_channel_only': True, 'Henry_visual_origin': 'user confirmed historical wide-field visual checks; exact workbook annotation rows only'},
               'limitations': ['Tissue segmentation is NMT-space software evidence, not native white-matter or registration acceptance.', 'Portal labels, both origin-policy lookups, Henry visual annotations and reviewed fine labels remain separate.', 'Historical 832 machine review categories are not human decisions. Missing/partial image sources remain explicit.', 'Only available cached NMT graphs map-ready; all 8746 identities retained in full manifest.']}
    with (out / targets[4]).open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with (out / targets[5]).open('x', encoding='utf-8') as f:
        f.write('# Coarse INS review and map source inventory\n\nThe priority CSV is a retrieval queue, not corrected anatomy. `reviewed_INS` uses exact Henry visual annotations or hash-bound case decisions. Fine labels remain original and provisional where applicable. Adjacent numeric IDs are retrieval cues only.\n\nOfficial segmentation classes come from the NIfTI embedded AFNI table: 1 CSF, 2 GM, 3 scGM, 4 WM, 5 BV. Atlas background is not WM. Both current zero-center and literal published edge-origin policies are retained; the export convention and original transforms remain unverified.\n\nMap sources require valid connected rooted seven-column graphs and exact numeric equality of every cached copy after sorting node IDs. Selection is deterministic only after equality, preserving hashes and all source paths in graph readback. No image/network downloads, relabeling, canonical writes or anatomical acceptance. See provenance.json for exact counts and input hashes.\n')
    print(json.dumps({k: summary[k] for k in ('exact_neurons', 'review_queue', 'map_ready', 'graph_status_counts', 'map_sources_by_exact_sample', 'map_sources_by_evidence')}), flush=True)


if __name__ == '__main__':
    main()
