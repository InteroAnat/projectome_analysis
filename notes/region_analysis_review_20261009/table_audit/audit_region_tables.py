"""Read-only region-table audit; only new audit artifacts may be written."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
EXTENSIONS = {'.xlsx', '.csv', '.tsv', '.parquet'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, default=str)


def csvout(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()})


def text(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip()


def ids(df):
    for c in ('NeuronUID', 'neuron_uid', 'uid', 'UID'):
        if c in df:
            return [c]
    neuron = next((c for c in ('NeuronID', 'neuron_id') if c in df), None)
    sample = next((c for c in ('SampleID', 'sample', 'sample_id', 'fmost_id') if c in df), None)
    return ([sample] if sample else []) + [neuron] if neuron else []


def role(path):
    p = rel(path)
    rules = [('rollback', 'immutable_rollback_snapshot'), ('somainfo_Henry', 'immutable_human_visual_annotation_source'),
             ('group_analysis/combined/', 'protected_canonical_legacy_derivative'), ('/processed_neurons/', 'per_node_coordinate_derivative'),
             ('/staging_', 'candidate_staging_derivative'), ('/recovery/', 'candidate_recovery_derivative'),
             ('/projection_inputs/', 'explicit_versioned_map_source_manifest'), ('/projection_maps/', 'descriptive_map_QC_or_metric'),
             ('/endpoint_maps/', 'descriptive_map_QC_or_metric'), ('/step1_results/', 'historical_step1_derivative'),
             ('neuron_tables', 'historical_region_derivative_or_template'), ('/portal_audit_', 'metadata_inventory_or_QC'),
             ('/inventory/', 'metadata_inventory_or_QC'), ('/atlas_locations/', 'atlas_location_candidate_QC'),
             ('/classification/', 'candidate_visual_review_derivative'), ('/reference/', 'candidate_spatial_rule_reference'),
             ('/fnt/', 'FNT_candidate_metric_or_manifest'), ('notes/region_analysis_review', 'isolated_region_rerun_or_review_QC')]
    return next((value for key, value in rules if key in p), 'research_table_requires_context')


def inspect_frame(df, path, sheet):
    df.columns = [text(c) for c in df.columns]
    issues, keys = [], ids(df)
    schema = {'sheet': sheet, 'rows': len(df), 'columns': list(df.columns), 'identity_columns': keys,
              'missing_per_column': {c: int(df[c].map(text).eq('').sum()) for c in df}, 'numeric_metrics': {}}
    longform = sheet in ('Terminal_Sites', 'Outliers') or any(c in df for c in ('Target', 'target', 'target_id', 'target_index', 'endpoint_id', 'node_id', 'source_path', 'source', 'path'))
    schema['identity_grain'] = 'longform_or_source_record' if longform else 'neuron_or_manifest' if keys else 'other'
    if keys:
        kk = df[keys].apply(lambda c: c.map(text))
        missing = kk.eq('').any(axis=1)
        dup = kk.loc[~missing].duplicated(keep=False)
        schema.update(missing_identity_rows=int(missing.sum()), duplicate_identity_rows=int(dup.sum()))
        if dup.any() and not longform:
            issues.append({'check': 'duplicate_identity', 'status': 'requires_declared_grain_context', 'count': int(dup.sum()), 'examples': kk.loc[~missing].loc[dup].head(20).to_dict('records')})
    for sample, neuron, uid in [('SampleID', 'NeuronID', 'NeuronUID'), ('sample', 'neuron_id', 'uid')]:
        if {sample, neuron, uid} <= set(df):
            expected = df[sample].map(text) + '|' + df[neuron].map(text)
            actual = df[uid].map(text)
            # These are explicit producer formats, not interchangeable scientific identities.
            formats = [expected, expected.str.replace('|', ':', regex=False),
                       expected.str.replace('|', '::', regex=False)]
            if uid == 'NeuronUID' and 'fnt' in rel(path).lower():
                formats.append(df[sample].map(text) + '_' + df[neuron].map(text).str.removesuffix('.swc'))
            bad = ~pd.concat([actual.eq(x) for x in formats], axis=1).any(axis=1)
            if bad.any():
                issues.append({'check': 'composite_UID_mismatch', 'status': 'verified_identity_issue', 'count': int(bad.sum()), 'examples': df.loc[bad, [sample, neuron, uid]].head(15).to_dict('records')})
    if {'Soma_Region', 'Soma_Side'} <= set(df):
        extracted = df.Soma_Region.map(text).str.extract(r'^(?:C|S)([LR])_|^([LR])-', expand=True)
        prefix = extracted.iloc[:, 0].where(extracted.iloc[:, 0].notna(), extracted.iloc[:, 1])
        side = df.Soma_Side.map(text)
        bad = prefix.notna() & side.isin(['L', 'R']) & side.ne(prefix)
        if bad.any():
            issues.append({'check': 'region_prefix_vs_stored_side', 'status': 'preserved_label_conflict_not_anatomical_verdict', 'count': int(bad.sum()), 'examples': df.loc[bad, list(dict.fromkeys(keys + ['Soma_Region', 'Soma_Side']))].head(30).to_dict('records')})
    for c in df:
        if not re.search(r'(Length|_Count$|^N_|_mm$|_um$|^Soma_NII_|^Soma_Phys_|^Terminal_Count$)', c) or any(s in c for s in ('source', 'Source', 'xyz', 'XYZ')):
            continue
        raw = df[c].map(text)
        present = raw.ne('') & ~raw.str.startswith(('{', '[', '('))
        values = pd.to_numeric(raw.where(present), errors='coerce')
        finite = values[np.isfinite(values)]
        schema['numeric_metrics'][c] = {'scalar_nonempty': int(present.sum()), 'nonnumeric': int((present & values.isna()).sum()), 'min': float(finite.min()) if len(finite) else None, 'max': float(finite.max()) if len(finite) else None}
        if ('Length' in c or '_Count' in c or c.startswith('N_')) and (finite < 0).any():
            issues.append({'check': 'negative_length_or_count', 'status': 'verified_numeric_issue', 'column': c, 'count': int((finite < 0).sum())})
    count_cols = ['Terminal_Count', 'N_Ipsilateral', 'N_Contralateral', 'N_Laterality_Unknown']
    if set(count_cols) <= set(df):
        counts = df[count_cols].apply(pd.to_numeric, errors='coerce')
        bad = counts.notna().all(axis=1) & counts.iloc[:, 0].ne(counts.iloc[:, 1:].sum(axis=1))
        if bad.any():
            issues.append({'check': 'terminal_laterality_count_conservation', 'status': 'verified_numeric_issue', 'count': int(bad.sum()), 'examples': df.loc[bad, keys + count_cols].head(20).to_dict('records')})
    for item in issues:
        item.update(file=rel(path), sheet=sheet)
    return schema, issues


def workbook(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames, schemas, issues, strengths, pis = {}, [], [], [], []
    for ws in wb:
        iterator = ws.iter_rows(values_only=True)
        first = next(iterator, ())
        headers = [text(c) or f'__unnamed_{i+1}' for i, c in enumerate(first)]
        data = list(iterator)
        if len(headers) != len(set(headers)):
            issues.append({'file': rel(path), 'sheet': ws.title, 'check': 'duplicate_headers', 'status': 'verified_schema_issue', 'headers': headers})
            headers = [f'{c}__duplicate_{i}' if c in headers[:i] else c for i, c in enumerate(headers)]
        df = pd.DataFrame(data, columns=headers)
        frames[ws.title] = df
        schema, found = inspect_frame(df, path, ws.title)
        schemas.append(schema)
        issues += found
    wb.close()
    summary = frames.get('Summary')
    if summary is not None and 'NeuronID' in summary:
        expected = set(summary.NeuronID.map(text))
        for name, df in frames.items():
            if name.startswith(('Projection_Length', 'Projection_Strength', 'Laterality', 'Soma_Hierarchy')) and 'NeuronID' in df:
                observed = set(df.NeuronID.map(text))
                if observed != expected:
                    issues.append({'file': rel(path), 'sheet': name, 'check': 'cross_sheet_membership', 'status': 'verified_membership_issue', 'missing': sorted(expected-observed), 'extra': sorted(observed-expected)})
    for name, length in frames.items():
        if not name.startswith('Projection_Length'):
            continue
        targets = [c for c in length if c not in ('NeuronID', 'Neuron_Type', 'SampleID', 'NeuronUID')]
        for c in ('Pi', 'CL_Pi', 'CR_Pi', 'SL_Pi', 'SR_Pi'):
            if c in length:
                values = pd.to_numeric(length[c], errors='coerce').fillna(0)
                positive = values.gt(0)
                pis.append({'file': rel(path), 'sheet': name, 'target_column': c, 'domain': 'ambiguous_stripped_Pi' if c == 'Pi' else 'parainsula' if c.startswith('C') else 'pineal', 'positive_neuron_rows': int(positive.sum()), 'sum_stored_length_units': float(values.sum()), 'positive_IDs': length.loc[positive, ids(length)].to_dict('records') if ids(length) else []})
        strength_name = name.replace('Projection_Length', 'Projection_Strength', 1)
        if strength_name not in frames:
            continue
        strength = frames[strength_name]
        keycols = [c for c in ('SampleID', 'NeuronID') if c in length and c in strength]
        if len(length) != len(strength) or not keycols or any(not length[c].map(text).equals(strength[c].map(text)) for c in keycols):
            issues.append({'file': rel(path), 'sheet': strength_name, 'check': 'length_strength_identity_alignment', 'status': 'verified_identity_issue'})
            continue
        targets = [c for c in targets if c in strength]
        a = length[targets].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        b = strength[targets].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        expected = np.round(np.log10(a + 1), 4)
        bad = np.isfinite(a) & np.isfinite(b) & (np.abs(expected-b) > 5.001e-5)
        strengths.append({'file': rel(path), 'length_sheet': name, 'strength_sheet': strength_name, 'compared_cells': int(np.isfinite(a).sum()), 'mismatched_cells': int(bad.sum()), 'max_abs_error': float(np.nanmax(np.abs(expected-b))) if a.size else None, 'Pi_mismatches': int(bad[:, targets.index('Pi')].sum()) if 'Pi' in targets else 0})
        if bad.any():
            examples = [{'row_1based': int(i+2), 'NeuronID': text(length.iloc[i].get('NeuronID')), 'target': targets[j], 'length': float(a[i, j]), 'strength': float(b[i, j]), 'expected': float(expected[i, j])} for i, j in np.argwhere(bad)[:50]]
            issues.append({'file': rel(path), 'sheet': strength_name, 'check': 'strength_vs_round_log10_length_plus1', 'status': 'verified_export_numeric_issue', 'count': int(bad.sum()), 'examples': examples})
    with zipfile.ZipFile(path) as archive:
        formulas = sum(archive.read(n).count(b'<f') for n in archive.namelist() if n.startswith('xl/worksheets/') and n.endswith('.xml'))
    return {'sheets': schemas, 'formula_xml_elements': formulas, 'formula_evaluation': 'cached_values_only_no_recalculation'}, issues, strengths, pis


def delimited(path):
    count, cols, findings, checks = 0, [], [], []
    node_ids, parent_ids, roots, duplicates = set(), set(), 0, 0
    is_node = False
    for df in pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',', dtype=str, keep_default_na=False, chunksize=30000):
        count += len(df)
        cols = list(df.columns)
        is_node = {'id', 'parent', 'x', 'y', 'z'} <= set(df)
        if not is_node:
            schema, found = inspect_frame(df, path, '__csv__')
            checks.append(schema)
            findings += found
            continue
        node = pd.to_numeric(df.id, errors='coerce')
        parent = pd.to_numeric(df.parent, errors='coerce')
        roots += int(parent.eq(-1).sum())
        duplicates += int(node.duplicated().sum()) + len(node_ids.intersection(node.dropna().tolist()))
        node_ids.update(node.dropna().tolist())
        parent_ids.update(parent.loc[parent.ne(-1)].dropna().tolist())
        numeric_cols = [c for c in ('id', 'type', 'x', 'y', 'z', 'x_nii', 'y_nii', 'z_nii', 'radius', 'parent') if c in df]
        a = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(a.to_numpy()).all():
            findings.append({'file': rel(path), 'sheet': '__csv__', 'check': 'node_nonfinite_numeric', 'status': 'verified_numeric_issue'})
        if {'x_nii', 'y_nii', 'z_nii'} <= set(df):
            delta = np.abs(a[['x', 'y', 'z']].to_numpy()/250 - a[['x_nii', 'y_nii', 'z_nii']].to_numpy())
            if np.any(delta > .000050001):
                findings.append({'file': rel(path), 'sheet': '__csv__', 'check': 'node_XYZ_vs_250um_index', 'status': 'verified_encoding_issue', 'max_index_delta': float(delta.max())})
    if is_node and (roots != 1 or duplicates or not parent_ids <= node_ids):
        findings.append({'file': rel(path), 'sheet': '__csv__', 'check': 'node_parent_identity', 'status': 'verified_graph_table_issue', 'roots': roots, 'duplicates': duplicates, 'missing_parents': sorted(parent_ids-node_ids)[:20]})
    return {'sheets': [{'sheet': '__csv__', 'rows': count, 'columns': cols, 'grain': 'node' if is_node else 'table', 'node_roots': roots if is_node else None, 'node_ids': len(node_ids) if is_node else None, 'chunk_checks': checks}], 'full_content_read': True}, findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('inventory', 'workbooks', 'delimited'), required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    out.relative_to(AUDIT)
    if args.phase == 'inventory':
        out.mkdir(parents=True, exist_ok=False)
        roots = list(ROOT.glob('neuron_tables*')) + [ROOT/'group_analysis', ROOT/'R_analysis/tables', ROOT/'main_scripts/neuron_tables'] + list((ROOT/'notes').glob('region_analysis_review*'))
        roots = [r for r in roots if r.is_dir()]
        files = sorted({p for base in roots for p in base.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.is_relative_to(AUDIT)})
        entries, groups = [], defaultdict(list)
        for i, p in enumerate(files):
            stat = p.stat(); h = digest(p)
            entries.append({'file': rel(p), 'extension': p.suffix.lower(), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': h, 'role_status': role(p), 'role_basis': 'path_context_only_not_acceptance'})
            groups[h].append(rel(p))
            if i % 300 == 0:print(json.dumps({'hashed': i, 'total': len(files)}), flush=True)
        csvout(out/'file_inventory.csv', entries)
        save(out/'inventory_groups.json', {'observed_at': datetime.now().astimezone().isoformat(), 'roots': [rel(p) for p in roots], 'files': entries, 'groups': groups, 'unique_content_groups': len(groups), 'total_files': len(entries), 'total_bytes': sum(e['bytes'] for e in entries), 'audit_directory_excluded': True, 'canonical_writes': False})
        print(json.dumps({'files': len(entries), 'unique_contents': len(groups)}), flush=True)
        return
    inv = json.loads((out/'inventory_groups.json').read_text())
    selected = [(h, paths) for h, paths in inv['groups'].items() if (Path(paths[0]).suffix.lower()=='.xlsx') == (args.phase=='workbooks')]
    schemas, findings, strengths, pis = [], [], [], []
    for i, (h, names) in enumerate(selected):
        path = ROOT/names[0]
        try:
            if digest(path) != h:raise RuntimeError('Source changed since inventory')
            if args.phase == 'workbooks':
                result, found, st, pi = workbook(path);strengths += st;pis += pi
            elif path.suffix.lower()=='.parquet':
                df=pd.read_parquet(path);schema,found=inspect_frame(df,path,'__parquet__');result={'sheets':[schema]}
            else:
                result, found = delimited(path)
            if digest(path) != h:raise RuntimeError('Source changed during read')
            result.update(representative_file=names[0], sha256=h, identical_copies=names, status='readable')
            schemas.append(result)
            for r in found:r.update(sha256=h, identical_copy_count=len(names))
            findings += found
        except Exception as exc:
            schemas.append({'representative_file': names[0], 'sha256': h, 'identical_copies': names, 'status': 'unreadable_or_unassessed', 'error': str(exc)})
            findings.append({'file': names[0], 'sha256': h, 'check': 'parse_failure', 'status': 'explicit_unassessed', 'error': str(exc)})
        if i % (5 if args.phase=='workbooks' else 100)==0:print(json.dumps({'phase':args.phase,'checked':i+1,'total':len(selected),'findings':len(findings)}),flush=True)
    save(out/(args.phase+'_schemas.json'), {'observed_at': datetime.now().astimezone().isoformat(), 'content_groups': len(selected), 'represented_files': sum(len(n) for _, n in selected), 'schemas': schemas})
    csvout(out/(args.phase+'_findings.csv'), findings)
    if args.phase=='workbooks':
        csvout(out/'projection_strength_checks.csv',strengths)
        csvout(out/'Pi_target_observations.csv',pis)
    print(json.dumps({'phase':args.phase,'content_groups':len(selected),'represented_files':sum(len(n) for _,n in selected),'unreadable':sum(s['status']!='readable' for s in schemas),'findings':len(findings),'checks':dict(Counter(r['check'] for r in findings))}),flush=True)


if __name__ == '__main__':
    main()
