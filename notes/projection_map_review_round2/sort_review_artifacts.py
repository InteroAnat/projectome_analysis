"""Inventory frozen derivatives; archive only this review's temporary previews.

Historical hash-bound files retain their established paths. Original data,
MSTIM work, tracked caches and unrelated workspace files are never moved.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from inspect_projection_maps import ROOT, REVIEW, EVO, digest, require


def inventory(output):
    output = Path(output)
    require(not output.exists(), 'Use a fresh sorting receipt directory')
    output.mkdir(parents=True)
    catalog = REVIEW / 'publication_local_artifacts.json'
    entries = json.loads(catalog.read_text())['artifacts']
    texts = [(p.relative_to(ROOT).as_posix(), p.read_text(encoding='utf-8-sig', errors='replace'))
             for base in (REVIEW, EVO) for p in base.rglob('*')
             if p.is_file() and p.suffix.lower() in {'.json', '.md', '.py', '.txt'} and '__pycache__' not in p.parts]
    rows = []
    for item in entries:
        path = ROOT / item['path']
        require(path.is_file() and digest(path) == item['sha256'], 'Catalog artifact changed or missing: ' + item['path'])
        hash_refs = [relative for relative, text in texts if item['sha256'] in text and relative != catalog.relative_to(ROOT).as_posix()]
        name_refs = [relative for relative, text in texts if path.name in text]
        role = ('numerical_map' if path.name.endswith('.nii.gz') else 'historical_display' if path.suffix == '.png' else 'local_scientific_derivative_or_deferred_work')
        rows.append({'original_path': item['path'], 'current_path': item['path'], 'sha256': item['sha256'], 'bytes': item['bytes'], 'role': role,
                     'action': 'retain_frozen_path', 'hash_dependency_count': len(hash_refs), 'hash_dependency_examples': ';'.join(hash_refs[:6]),
                     'name_reference_count': len(name_refs), 'reason': 'Preserve established reproduction paths and dated evidence bindings; expose current outputs through indexes'})
    with (output / 'file_roles_and_dependencies.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'catalog_sha256': digest(catalog), 'script_sha256': digest(__file__),
               'inspected_artifacts': len(rows), 'text_files_searched': len(texts), 'historical_files_moved': 0,
               'source_data_modified': False, 'deferred_MSTIM_modified': False, 'policy': 'Current indexes plus immutable historical layout; do not relocate source-bound scientific derivatives'}
    (output / 'sorting_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def archive_previews(output):
    """Checked absolute paths, disjoint source/destination, exact byte readback."""
    output = Path(output)
    require(output.is_dir(), 'Create and review the dependency inventory first')
    source = (ROOT / 'work/projection_map_review_round2/paper_figures').resolve()
    destination = (ROOT / 'archive_local/projection_map_review_round2/paper_figures').resolve()
    require(source.is_relative_to(ROOT / 'work/projection_map_review_round2'), 'Unexpected preview source')
    require(destination.is_relative_to(ROOT / 'archive_local/projection_map_review_round2'), 'Unexpected archive destination')
    require(source.is_dir() and not destination.exists(), 'Archive only once into a fresh directory')
    files = sorted(source.iterdir())
    require(len(files) == 2 and all(p.is_file() and p.suffix == '.png' for p in files), 'Expected only the two inspected local PDF previews')
    mapping = [{'old_path': p.relative_to(ROOT).as_posix(), 'new_path': (destination / p.name).relative_to(ROOT).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size,
                'reason': 'Local copyrighted figure preview; retain privately outside active reports'} for p in files]
    (output / 'archive_plan.json').write_text(json.dumps(mapping, indent=2) + '\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    for item in mapping:
        require(not (ROOT / item['old_path']).exists() and digest(ROOT / item['new_path']) == item['sha256'], 'Archive readback differs')
    (output / 'archive_readback.json').write_text(json.dumps({'status': 'passed_checked_paths_and_exact_bytes', 'files': mapping}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive-previews', action='store_true', help='Apply only the explicit two-file local preview archive after inventory')
    args = parser.parse_args()
    if args.archive_previews:
        archive_previews(args.output)
    else:
        print(inventory(args.output)['inspected_artifacts'], 'frozen local artifacts inventoried')
