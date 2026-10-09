"""Archive this review's two superseded development directories only."""
from datetime import datetime, timezone
import json
import shutil
from inspect_projection_maps import ROOT, digest, require


def archive():
    review = (ROOT / 'notes/projection_map_review_round2').resolve()
    archive_root = (ROOT / 'archive_local/projection_map_review_round2/development').resolve()
    require(review.is_relative_to(ROOT / 'notes'), 'Unexpected review root')
    require(archive_root.is_relative_to(ROOT / 'archive_local/projection_map_review_round2'), 'Unexpected archive root')
    planned = []
    for name in ('inspection', 'validated_reproduction'):
        source, destination = (review / name).resolve(), (archive_root / name).resolve()
        require(source.is_relative_to(review) and destination.is_relative_to(archive_root), 'Unsafe archive path')
        require(source.is_dir() and not destination.exists(), 'Use the existing development run and a fresh archive')
        files = [p for p in source.rglob('*') if p.is_file()]
        require(all(p.suffix in {'.json', '.csv'} for p in files), 'Only development receipts/configuration may move')
        records = [{'old_path': p.relative_to(ROOT).as_posix(), 'new_path': (destination / p.relative_to(source)).relative_to(ROOT).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in files]
        planned.append((source, destination, records))
    output = review / 'sorting/development_archive_readback.json'
    require(not output.exists(), 'Preserve the prior archive receipt')
    (review / 'sorting/development_archive_plan.json').write_text(json.dumps([record for _, _, records in planned for record in records], indent=2) + '\n')
    for source, destination, records in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        for record in records:
            require(digest(ROOT / record['new_path']) == record['sha256'] and not (ROOT / record['old_path']).exists(), 'Archived development bytes differ')
    output.write_text(json.dumps({'status': 'passed_exact_bytes', 'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__),
                                'reason': 'First inspection used development code; failed driver run exposed relative-path handling. Final evidence is reproduction_checked only.',
                                'files': [record for _, _, records in planned for record in records]}, indent=2) + '\n')


if __name__ == '__main__':
    archive()
