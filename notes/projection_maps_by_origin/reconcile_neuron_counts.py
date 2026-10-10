"""Reconcile historical selected identities and current evidence by monkey.

July tracker totals are dated evidence. The local canonical workbook supplies
the older identity set; matching counts do not establish its July byte lineage.
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from map_projections_by_soma_origin import DEFAULT_LEDGER, evidence_categories


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(frame):
    if frame[['SampleID', 'NeuronID']].duplicated().any():
        raise ValueError('Duplicated exact neuron identity')
    expected = frame.SampleID + '::' + frame.NeuronID
    if not frame.NeuronUID.eq(expected).all():
        raise ValueError('NeuronUID does not match its exact sample and filename')
    return set(frame.NeuronUID)


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh count reconciliation destination')
    paths = {
        'July13_tracker': ROOT / 'group_analysis/docs/data_progress_table_20260713_1649.csv',
        'July27_tracker': ROOT / 'group_analysis/docs/data_progress_table_20260727_1620.csv',
        'older_combined306': ROOT / 'group_analysis/combined/multi_monkey_INS_combined.xlsx',
        'older_harmonized353': ROOT / 'group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx',
        'September26_staging420': ROOT / 'group_analysis/staging_20260926/combined/multi_monkey_INS_combined_harmonized.xlsx',
        'current462': DEFAULT_LEDGER,
        'registry': ROOT / 'group_analysis/docs/dataset_status_manifest.csv',
        'all_neuron_inventory': ROOT / 'group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/all_neuron_review_manifest.csv',
    }
    frames = {key: pd.read_excel(paths[key], sheet_name='Summary', dtype=str).fillna('')
              for key in ('older_combined306', 'older_harmonized353', 'September26_staging420')}
    current = pd.read_csv(DEFAULT_LEDGER, dtype=str, keep_default_na=False)
    frames['current462'] = current
    sets = {key: identities(frame) for key, frame in frames.items()}
    assert [len(sets[k]) for k in frames] == [306, 353, 420, 462]
    assert sets['older_harmonized353'] <= sets['current462']
    tracker = pd.read_csv(paths['July27_tracker'], dtype=str, keep_default_na=False)
    tracker = tracker[tracker.monkey_id.ne('TOTAL')].set_index('fmost_id')
    july13 = pd.read_csv(paths['July13_tracker'], dtype=str, keep_default_na=False)
    assert july13.set_index('fmost_id').insula_in_combined_n.to_dict() == pd.read_csv(paths['July27_tracker'], dtype=str).set_index('fmost_id').insula_in_combined_n.to_dict()
    older = frames['older_harmonized353'].groupby('SampleID').size().to_dict()
    assert all(int(r.insula_in_combined_n) == older.get(s, 0) for s, r in tracker.iterrows())
    current['OriginEvidenceCategory'] = evidence_categories(current)
    current['InOlderHarmonized353'] = current.NeuronUID.isin(sets['older_harmonized353'])
    current['InSeptemberStaging420'] = current.NeuronUID.isin(sets['September26_staging420'])
    rows = []
    for (animal, sample), part in current.groupby(['AnimalID', 'SampleID'], sort=True):
        counts = part.OriginEvidenceCategory.value_counts()
        row = {'AnimalID': animal, 'SampleID': sample, 'July_tracker_selected': int(tracker.loc[sample].insula_in_combined_n),
               'older_combined306': sum(frames['older_combined306'].SampleID.eq(sample)),
               'September_staging_selected': sum(frames['September26_staging420'].SampleID.eq(sample)),
               'current_selected': len(part), 'added_vs_older353': sum(~part.InOlderHarmonized353),
               'removed_vs_older353': len(set(frames['older_harmonized353'].query('SampleID == @sample').NeuronUID) - sets['current462']),
               'atlas_insula': int(counts.get('atlasAndHenry', 0) + counts.get('atlasOnly', 0)),
               'Henry_INS_outside_atlas_insula': int(counts.get('HenryPrCOConflict', 0) + counts.get('HenryUnassigned', 0)),
               'remaining_candidates': int(counts.get('neighborCandidate', 0) + counts.get('unassignedCandidate', 0)),
               'endpoint_eligible': int(part.EndpointEligible.eq('True').sum())}
        row.update({key: int(counts.get(key, 0)) for key in ('atlasAndHenry', 'atlasOnly', 'HenryPrCOConflict', 'HenryUnassigned', 'neighborCandidate', 'unassignedCandidate')})
        rows.append(row)
    table = pd.DataFrame(rows)
    missing = frames['September26_staging420'][~frames['September26_staging420'].NeuronUID.isin(sets['current462'])].copy()
    inventory = pd.read_csv(paths['all_neuron_inventory'], dtype=str, keep_default_na=False)
    inventory['NeuronUID'] = inventory['sample'] + '::' + inventory['neuron_id']
    reasons = inventory.set_index('NeuronUID')
    missing['current_exclusion_reason'] = missing.NeuronUID.map(reasons.animal_aggregation_exclusion_reason)
    missing['registry_animal'] = missing.NeuronUID.map(reasons.registry_animal)
    assert len(missing) == 8 and missing.SampleID.eq('250432').all() and missing.registry_animal.eq('').all()
    assert missing.current_exclusion_reason.eq('Exact sample/channel absent from animal registry; no animal ID inferred').all()
    output.mkdir(parents=True)
    table.to_csv(output / 'per_monkey_counts_history_and_evidence.csv', index=False)
    fields = ['NeuronUID', 'SampleID', 'NeuronID', 'AnimalID', 'ARMIndex', 'ARMFullName', 'OriginEvidenceCategory', 'InOlderHarmonized353', 'InSeptemberStaging420', 'EndpointEligible']
    current[fields].to_csv(output / 'current462_identity_comparison.csv', index=False)
    missing[['NeuronUID', 'SampleID', 'NeuronID', 'Soma_Region_Refined', 'Soma_Region_Source', 'registry_animal', 'current_exclusion_reason']].to_csv(output / 'September_entries_outside_current_monkey_maps.csv', index=False)
    pd.crosstab(current.InOlderHarmonized353, current.OriginEvidenceCategory).to_csv(output / 'retained_and_added_by_evidence.csv')
    labels = table.AnimalID + '\n' + table.SampleID
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for offset, column, color, name in [(-.24, 'July_tracker_selected', '#829ab1', 'July tracker (353)'), (0, 'September_staging_selected', '#b89e68', 'September staging: registered monkeys (412)'), (.24, 'current_selected', '#466c93', 'Current maps (462)')]:
        bars = axes[0].bar(x + offset, table[column], .23, color=color, label=name)
        axes[0].bar_label(bars, fontsize=7, padding=2)
    bottom = np.zeros(len(table))
    for column, color, name in [('atlas_insula', '#237f98', 'Named ARM insula soma assignment (301)'), ('Henry_INS_outside_atlas_insula', '#e3a15d', 'Henry coarse INS; conflicting/unassigned ARM (48)'), ('remaining_candidates', '#b9bfc6', 'Other candidates (113)')]:
        bars = axes[1].bar(x, table[column], .65, bottom=bottom, color=color, label=name)
        for i, value in enumerate(table[column]):
            if value:
                axes[1].text(i, bottom[i] + value / 2, str(value), ha='center', va='center', fontsize=8)
        bottom += table[column]
    axes[1].bar_label(bars, labels=table.current_selected, fontsize=8, padding=2)
    for ax, title in zip(axes, ('Historical selected counts', 'Current selected neurons by source evidence')):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel('Monkey ID / fMOST dataset ID')
        ax.set_ylabel('Selected reconstructions')
        ax.set_ylim(0, 305)
        ax.set_title(title)
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Insula-projectome selection grew from 353 to 462; ARM assignment is a separate count\nSeptember staging also contains 8 dataset-250432 entries without verified monkey identity (total 420).\nEvidence categories do not establish anatomical acceptance; unequal sampling is visible.', fontsize=11)
    fig.savefig(output / 'per_monkey_history_and_evidence.png', dpi=160)
    plt.close(fig)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'producer_sha256': sha(__file__),
              'sources': {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in paths.items()},
              'counts': {'July_tracker': 353, 'September_staging': 420, 'current': 462, 'older353_retained': 353, 'added_vs_older353': 109,
                         'September_retained': 412, 'added_vs_September': 50, 'September_unregistered_outside_maps': 8,
                         'older353_current_ARM_insula': 258, 'added_current_ARM_insula': 43, 'current_ARM_insula': 301,
                         'current_INS_supported_union': 349, 'other_candidates': 113},
              'date_limit': 'July tracker is dated; the local 353-workbook identity comparison is not proof of identical July workbook bytes.',
              'anatomical_acceptance': False,
              'artifacts': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'reconciliation_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(table.to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
