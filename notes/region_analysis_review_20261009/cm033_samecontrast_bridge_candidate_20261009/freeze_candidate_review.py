"""Additive candidate review binding after root/Codex actual PNG inspection."""
from datetime import datetime, timezone
import json
from pathlib import Path

from independent_coordinate_checks import binding

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / 'candidate_image_review.json'
    if output.exists():
        raise FileExistsError(output)
    names = ['README.md', 'run_declared_coordinate_bridge.py', 'bridge_candidate_provenance.json',
             'candidate_saved_readback.json', 'independent_coordinate_checks.py', 'independent_coordinate_checks.json',
             'prospective_coordinate_composition.json', 'prospective_statistic_chain.json',
             'coarse_qc/cm033_forward_mean_intensity_correspondence.png',
             'coarse_qc/cm033_inverse_mean_intensity_correspondence.png', 'fit_execution_log.txt',
             'statistic_chain/chain_provenance.json', 'freeze_candidate_review.py']
    receipt = dict(status='provisional_computational_correspondence_image_reviewed_no_anatomical_acceptance',
                   reviewed_utc=datetime.now(timezone.utc).isoformat(), reviewers=['Codex projection_methods', 'Codex root'],
                   both_actual_forward_and_inverse_PNGs_inspected=True, readable_titles_and_plane_labels=True,
                   observation='Gross structures and midline correspond reasonably at the six displayed slices; local/peripheral/smoothing differences remain.',
                   residual_rotation_degrees=2.067773891766367, unchanged_residual_gate_degrees=30,
                   B_is_coordinate_recoding_not_accepted_physical_rotation=True, historical_payload_proven=False,
                   anatomical_acceptance=False, third_fit_started=False,
                   later_statistic_chain='Separately authorized and producer-completed; independent saved-transform reproduction is a distinct gate.',
                   bindings=[binding(HERE / name) for name in names])
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({'receipt': binding(output), 'bindings': len(names)}))


if __name__ == '__main__':
    main()
