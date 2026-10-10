"""Record numerical density ranges and figure-only colour saturation."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'group_analysis/evolution_20261008/soma_origin_maps_20261010'


def build(output_root, output):
    output_root, output = Path(output_root), Path(output)
    if output.exists():
        raise ValueError('Preserve prior scale QC; use a fresh destination')
    rows = []
    for family in ('named_ARM_origins', 'evidence_categories'):
        directory = output_root / family
        run = json.loads((directory / 'run_provenance.json').read_text())
        upper = float(run['projection_colour_upper'])
        threshold = 10 ** upper - 1
        for view, entry in run['outputs'].items():
            binding = entry['artifacts']['projection_density']
            path = directory / binding['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
                raise ValueError('Numerical map changed')
            data = np.asarray(nib.load(path).dataobj)
            positive, saturated = 0, 0
            for panel in entry['panels']:
                if panel.get('kind') == 'soma':
                    continue
                values = np.take(data, panel['cut'], axis=panel['axis'])
                positive += int(np.count_nonzero(values > 0))
                saturated += int(np.count_nonzero(np.log10(1 + values) > upper))
            rows.append({'family': family, 'view': view, 'map_path': str(path.resolve()), 'sha256': binding['sha256'],
                         'density_unit': 'candidate ends/template mm3/eligible reconstructed neuron; equal-animal mean',
                         'numerical_max_density': float(data.max()), 'colour_upper_log10': upper,
                         'approx_display_saturation_density': threshold, 'positive_marker_instances_in_three_slices': positive,
                         'saturated_marker_instances': saturated, 'saturated_fraction_of_positive': saturated / positive if positive else 0,
                         'numerical_values_capped': False})
    pd.DataFrame(rows).to_csv(output, index=False)
    print('Recorded scale QC for', len(rows), 'maps; saturation counts concern display instances, not unique whole-brain voxels.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output_root, args.output)
