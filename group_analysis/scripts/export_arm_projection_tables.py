"""Export one source-bound ARM L1–L6 workbook from completed map runs.

Candidate endpoints are reconstruction proxies, not verified terminals. Their
denominator is eligible neurons within each animal/source group. Axon lengths
are child-type-2 template-space millimetres per selected neuron, not legacy
retained reconstruction lengths. Six actual ARM volumes remain separate.
No NIfTI is created or modified. Existing outputs are never replaced.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
import nibabel as nib
from openpyxl import Workbook
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'main_scripts'))
from projection_maps import axon_length_map
from endpoint_atlas import coded_mm_grid, matching_grid
IDENTITY = ['SampleID', 'NeuronID']
SOURCE_FIELDS = ['ARMLevel', 'ARMIndex', 'ARMAbbreviation', 'ARMFullName', 'Hemisphere', 'SourceARMStatus']

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, nullable=False):
    if nullable and (value is None or str(value) == ''):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Expected finite integral label/count') from exc
    require(number.is_finite() and number == number.to_integral_value(),
        'Expected finite integral label/count')
    return int(number)

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def uid(frame):
    return frame.SampleID + '::' + frame.NeuronID

def exact_rows(frame, manifest, label):
    require(set(IDENTITY) <= set(frame), f'{label}: missing identity fields')
    require(not frame.duplicated(IDENTITY).any(), f'{label}: duplicate identities')
    frame = frame.copy()
    frame.index = uid(frame)
    require(set(frame.index) == set(uid(manifest)), f'{label}: missing/extra identities')
    return frame.loc[uid(manifest)].reset_index(drop=True)

def checked_records(manifest, input_root):
    spec = importlib.util.spec_from_file_location('arm_export_manifest',
        Path(__file__).with_name('build_projection_maps.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checked_manifest(manifest, input_root)

def catalog(atlas_path, key_path, grid):
    image = nib.load(str(atlas_path))
    matching_grid(image, grid, spatial_only=True)
    require(image.shape == (*grid.shape, 1, 6), 'ARM requires six actual volumes')
    data = np.asanyarray(image.dataobj)
    require(np.issubdtype(data.dtype,
        np.integer) and np.all(data >= 0),
        'ARM labels must be nonnegative integers')
    key = read_csv(key_path) if str(key_path).endswith('.csv') else pd.read_csv(key_path,
        sep='\t',
        dtype=str,
        keep_default_na=False)
    require({'Index',
        'Abbreviation',
        'Full_Name',
        'First_Level',
        'Last_Level'} <= set(key),
        'Incomplete official ARM key')
    labels = {}
    for row in key.to_dict('records'):
        index = integer(row['Index'])
        first = integer(row['First_Level'])
        last = integer(row['Last_Level'])
        require(index > 0 and index not in labels and (1 <= first <= last <= 6),
            'Invalid ARM key index/level range')
        require(bool(row['Full_Name'].strip()) and bool(row['Abbreviation'].strip()),
            'Blank official ARM names')
        labels[index] = row | {'first': first, 'last': last}
    levels = []
    rows = []
    for level in range(1, 7):
        values = np.ascontiguousarray(data[..., 0, level - 1]).ravel()
        levels.append(values)
        (indices, counts) = np.unique(values, return_counts=True)
        for (index, count) in zip(indices, counts):
            index = int(index)
            metadata = labels.get(index)
            if index == 0:
                status = 'zero_unassigned'
            elif metadata is None:
                status = 'unmapped_label'
            elif metadata['first'] <= level <= metadata['last']:
                status = 'mapped'
            else:
                status = 'key_level_conflict'
            abbreviation = metadata['Abbreviation'] if metadata else ''
            if metadata:
                name = metadata['Full_Name']
            else:
                name = 'Atlas label 0 (unassigned)' if index == 0 else f'Unmapped ARM index {index}'
            prefix = abbreviation[:2]
            domain = 'Cortex' if prefix in ('CL', 'CR') else 'Subcortex'
            if prefix not in ('CL', 'CR', 'SL', 'SR'):
                domain = 'Unassigned'
            side = 'L' if prefix in ('CL', 'SL') else 'R'
            if domain == 'Unassigned':
                side = 'Unknown'
            rows.append(dict(TargetID=f'L{level}:{status}:{index}',
                Level=level,
                ARMIndex=index,
                OfficialFullName=name,
                Abbreviation=abbreviation,
                Domain=domain,
                Hemisphere=side,
                TargetStatus=status,
                ActualVolumeMm3=int(count) * grid.voxel_volume_mm3,
                KeyRangeConflict=status == 'key_level_conflict'))
        if not np.any(indices == 0):
            rows.append(dict(TargetID=f'L{level}:zero_unassigned:0',
                Level=level,
                ARMIndex=0,
                OfficialFullName='Atlas label 0 (unassigned)',
                Abbreviation='',
                Domain='Unassigned',
                Hemisphere='Unknown',
                TargetStatus='zero_unassigned',
                ActualVolumeMm3=0.0,
                KeyRangeConflict=False))
        rows.append(dict(TargetID=f'L{level}:out_of_FOV:outside',
            Level=level,
            ARMIndex=None,
            OfficialFullName='Outside reference field of view',
            Abbreviation='',
            Domain='Outside',
            Hemisphere='Unknown',
            TargetStatus='out_of_FOV',
            ActualVolumeMm3=None,
            KeyRangeConflict=False))
    return (pd.DataFrame(rows), levels)

def regional_lengths(data, levels, targets):
    occupied = np.flatnonzero(data)
    weights = data.ravel()[occupied]
    result = {}
    lookup = {(int(row.Level),
        integer(row.ARMIndex)): row.TargetID for row in targets.itertuples() if row.TargetStatus != 'out_of_FOV'}
    for (level, labels) in enumerate(levels, 1):
        sums = np.bincount(labels[occupied].astype(np.int64), weights=weights)
        for index in np.flatnonzero(sums):
            result[lookup[level, int(index)]] = float(sums[index])
    return result

def validate_source_labels(frame, targets, atlas_hash, key_hash):
    """Keep official ARM parcels and unassigned source QC strata distinct."""
    require(set(SOURCE_FIELDS) <= set(frame), 'Final ledger lacks full ARM source metadata')
    lookup = {(int(row.Level),
        integer(row.ARMIndex)): row for row in targets.itertuples() if row.TargetStatus != 'out_of_FOV'}
    for row in frame.to_dict('records'):
        level = integer(row['ARMLevel'])
        index = integer(row['ARMIndex'])
        side = row['Hemisphere']
        require(level == 6 and side in ('L',
            'R') and ((level,
            index) in lookup),
            'Invalid actual finest ARM source label/side')
        metadata = lookup[level, index]
        require(index == 0 or side == metadata.Hemisphere,
            'Positive ARM source hemisphere differs from official key prefix')
        require(row['Subregion'] == f'ARM{level}_{index}_{side}', 'Source group differs from ARM index/side')
        expected_name = 'Atlas background' if index == 0 else metadata.OfficialFullName
        require(row['ARMFullName'] == expected_name and row['ARMAbbreviation'] == metadata.Abbreviation and (row['SourceARMStatus'] == metadata.TargetStatus),
            'Source official ARM metadata differs from key/resource')
        for (field, expected) in (('AtlasSHA256', atlas_hash), ('AtlasKeySHA256', key_hash)):
            if field in row:
                require(row[field] == expected, f'Source {field} differs from bound resource')
    metadata = frame[['Subregion', *SOURCE_FIELDS]].drop_duplicates()
    require(not metadata.Subregion.duplicated().any(), 'Source group has conflicting ARM metadata')
    return metadata

def conditional_means(counts, lengths, eligible, summary):
    """Within-animal conditional means, then equal available-animal means."""
    # Endpoint means use only eligible neurons; axon means use every selected neuron.
    animals = []
    for ((animal, source), indices) in summary.groupby(['AnimalID', 'Subregion'], sort=True).groups.items():
        indices = np.asarray(list(indices))
        valid = indices[eligible[indices]]
        n = len(valid)
        for target in counts.columns:
            animals.append(dict(AnimalID=animal,
                Subregion=source,
                TargetID=target,
                NSelected=len(indices),
                NEndpointEligible=n,
                CandidateEndpointMean=float(counts.loc[valid,
                target].mean()) if n else np.nan,
                EndpointNeuronFrequency=float((counts.loc[valid,
                target] > 0).mean()) if n else np.nan,
                AxonTemplateLengthMeanMm=float(lengths.loc[indices,
                target].mean())))
    animals = pd.DataFrame(animals)
    groups = []
    for ((source, target), frame) in animals.groupby(['Subregion', 'TargetID'], sort=True):
        available = frame.NEndpointEligible.gt(0)
        groups.append(dict(Subregion=source,
            TargetID=target,
            NSelected=int(frame.NSelected.sum()),
            NEndpointEligible=int(frame.NEndpointEligible.sum()),
            NEndpointAnimals=int(available.sum()),
            NAxonAnimals=len(frame),
            CandidateEndpointMean=frame.loc[available,
            'CandidateEndpointMean'].mean(),
            EndpointNeuronFrequency=frame.loc[available,
            'EndpointNeuronFrequency'].mean(),
            AxonTemplateLengthMeanMm=frame.AxonTemplateLengthMeanMm.mean()))
    return (animals, pd.DataFrame(groups))

def _write_sheet(workbook, name, frame):
    sheet = workbook.create_sheet(name)
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value.item() if isinstance(value,
            np.generic) else value for value in row])

def export(endpoint_run,
    axon_run,
    output,
    *,
    manifest=None,
    additional_endpoint_run=None,
    additional_axon_run=None,
    input_root=ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh hierarchy-table output directory')
    require(bool(additional_endpoint_run) == bool(additional_axon_run),
        'Additional runs must be supplied as a pair')
    pairs = [(Path(endpoint_run), Path(axon_run))]
    if additional_endpoint_run:
        pairs.append((Path(additional_endpoint_run), Path(additional_axon_run)))
    main_ep = json.loads((pairs[0][0] / 'run_provenance.json').read_text(encoding='utf-8'))
    manifest = Path(manifest or pairs[0][0] / 'input_manifest.csv')
    records = checked_records(manifest, input_root)
    frame = read_csv(manifest)
    frame['NeuronUID'] = uid(frame)
    grid = coded_mm_grid(nib.load(main_ep['reference']))
    (targets,
        levels) = catalog(main_ep['inputs']['atlas']['path'],
        main_ep['inputs']['atlas_key']['path'],
        grid)
    source_metadata = validate_source_labels(frame,
        targets,
        main_ep['inputs']['atlas']['sha256'],
        main_ep['inputs']['atlas_key']['sha256'])
    target_ids = targets.TargetID.tolist()
    target_position = {name: i for (i, name) in enumerate(target_ids)}
    row_position = {name: i for (i, name) in enumerate(frame.NeuronUID)}
    bindings = {str(manifest.resolve()): sha256(manifest)}
    runs = []
    seen = set()
    qc_frames = []
    axon_frames = []
    target_frames = []
    for (ep_dir, ax_dir) in pairs:
        ep = json.loads((ep_dir / 'run_provenance.json').read_text(encoding='utf-8'))
        ax = json.loads((ax_dir / 'run_provenance.json').read_text(encoding='utf-8'))
        require(ep['status'] == 'software_verified_candidate_endpoints' and ax['status'] == 'software_verified_descriptive',
            'Runs must be completed software-verified outputs')
        for (name, binding) in ep['inputs'].items():
            require(sha256(binding['path']) == binding['sha256'], f'Changed endpoint input: {name}')
            bindings[str(Path(binding['path']).resolve())] = binding['sha256']
        require(ep['reference_sha256'] == main_ep['reference_sha256'] == ax['reference_sha256'],
            'Reference binding differs')
        for recorded in (ep, ax):
            require(tuple(recorded['shape']) == grid.shape and np.allclose(recorded['affine_mm'],
                grid.affine_mm,
                rtol=0,
                atol=1e-07) and np.isclose(recorded['voxel_volume_mm3'],
                grid.voxel_volume_mm3,
                rtol=0,
                atol=1e-10),
                'Recorded map geometry differs from reference')
        for name in ('atlas', 'atlas_key'):
            require(ep['inputs'][name]['sha256'] == main_ep['inputs'][name]['sha256'],
                'Runs use different ARM atlas/key')
        require(sha256(ax['reference']) == ax['reference_sha256'], 'Changed axon reference')
        for name in ('projection_maps.py', 'swc_validation.py'):
            require(sha256(ROOT / 'main_scripts' / name) == ax['source_code_sha256'][name],
                f'Raster kernel version differs: {name}')
        for (name, artifact) in ep['artifacts'].items():
            require(sha256(ep_dir / name) == artifact['sha256'], f'Endpoint artifact hash mismatch: {name}')
        require(sha256(ep_dir / 'input_manifest.csv') == sha256(ax_dir / 'input_manifest.csv') == ax['manifest_sha256'],
            'Run manifests differ')
        run_frame = read_csv(ep_dir / 'input_manifest.csv')
        ids = uid(run_frame).tolist()
        require(len(set(ids)) == len(ids) and (not seen.intersection(ids)),
            'Duplicate or overlapping run identities')
        require(set(ids) <= set(row_position), 'Run contains identities outside final manifest')
        seen.update(ids)
        for item in run_frame.to_dict('records'):
            original = frame.iloc[row_position[item['SampleID'] + '::' + item['NeuronID']]]
            for field in ('AnimalID', 'SWCSHA256', 'ReferenceSHA256', 'CoordinateFrame', 'IndexScaleUm'):
                require(item[field] == original[field],
                    f'Combined/source manifest identity or encoding mismatch: {field}')
        qc = exact_rows(read_csv(ep_dir / 'per_neuron_qc.csv'), run_frame, 'endpoint QC')
        metrics = exact_rows(read_csv(ax_dir / 'per_neuron_measurements.csv'), run_frame, 'axon measurements')
        for table in (qc, metrics):
            for field in ('AnimalID', 'Subregion', 'SWCSHA256'):
                require(table[field].tolist() == run_frame[field].tolist(),
                    f'Run per-neuron metadata mismatch: {field}')
        require(integer(ep['n_selected']) == len(run_frame) and integer(ax['selected_neurons']) == len(run_frame) and (integer(ep['n_computable']) == sum((integer(x) for x in qc.n_computable))),
            'Recorded run denominators differ')
        expected_animals = set(zip(run_frame.AnimalID, run_frame.Subregion))
        expected_groups = set(run_frame.Subregion)
        for recorded in (ep, ax):
            actual_animals = [(row['AnimalID'], row['Subregion']) for row in recorded['animal_maps']]
            actual_groups = [row['Subregion'] for row in recorded['group_maps']]
            require(len(actual_animals) == len(set(actual_animals)) and set(actual_animals) == expected_animals and (len(actual_groups) == len(set(actual_groups))) and (set(actual_groups) == expected_groups),
                'Saved map group coverage differs')
        qc['OriginalEndpointSourceGroup'] = qc.Subregion
        qc['EndpointRun'] = str(ep_dir.resolve())
        metrics['OriginalAxonSourceGroup'] = metrics.Subregion
        qc_frames.append(qc)
        axon_frames.append(metrics)
        subset = read_csv(ep_dir / 'per_neuron_target_counts.csv')
        subset['NeuronUID'] = uid(subset)
        require(set(subset.NeuronUID) <= set(ids), 'Target rows contain unknown identities')
        target_frames.append(subset)
        runs.append(dict(endpoint_dir=ep_dir, axon_dir=ax_dir, endpoint=ep, axon=ax, frame=run_frame))
        for path in (ep_dir / 'run_provenance.json',
            ax_dir / 'run_provenance.json',
            ax_dir / 'per_neuron_measurements.csv'):
            bindings[str(path.resolve())] = sha256(path)
    require(seen == set(frame.NeuronUID), 'Final manifest has unrepresented run identities')
    require(all((record['ReferenceSHA256'] == main_ep['reference_sha256'] for record in records)),
        'Combined manifest reference mismatch')
    qc = exact_rows(pd.concat(qc_frames, ignore_index=True), frame, 'combined QC')
    metrics = exact_rows(pd.concat(axon_frames, ignore_index=True), frame, 'combined axon metrics')
    eligible = np.asarray([integer(value) == 1 for value in qc.n_computable])
    candidates = np.asarray([integer(value) for value in qc.candidate_axon_endpoint_count])
    require(np.array_equal(eligible,
        candidates > 0),
        'Endpoint eligibility differs from full-graph candidate evidence')
    counts = pd.DataFrame(0.0, index=range(len(frame)), columns=target_ids)
    counts.loc[~eligible, :] = np.nan
    long = pd.concat(target_frames, ignore_index=True)
    observed = set()
    totals = np.zeros((len(frame), 6), dtype=np.int64)
    for row in long.itertuples():
        level = integer(row.level)
        index = integer(row.target_index, nullable=True)
        token = f"L{level}:{row.target_status}:{(index if index is not None else 'outside')}"
        require(token in target_position, 'Target row differs from actual ARM level/key status')
        neuron = row_position[row.NeuronUID]
        count = integer(row.endpoint_count)
        for field in ('AnimalID', 'Subregion', 'SWCSHA256'):
            require(getattr(row,
                field) == qc.loc[neuron,
                field],
                f'Target row identity metadata differs: {field}')
        metadata = targets.iloc[target_position[token]]
        if metadata.Abbreviation:
            require(row.target_abbreviation == metadata.Abbreviation and row.target_full_name == metadata.OfficialFullName,
                'Target official ARM names differ')
        if row.target_status == 'mapped':
            require(np.isclose(float(row.target_volume_mm3),
                metadata.ActualVolumeMm3,
                rtol=1e-09),
                'Target volume differs from actual ARM grid')
        require(integer(row.neurons_with_target_evidence) == 1,
            'Regional endpoint evidence must count each neuron once')
        require(eligible[neuron] and count > 0 and ((neuron,
            token) not in observed),
            'Duplicate/unassessed/invalid endpoint target row')
        observed.add((neuron, token))
        counts.loc[neuron, token] = count
        totals[neuron, level - 1] += count
    require(np.array_equal(totals,
        np.repeat(candidates[:,
        None],
        6,
        axis=1)),
        'Endpoint target counts do not conserve full-graph candidate counts at every level')
    # Create derivatives only after identity, source and endpoint conservation checks pass.
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='running',
        created_utc=datetime.now(timezone.utc).isoformat(),
        manifest=str(manifest.resolve()),
        manifest_sha256=sha256(manifest),
        reference_sha256=main_ep['reference_sha256'],
        atlas_sha256=main_ep['inputs']['atlas']['sha256'],
        atlas_key_sha256=main_ep['inputs']['atlas_key']['sha256'],
        source_bindings=bindings,
        exporter_sha256=sha256(__file__),
        n_selected=len(frame),
        n_endpoint_eligible=int(eligible.sum()),
        scientific_anatomical_acceptance=False,
        atlas_policy='six actual ARM volumes; no inferred parent pooling',
        endpoint_policy='candidate counts and distinct neuron presence; ineligible NA; per-animal/source eligible denominator',
        axon_policy='all selected child-type2 template length mm; not legacy retained length',
        group_policy='equal available-animal means; no absent animal/source imputation',
        versions={'python': sys.version,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'nibabel': nib.__version__})
    lengths = pd.DataFrame(0.0, index=range(len(frame)), columns=target_ids)
    try:
        for (i, record) in enumerate(records):
            (data,
                measurement) = axon_length_map(record['source_path'].read_text(encoding='utf-8-sig'),
                grid,
                coordinate_frame=record['CoordinateFrame'],
                index_scale_um=record['index_scale_um'],
                source=str(record['source_path']))
            require(sha256(record['source_path']) == record['expected_sha256'],
                'SWC changed during rasterization')
            for name in ('selected_axon_length_mm', 'in_reference_length_mm', 'outside_reference_length_mm'):
                require(np.isclose(measurement[name],
                    float(metrics.loc[i,
                    name]),
                    rtol=1e-09,
                    atol=1e-09),
                    'Recomputed source length differs from saved measurement')
            for (token, value) in regional_lengths(data, levels, targets).items():
                lengths.loc[i, token] = value
            for level in range(1, 7):
                lengths.loc[i, f'L{level}:out_of_FOV:outside'] = measurement['outside_reference_length_mm']
            if (i + 1) % 20 == 0:
                print(f'Rasterized {i + 1}/{len(records)} sources once', flush=True)
        (animals, groups) = conditional_means(counts, lengths, eligible, frame)
        target_metadata = targets.rename(columns={'Hemisphere': 'TargetHemisphere',
            'ARMIndex': 'TargetARMIndex'})
        animals = animals.merge(source_metadata,
            on='Subregion',
            validate='many_to_one').merge(target_metadata,
            on='TargetID',
            validate='many_to_one')
        groups = groups.merge(source_metadata,
            on='Subregion',
            validate='many_to_one').merge(target_metadata,
            on='TargetID',
            validate='many_to_one')
        # Reconcile maps with their original run groups, before final-ledger regrouping.
        checks = []
        for run in runs:
            runframe = run['frame']
            runids = uid(runframe)
            for (kind,
                metric,
                path_key,
                hash_key,
                values) in (('endpoint',
                'candidate_count',
                'count_path',
                'count_sha256',
                counts),
                ('axon',
                'axon_length_mm',
                'length_path',
                'length_sha256',
                lengths)):
                metadata = run[kind]
                directory = run[kind + '_dir']
                for scope in ('animal_maps', 'group_maps'):
                    for entry in metadata[scope]:
                        if kind == 'endpoint' and (not entry['map_available']):
                            continue
                        source = entry['Subregion']
                        selected = runframe.Subregion.eq(source)
                        if scope == 'animal_maps':
                            selected &= runframe.AnimalID.eq(entry['AnimalID'])
                        positions = np.asarray([row_position[value] for value in runids[selected]])
                        local = frame.loc[positions, ['AnimalID', 'Subregion']].copy()
                        local['Subregion'] = source
                        means = []
                        for animal in runframe.loc[selected, 'AnimalID'].unique():
                            indices = positions[frame.loc[positions, 'AnimalID'].eq(animal).to_numpy()]
                            if kind == 'endpoint':
                                indices = indices[eligible[indices]]
                            if len(indices):
                                means.append(values.loc[indices].mean().to_numpy())
                        expected = np.mean(means, axis=0)
                        path = directory / entry[path_key]
                        require(sha256(path) == entry[hash_key], 'Saved map hash differs')
                        image = nib.load(str(path))
                        matching_grid(image, grid)
                        data = np.asanyarray(image.dataobj)
                        require(np.isfinite(data).all() and np.all(data >= 0),
                            'Invalid saved spatial map values')
                        actual = regional_lengths(data, levels, targets)
                        error = 0.0
                        for (token, column) in target_position.items():
                            if ':out_of_FOV:' in token:
                                continue
                            value = actual.get(token, 0.0)
                            error = max(error, abs(value - expected[column]))
                            require(np.isclose(value,
                                expected[column],
                                rtol=1e-05,
                                atol=1e-05),
                                f'Saved NIfTI regional reconciliation failed: {path}: {token}')
                        checks.append(dict(Run=str(directory),
                            Scope=scope,
                            AnimalID=entry.get('AnimalID',
                            ''),
                            Subregion=source,
                            Metric=metric,
                            MaxAbsoluteRegionalError=error,
                            MapSHA256=entry[hash_key]))
        summary = frame.copy()
        summary['EndpointEligible'] = eligible
        for field in qc:
            if field not in frame:
                summary['Endpoint_' + field] = qc[field]
        for field in metrics:
            if field not in frame:
                summary['Axon_' + field] = metrics[field]
        sparse = []
        for i in range(len(frame)):
            for token in target_ids:
                end = counts.loc[i, token]
                length = lengths.loc[i, token]
                if length > 0 or (eligible[i] and end > 0):
                    sparse.append(dict(NeuronUID=frame.loc[i,
                        'NeuronUID'],
                        SampleID=frame.loc[i,
                        'SampleID'],
                        NeuronID=frame.loc[i,
                        'NeuronID'],
                        AnimalID=frame.loc[i,
                        'AnimalID'],
                        Subregion=frame.loc[i,
                        'Subregion'],
                        TargetID=token,
                        Level=int(token[1]),
                        EndpointEligible=bool(eligible[i]),
                        CandidateEndpointCount=end,
                        EndpointPresence=int(end > 0) if eligible[i] else np.nan,
                        AxonTemplateLengthMm=length))
        cache_parameters = {'manifest': sha256(manifest),
            'atlas': provenance['atlas_sha256'],
            'atlas_key': provenance['atlas_key_sha256'],
            'exporter': provenance['exporter_sha256'],
            'reference': provenance['reference_sha256'],
            'kernel': sha256(ROOT / 'main_scripts/projection_maps.py'),
            'frame_and_scales': 'exact per-neuron manifest',
            'measurement': 'child_type2_reference_mm_six_actual_volumes'}
        provenance['regional_cache_key'] = hashlib.sha256(json.dumps(cache_parameters,
            sort_keys=True).encode()).hexdigest()
        provenance['regional_cache_parameters'] = cache_parameters
        for (name,
            table) in [('neuron_summary.csv',
            summary),
            ('targets.csv',
            targets),
            ('per_neuron_regional_measures.csv',
            pd.DataFrame(sparse)),
            ('map_regional_reconciliation.csv',
            pd.DataFrame(checks))]:
            table.to_csv(output / name, index=False)
        workbook = Workbook(write_only=True)
        for (name,
            table) in [('Summary',
            summary),
            ('Targets',
            targets),
            ('Animal_Means',
            animals),
            ('Group_Means',
            groups)]:
            _write_sheet(workbook, name, table)
        identity = summary[['NeuronUID',
            'SampleID',
            'NeuronID',
            'AnimalID',
            'Subregion',
            *SOURCE_FIELDS,
            'EndpointEligible']]
        for level in range(1, 7):
            selected = targets[targets.Level.eq(level)]
            columns = selected.TargetID.tolist()
            display = {row.TargetID: f'{row.TargetID} | {row.OfficialFullName} | {row.Domain} | {row.Hemisphere}' for row in selected.itertuples()}
            for (name,
                table) in [('EP_Count',
                counts),
                ('EP_Presence',
                (counts > 0).astype(float).where(counts.notna())),
                ('AxonLen_mm',
                lengths)]:
                _write_sheet(workbook,
                    f'L{level}_{name}',
                    pd.concat([identity,
                    table[columns].rename(columns=display)],
                    axis=1))
        workbook.save(output / 'arm_projection_hierarchy_tables.xlsx')
        require(all((sha256(path) == expected for (path,
            expected) in bindings.items())),
            'Pinned input changed during export')
        provenance.update(status='software_verified_descriptive_tables',
            finished_utc=datetime.now(timezone.utc).isoformat(),
            regional_reconciliation_maps=len(checks),
            key_level_conflicts=targets.loc[targets.KeyRangeConflict].to_dict('records'),
            artifacts={path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    except Exception as exc:
        provenance.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with (output / 'export_provenance.json').open('x', encoding='utf-8') as stream:
            json.dump(provenance, stream, indent=2, allow_nan=False)
    return provenance
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-run', type=Path, required=True)
    parser.add_argument('--axon-run', type=Path, required=True)
    parser.add_argument('--manifest',
        type=Path,
        help='Optional final ARM ledger; exact union of supplied run identities')
    parser.add_argument('--additional-endpoint-run', type=Path)
    parser.add_argument('--additional-axon-run', type=Path)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--output',
        type=Path,
        required=True,
        help='Fresh directory; one workbook, sparse measures, source/QC metadata')
    args = parser.parse_args()
    result = export(args.endpoint_run,
        args.axon_run,
        args.output,
        manifest=args.manifest,
        additional_endpoint_run=args.additional_endpoint_run,
        additional_axon_run=args.additional_axon_run,
        input_root=args.input_root)
    print(f"{result['status']}: {result['n_selected']} neurons, {result['n_endpoint_eligible']} endpoint-eligible",
        flush=True)
