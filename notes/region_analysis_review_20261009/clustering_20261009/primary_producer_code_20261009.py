"""Exploratory ARM6 projection-profile clustering on one exact selected ledger.

Whole-axon template length and candidate axon ends are separate observables.
No biological terminal acceptance, soma relabeling or independent-neuron inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import platform
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'main_scripts'))
from clustering_validation import safe_linkage, evaluate_k, validate_distance_matrix

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def bools(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Missing/invalid eligibility boolean')
    return values.isin(['true', '1']).to_numpy()

def exact_uids(frame):
    required = {'SampleID', 'NeuronID'}
    if not required.issubset(frame):
        raise ValueError('SampleID and NeuronID required')
    ids = frame.SampleID.astype(str) + '::' + frame.NeuronID.astype(str)
    if ids.duplicated().any():
        raise ValueError('Duplicate exact neuron identity')
    if 'NeuronUID' in frame and (not ids.equals(frame.NeuronUID.astype(str))):
        raise ValueError('NeuronUID does not match exact sample/channel and neuron filename')
    return ids

def prepare_target_catalog(targets):
    """Validate status-aware observed labels; outside has no official numeric ID."""
    required = {'TargetID', 'Level', 'ARMIndex', 'TargetStatus'}
    if not required.issubset(targets):
        raise ValueError('Target status and exact atlas index are required')
    result = targets.copy()
    levels = pd.to_numeric(result.Level, errors='raise')
    if not np.isfinite(levels).all() or not np.equal(levels, np.floor(levels)).all() or not levels.between(1, 6).all():
        raise ValueError('ARM hierarchy levels must be integers1 through6')
    result['Level'] = levels.astype(int)
    indices = pd.to_numeric(result.ARMIndex.replace('', np.nan), errors='raise')
    outside = result.TargetStatus.eq('out_of_FOV')
    if (indices.isna() & ~outside).any() or (outside & indices.notna()).any():
        raise ValueError('Only explicit out_of_FOV targets may have absent index')
    observed = indices[~outside]
    if (observed < 0).any() or not np.equal(observed, np.floor(observed)).all():
        raise ValueError('Observed ARM indices must be nonnegative integers')
    result['ARMIndex'] = indices.fillna(-1).astype(int)
    return result

def build_profiles(summary, targets, measures, manifest):
    """Reconstruct sparse measured profiles, never missing endpoint zero imputation."""
    summary = summary.copy()
    manifest = manifest.copy()
    summary['NeuronUID'] = exact_uids(summary)
    manifest['NeuronUID'] = exact_uids(manifest)
    if set(summary.NeuronUID) != set(manifest.NeuronUID):
        raise ValueError('Summary/manifest exact identities differ')
    summary = summary.set_index('NeuronUID').loc[manifest.NeuronUID].copy()
    manifest = manifest.set_index('NeuronUID')
    for col in ['AnimalID', 'SWCSHA256']:
        if col not in summary or col not in manifest:
            raise ValueError('Required source binding missing: ' + col)
        if not summary[col].astype(str).equals(manifest[col].astype(str)):
            raise ValueError('Source identity/hash binding differs: ' + col)
    if targets.TargetID.duplicated().any():
        raise ValueError('Duplicate target identity')
    t = prepare_target_catalog(targets)
    l6 = t[t.Level == 6].copy()
    if l6.empty:
        raise ValueError('Exclusive level6 target definitions missing')
    positive = l6[(l6.ARMIndex > 0) & (l6.TargetStatus == 'mapped')].copy()
    if ((l6.ARMIndex > 0) & (l6.TargetStatus != 'mapped')).any():
        raise ValueError('Positive level6 target is unmapped or has key-level conflict')
    if positive.empty:
        raise ValueError('No positive ARM6 target')
    m = measures.copy()
    if not {
        'NeuronUID',
        'TargetID',
        'Level',
        'AxonTemplateLengthMm',
        'CandidateEndpointCount',
        'EndpointPresence',
    }.issubset(m):
        raise ValueError('Required measure columns missing')
    if not set(m.NeuronUID).issubset(summary.index):
        raise ValueError('Foreign source identity in measured rows')
    if not set(m.TargetID).issubset(t.TargetID):
        raise ValueError('Unknown target in measured rows')
    if m.duplicated(['NeuronUID', 'TargetID']).any():
        raise ValueError('Duplicate neuron-target measure')
    levels = pd.to_numeric(m.Level, errors='raise')
    if not np.isfinite(levels).all() or not np.equal(levels, np.floor(levels)).all():
        raise ValueError('Measure hierarchy levels must be integers')
    m['Level'] = levels.astype(int)
    for field in ['SampleID', 'NeuronID', 'AnimalID']:
        if field in m:
            expected = summary.loc[m.NeuronUID, field].astype(str).to_numpy()
            if not np.array_equal(m[field].astype(str).to_numpy(), expected):
                raise ValueError('Long-row identity metadata mismatch: ' + field)
    if 'EndpointEligible' in m:
        expected = bools(summary.loc[m.NeuronUID, 'EndpointEligible'])
        if not np.array_equal(bools(m.EndpointEligible), expected):
            raise ValueError('Long-row eligibility differs from summary')
    target_levels = t.set_index('TargetID').Level
    if not np.array_equal(m.Level.to_numpy(), target_levels.loc[m.TargetID].to_numpy()):
        raise ValueError('Measure/target hierarchy mismatch')
    m = m[m.Level == 6].copy()
    uids = summary.index
    columns = l6.TargetID.tolist()
    elig = bools(summary.EndpointEligible)
    axon = pd.DataFrame(0.0, index=uids, columns=columns)
    ends = pd.DataFrame(0.0, index=uids, columns=columns)
    ends.loc[~elig, :] = np.nan
    for row in m.itertuples(index=False):
        length = float(row.AxonTemplateLengthMm)
        if not np.isfinite(length) or length < 0:
            raise ValueError('Invalid measured axon length')
        axon.at[row.NeuronUID, row.TargetID] = length
        e = bool(elig[uids.get_loc(row.NeuronUID)])
        value = pd.to_numeric(pd.Series([row.CandidateEndpointCount]), errors='coerce').iloc[0]
        presence = pd.to_numeric(pd.Series([row.EndpointPresence]), errors='coerce').iloc[0]
        if not e:
            if pd.notna(value) or pd.notna(presence):
                raise ValueError('Ineligible endpoint measurement must be NA, not zero')
        else:
            if pd.isna(value) or value < 0 or value != np.floor(value) or (presence != int(value > 0)):
                raise ValueError('Invalid candidate count/presence or missing eligible measurement')
            ends.at[row.NeuronUID, row.TargetID] = value
    poscols = positive.TargetID.tolist()
    axon_total = axon.sum(axis=1)
    end_total = ends.sum(axis=1, min_count=1)
    ledger = summary.copy()
    ledger['AxonLengthAllStatusesMm'] = axon_total
    ledger['AxonLengthMappedL6Mm'] = axon[poscols].sum(axis=1)
    ledger['AxonMappedFraction'] = ledger.AxonLengthMappedL6Mm / axon_total.replace(0, np.nan)
    ledger['CandidateEndsAllStatuses'] = end_total
    ledger['CandidateEndsMappedL6'] = ends[poscols].sum(axis=1, min_count=1)
    ledger['CandidateEndsMappedFraction'] = ledger.CandidateEndsMappedL6 / end_total.replace(
        0,
        np.nan,
    )
    for (index, label) in [(0, 'AtlasBackground'), (-1, 'OutOfFOV')]:
        cols = l6.loc[l6.ARMIndex == index, 'TargetID'].tolist()
        ledger['Axon' + label + 'LengthMm'] = axon[cols].sum(axis=1)
        ledger['CandidateEnds' + label] = ends[cols].sum(
            axis=1,
            min_count=1,
        ) if cols else np.where(elig, 0.0, np.nan)
    ledger['AxonProfileEligible'] = ledger.AxonLengthMappedL6Mm > 0
    ledger['EndpointProfileEligible'] = elig & (ledger.CandidateEndsMappedL6 > 0)
    ledger['AxonProfileStatus'] = np.where(
        ledger.AxonProfileEligible,
        'measured_positive_mapped_profile',
        np.where(
            ledger.AxonLengthAllStatusesMm > 0,
            'positive_axon_only_unassigned_or_outside_unassessed_profile',
            'no_positive_type2_axon_length_unassessed_profile',
        ),
    )
    ledger['EndpointProfileStatus'] = np.where(
        ~elig,
        'no_eligible_candidate_ends_unassessed',
        np.where(
            ledger.EndpointProfileEligible,
            'eligible_positive_mapped_profile',
            'eligible_but_no_mapped_target_ends_unassessed_profile',
        ),
    )
    return (ledger, positive.set_index('TargetID'), axon[poscols], ends[poscols], axon, ends)

def profile_distance(raw, representation='hellinger', method='average'):
    if method != 'average':
        raise ValueError('This profile workflow declares average linkage; Ward not accepted implicitly')
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2 or len(a) < 3 or (not np.isfinite(a).all()) or (a < 0).any() or (a.sum(axis=1) <= 0).any():
        raise ValueError('Profiles require finite nonnegative values and positive measured row totals')
    proportions = a / a.sum(axis=1, keepdims=True)
    if representation == 'hellinger':
        features = np.sqrt(proportions)
        dist = squareform(pdist(features)) / np.sqrt(2.0)
    elif representation == 'logrelative':
        features = np.log1p(1000.0 * proportions)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        dist = squareform(pdist(features))
    elif representation == 'presence_jaccard':
        features = (a > 0).astype(float)
        dist = squareform(pdist(features, metric='jaccard'))
    else:
        raise ValueError('Unknown representation')
    validate_distance_matrix(dist)
    return (features, dist, proportions)

def stratified_subsets(groups, repeats=100, fraction=0.8, seed=42, balanced=False):
    """Without replacement, retaining each animal; capped resamples expose dominance."""
    groups = np.asarray(groups).astype(str)
    if repeats < 1 or not 0 < fraction < 1 or len(groups) < 3:
        raise ValueError('Invalid resampling parameters')
    rng = np.random.default_rng(seed)
    levels = sorted(set(groups))
    by = [np.flatnonzero(groups == g) for g in levels]
    cap = min((max(1, int(np.floor(len(ix) * fraction))) for ix in by))
    result = []
    for _ in range(repeats):
        subset = np.sort(np.concatenate([rng.choice(
            ix,
            cap if balanced else max(1, int(np.floor(len(ix) * fraction))),
            replace=False,
        ) for ix in by]))
        result.append(subset)
    return result

def conditional_resampling(dist, groups, labels_by_k, repeats, fraction, seed):
    result = []
    cluster_rows = []
    for balanced in (False, True):
        scheme = 'equal_animal_cap' if balanced else 'within_animal_stratified'
        if balanced and len(set(groups)) == 1:
            for k in labels_by_k:
                result.append({'requested_k': k, 'scheme': scheme, 'evaluated_repeats': 0,
                               'subsample_n': np.nan, 'ari_mean': np.nan, 'ari_p05': np.nan,
                               'status': 'one_animal_cap_not_additional_sensitivity',
                               'n_animals': 1, 'animal_counts_per_subsample': '{}',
                               'source_retention_fraction': np.nan})
            continue
        samples = stratified_subsets(groups, repeats, fraction, seed, balanced)
        trees = [safe_linkage(dist[np.ix_(ix, ix)]) if len(ix) >= 3 else None for ix in samples]
        for (k, labels) in labels_by_k.items():
            aris = []
            overlaps = {int(c): [] for c in np.unique(labels)}
            for (ix, tree) in zip(samples, trees):
                if tree is None or len(ix) <= k:
                    continue
                sub = fcluster(tree, t=k, criterion='maxclust')
                aris.append(adjusted_rand_score(labels[ix], sub))
                for c in np.unique(labels[ix]):
                    mask = labels[ix] == c
                    if mask.sum() < 2:
                        continue
                    overlaps[int(c)].append(max((float(np.logical_and(
                        mask,
                        sub == other,
                    ).sum() / np.logical_or(mask, sub == other).sum()) for other in np.unique(sub))))
            result.append({
                'requested_k': k,
                'scheme': scheme,
                'evaluated_repeats': len(aris),
                'subsample_n': len(samples[0]),
                'ari_mean': float(np.mean(aris)) if aris else np.nan,
                'ari_p05': float(np.quantile(aris, 0.05)) if aris else np.nan,
                'status': 'evaluated' if aris else 'insufficient_equal_animal_sample',
                'n_animals': len(set(groups)),
                'animal_counts_per_subsample': json.dumps(dict(Counter(np.asarray(groups).astype(str)[samples[0]]))),
                'source_retention_fraction': len(samples[0]) / len(groups),
            })
            for (c, scores) in overlaps.items():
                cluster_rows.append({
                    'requested_k': k,
                    'scheme': scheme,
                    'cluster': c,
                    'jaccard_mean': np.mean(scores) if scores else np.nan,
                    'jaccard_p05': np.quantile(scores, 0.05) if scores else np.nan,
                    'assessed_repeats': len(scores),
                })
    return (pd.DataFrame(result), pd.DataFrame(cluster_rows))

def cut_choice(diagnostics):
    admissible = diagnostics[(diagnostics.smallest_cluster >= 5) & (diagnostics.largest_cluster_fraction <= 0.9) & (diagnostics.realized_k == diagnostics.requested_k) & diagnostics.silhouette.notna()]
    status = 'balanced_size_diagnostic_cut' if len(admissible) else 'no_balanced_size_cut_silhouette_only_for_display'
    frame = admissible if len(admissible) else diagnostics[diagnostics.silhouette.notna()]
    if frame.empty:
        return (None, 'no_separated_partition')
    row = frame.sort_values(['silhouette', 'requested_k'], ascending=[False, True]).iloc[0]
    return (int(row.requested_k), status)

def run_analysis(
    raw,
    metadata,
    targets,
    out,
    k_values,
    repeats,
    fraction,
    seed,
    representation,
    source_scope,
):
    """Fit descriptive candidates; return compact tables without duplicate matrices."""
    (features, distances, proportions) = profile_distance(raw, representation)
    matrix = pd.DataFrame(distances, index=raw.index, columns=raw.index)
    subset_size = max(3, int(np.floor(len(raw) * fraction)))
    candidate_counts = [k for k in k_values if k < subset_size]
    if not candidate_counts:
        raise ValueError('Too few profiles for requested candidate cuts')
    animals = metadata.AnimalID.astype(str)
    (
        diagnostics,
        labels,
        stability,
        holdouts,
    ) = evaluate_k(matrix, candidate_counts, repeats=repeats, fraction=fraction, seed=seed, groups=animals)
    for covariate in ['AnimalID', 'SampleID', 'Hemisphere', 'EvidenceSourceGroup', 'ARMFullName']:
        if covariate in metadata:
            diagnostics[covariate + '_cluster_NMI'] = [normalized_mutual_info_score(
                metadata[covariate].astype(str),
                labels[k],
            ) for k in diagnostics.requested_k]
    (
        animal_stability,
        cluster_resamples,
    ) = conditional_resampling(distances, animals, labels, repeats, fraction, seed)
    (display_k, display_status) = cut_choice(diagnostics)
    assignments = pd.DataFrame({'NeuronUID': raw.index})
    (target_profiles, composition, quality, stability_tables) = ([], [], [], [])
    for (k, partition) in labels.items():
        assignments['k' + str(k)] = partition
        stable = stability[k].copy()
        stable['requested_k'] = k
        stable['scheme'] = 'uniform_source_subset'
        stability_tables.append(stable)
        if k != display_k:
            continue
        for cluster in sorted(set(partition)):
            member_mask = partition == cluster
            members = metadata.loc[member_mask]
            relative = pd.DataFrame(
                proportions[member_mask],
                index=members.index,
                columns=raw.columns,
            )
            mean_neuron = relative.mean(axis=0)
            mean_animal = relative.assign(AnimalID=members.AnimalID.to_numpy()).groupby('AnimalID').mean().mean(axis=0)
            for target in raw.columns:
                info = targets.loc[target]
                target_profiles.append({
                    'requested_k': k,
                    'cluster': int(cluster),
                    'TargetID': target,
                    'ARMIndex': info.ARMIndex,
                    'ARMFullName': info.OfficialFullName,
                    'TargetHemisphere': info.Hemisphere,
                    'n_neurons': int(member_mask.sum()),
                    'n_animals': members.AnimalID.nunique(),
                    'mean_neuron_relative_profile': mean_neuron[target],
                    'equal_contributing_animal_relative_profile': mean_animal[target],
                    'target_presence_fraction': float((raw.loc[member_mask, target] > 0).mean()),
                })
            categorical = [
                'AnimalID',
                'SampleID',
                'Hemisphere',
                'EvidenceSourceGroup',
                'SourceCohort',
                'ARMFullName',
                'Henry_coarse_INS_visual_evidence',
                'EndpointProfileStatus',
            ]
            for variable in categorical:
                if variable not in members:
                    continue
                for (
                    value,
                    count,
                ) in members[variable].fillna('unassessed').astype(str).value_counts().items():
                    composition.append({
                        'requested_k': k,
                        'cluster': int(cluster),
                        'variable': variable,
                        'value': value,
                        'count': int(count),
                        'fraction': count / len(members),
                    })
            for variable in metadata.select_dtypes(include='number').columns:
                values = pd.to_numeric(members[variable], errors='coerce')
                finite = values[np.isfinite(values)]
                if len(finite):
                    quality.append({
                        'requested_k': k,
                        'cluster': int(cluster),
                        'variable': variable,
                        'n_assessed': len(finite),
                        'median': finite.median(),
                        'q25': finite.quantile(0.25),
                        'q75': finite.quantile(0.75),
                    })
    summary = {
        'scope': source_scope,
        'representation': representation,
        'n_profiles': len(raw),
        'n_animals': animals.nunique(),
        'n_targets': raw.shape[1],
        'display_cut_k': display_k,
        'display_cut_status': display_status,
        'animal_replication_status': 'one_animal_only_no_holdout_prediction' if animals.nunique() == 1 else 'remaining_cohort_holdout_sensitivity_not_prediction',
        'linkage': 'average',
        'k_values': candidate_counts,
        'fit_weighting': 'one row per neuron; fitted partition is neuron-weighted',
    }
    tables = {
        'candidate_k_diagnostics': diagnostics,
        'animal_aware_resampling': animal_stability,
        'animal_aware_cluster_jaccard': cluster_resamples,
        'uniform_source_cluster_stability': pd.concat(stability_tables, ignore_index=True),
        'leave_one_animal_out': holdouts,
        'cluster_target_profiles': pd.DataFrame(target_profiles),
        'cluster_source_animal_composition': pd.DataFrame(composition),
        'cluster_QC_covariates': pd.DataFrame(quality),
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
        assignments.to_csv(out / 'candidate_cluster_assignments.csv', index=False)
        for name in ['candidate_k_diagnostics', 'animal_aware_resampling', 'leave_one_animal_out']:
            tables[name].to_csv(out / (name + '.csv'), index=False)
        (out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {
        'summary': summary,
        'labels': labels,
        'diagnostics': diagnostics,
        'metadata': metadata,
        'raw': raw,
        'profiles': tables['cluster_target_profiles'],
        'animal_stability': animal_stability,
        'tables': tables,
    }

def validate_completed_export(export_dir, provenance_path, manifest_path):
    """Require completed producer status and exact named artifact/hash bindings."""
    producer = json.loads(provenance_path.read_text(encoding='utf-8'))
    if producer.get('status') != 'software_verified_descriptive_tables':
        raise ValueError('Producer export is not completed/software verified')
    if digest(manifest_path) != producer.get('manifest_sha256'):
        raise ValueError('Exporter does not bind the exact selected manifest')
    for name in ['neuron_summary.csv', 'targets.csv', 'per_neuron_regional_measures.csv']:
        if digest(export_dir / name) != producer.get('artifacts', {}).get(name):
            raise ValueError('Exporter artifact hash mismatch: ' + name)
    return producer

def create_figures(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = [(
        name,
        r,
    ) for (name, r) in results.items() if name in ['all_axon_hellinger', 'all_endpoint_hellinger']]
    (fig, axs) = plt.subplots(1, len(primary), figsize=(6 * len(primary), 4), squeeze=False)
    for (ax, (name, r)) in zip(axs[0], primary):
        d = r['diagnostics']
        ax.plot(d.requested_k, d.silhouette, 'o-', label='Silhouette')
        s = r['animal_stability']
        s = s[s.scheme == 'within_animal_stratified']
        ax.plot(s.requested_k, s.ari_mean, 's-', label='Within-animal subset ARI')
        ax.set(
            xlabel='Requested candidate cluster count',
            ylabel='Diagnostic value',
            title='Axon-labelled length profiles' if 'axon' in name else 'Candidate axon ends',
        )
        ax.legend(fontsize=8)
        displayed = np.r_[d.silhouette.to_numpy(), s.ari_mean.to_numpy()]
        finite = displayed[np.isfinite(displayed)]
        lower = min(-0.1, float(finite.min()) - 0.05) if len(finite) else -0.1
        ax.set_ylim(lower, 1.05)
    fig.tight_layout()
    fig.savefig(out / 'candidate_cut_diagnostics.png', dpi=160)
    plt.close(fig)
    for (name, r) in primary:
        k = r['summary']['display_cut_k']
        if k is None:
            continue
        p = r['profiles']
        p = p[p.requested_k == k]
        table = p.pivot(
            index='cluster',
            columns='TargetID',
            values='equal_contributing_animal_relative_profile',
        )
        top = table.mean().nlargest(min(18, len(table.columns))).index
        names = p.drop_duplicates('TargetID').set_index('TargetID').ARMFullName
        (fig, ax) = plt.subplots(figsize=(12, 2 + 0.5 * len(table)))
        im = ax.imshow(table[top], aspect='auto', cmap='magma', vmin=0)
        ax.set_xticks(range(len(top)), [names[x] for x in top], rotation=65, ha='right', fontsize=7)
        ax.set_yticks(range(len(table)), ['Candidate cluster ' + str(c) for c in table.index])
        ax.set_title(('Axon-labelled length' if 'axon' in name else 'Candidate axon ends') + f': descriptive k={k} target profiles')
        fig.colorbar(im, ax=ax, label='Mean relative profile, equal contributing animals')
        fig.tight_layout()
        fig.savefig(out / (name + '_target_profiles.png'), dpi=160)
        plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--export-provenance', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--k-values', type=int, nargs='+', default=list(range(2, 9)))
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.relative_to(ROOT / 'notes/region_analysis_review_20261009/clustering_20261009')
    if output.exists():
        raise FileExistsError(output)
    if args.repeats < 1 or not 0 < args.fraction < 1 or any((k < 2 for k in args.k_values)) or (len(set(args.k_values)) != len(args.k_values)):
        raise ValueError('Invalid clustering parameters')
    producer = validate_completed_export(args.export_dir, args.export_provenance, args.manifest)
    input_paths = [
        args.manifest,
        args.export_provenance,
    ] + [args.export_dir / name for name in ['neuron_summary.csv', 'targets.csv', 'per_neuron_regional_measures.csv']]
    input_hashes = {str(path.resolve()): digest(path) for path in input_paths}

    def load(name):
        return pd.read_csv(args.export_dir / name, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    if len(manifest) != 462:
        raise ValueError('Expected exact selected462 ledger; do not expand a cohort')
    (
        ledger,
        targets,
        axon,
        ends,
        _,
        _,
    ) = build_profiles(load('neuron_summary.csv'), load('targets.csv'), load('per_neuron_regional_measures.csv'), manifest)
    source_hashes = {}
    for source in manifest.itertuples(index=False):
        if digest(source.SWCPath) != source.SWCSHA256:
            raise ValueError('Selected source hash mismatch: ' + source.NeuronID)
        source_hashes[str(Path(source.SWCPath).resolve())] = source.SWCSHA256
    for column in ledger.columns:
        if any((word in column.lower() for word in [
            'count',
            'length',
            'fraction',
            'node',
            'eligible',
        ])):
            numeric = pd.to_numeric(ledger[column], errors='coerce')
            nonempty = (ledger[column].astype(str) != '').sum()
            if numeric.notna().any() and numeric.notna().sum() == nonempty:
                ledger[column] = numeric
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    families = [
        ('axon', axon, ledger.AxonProfileEligible.astype(bool)),
        ('endpoint', ends, ledger.EndpointProfileEligible.astype(bool)),
    ]
    for (family, raw, eligible) in families:
        scopes = [
            ('all', pd.Series(True, index=ledger.index)),
            (
                'Henry_only',
                pd.Series(bools(ledger.Henry_coarse_INS_visual_evidence), index=ledger.index),
            ),
        ]
        for (scope, scope_mask) in scopes:
            selected = eligible & scope_mask
            modes = ['hellinger', 'logrelative'] if scope == 'all' else ['hellinger']
            if family == 'endpoint' and scope == 'all':
                modes.append('presence_jaccard')
            for mode in modes:
                name = f'{scope}_{family}_{mode}'
                result = run_analysis(
                    raw.loc[selected],
                    ledger.loc[selected],
                    targets,
                    None,
                    args.k_values,
                    args.repeats,
                    args.fraction,
                    args.seed,
                    mode,
                    scope,
                )
                results[name] = result
                for (k, labels) in result['labels'].items():
                    ledger.loc[selected, name + '_k' + str(k)] = labels
    ledger.to_csv(
        output / 'all462_feature_eligibility_and_assignments.csv',
        index_label='NeuronUID',
    )
    targets.to_csv(output / 'exclusive_ARM6_feature_dictionary.csv', index_label='TargetID')
    ledger.groupby(
        ['AnimalID', 'SampleID'],
        dropna=False,
    ).size().reset_index(name='n_selected').to_csv(output / 'animal_sample_crosswalk.csv', index=False)
    for table_name in next(iter(results.values()))['tables']:
        combined = []
        for (analysis, result) in results.items():
            frame = result['tables'][table_name].copy()
            frame.insert(0, 'analysis', analysis)
            combined.append(frame)
        pd.concat(combined, ignore_index=True).to_csv(output / (table_name + '.csv'), index=False)
    sensitivity = []
    for family in ['axon', 'endpoint']:
        primary = results[f'all_{family}_hellinger']
        for (name, result) in results.items():
            if f'_{family}_' not in name or name == f'all_{family}_hellinger':
                continue
            common = primary['metadata'].index.intersection(result['metadata'].index)
            for k in sorted(set(primary['labels']) & set(result['labels'])):
                a = pd.Series(primary['labels'][k], index=primary['metadata'].index).loc[common]
                b = pd.Series(result['labels'][k], index=result['metadata'].index).loc[common]
                sensitivity.append({
                    'family': family,
                    'comparison': name,
                    'requested_k': k,
                    'n_exact_common_UIDs': len(common),
                    'partition_ARI': adjusted_rand_score(a, b),
                    'meaning': 'same-source representation/subset sensitivity; no biological acceptance',
                })
    pd.DataFrame(sensitivity).to_csv(
        output / 'representation_and_Henry_sensitivity.csv',
        index=False,
    )
    create_figures(results, output)
    for (path, expected) in {**input_hashes, **source_hashes}.items():
        if digest(path) != expected:
            raise ValueError('Protected input/source changed during analysis: ' + path)
    methods = [
        ROOT / 'group_analysis/evolution_20261008/references/scientific_basis_20261009.md',
        ROOT / 'group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md',
        ROOT / 'notes/clustering_review_20261002/README.md',
        ROOT / 'notes/clustering_review_20261002/validation_summary.json',
        ROOT / 'main_scripts/clustering_validation.py',
    ]
    report = {
        'status': 'exploratory_profile_clustering_completed_software_only',
        'generated_at': datetime.now().astimezone().isoformat(),
        'input_hashes': input_hashes,
        'selected_source_hashes': source_hashes,
        'code_sha256': digest(__file__),
        'methods_and_baseline_hashes': {str(path.relative_to(ROOT)): digest(path) for path in methods},
        'versions': {
            'python': platform.python_version(),
            **{name: importlib.metadata.version(name) for name in [
                'numpy',
                'pandas',
                'scipy',
                'scikit-learn',
                'matplotlib',
            ]},
        },
        'parameters': {
            'k_values': args.k_values,
            'repeats': args.repeats,
            'fraction': args.fraction,
            'seed': args.seed,
            'linkage': 'average',
            'primary': 'sqrt(relative mapped positive ARM6 profile), Hellinger distance',
            'log_sensitivity': 'log1p(1000*relative profile), row L2 normalized Euclidean',
            'end_presence_sensitivity': 'Jaccard of eligible mapped target presence',
            'target_hemisphere': 'absolute atlas side retained; source hemisphere composition reported',
            'target_standardization': 'none; no cross-target zscore or mixed hierarchy levels',
            'resampling': 'uniform source subsets, within-animal80percent and equal-animal-cap subsets without replacement; remaining-cohort animal holdouts',
        },
        'ledger_neurons': len(ledger),
        'Henry_visual_neurons': int(bools(ledger.Henry_coarse_INS_visual_evidence).sum()),
        'analyses': {name: result['summary'] for (name, result) in results.items()},
        'source_coverage_limit': 'Exact selected462, not every8746inventory identity or1912cached root; no new cohort',
        'scientific_limits': [
            'candidate axon ends are not reviewed terminal arbors, boutons or synapses',
            'template-space child-type2 edge length is not calibrated native length',
            'mapped-target composition is conditional; background/outside coverage retained',
            'normalization removes total extent and can amplify incomplete/low-coverage traces',
            'fitted partition is neuron-weighted; animal-cap stability does not rebalance the fit',
            'one animal supplies all260Henry cases; no across-animal confirmation',
            'sample/channel and animal are confounded acquisition blocks here',
            'stable partitions need not be discrete biological classes',
            'original coordinate/export/registration acceptance remains pending',
        ],
        'network_requests': 0,
        'canonical_changes': False,
        'anatomical_acceptance': False,
        'statistical_p_values': False,
        'output_hashes': {str(path.relative_to(output)): digest(path) for path in output.rglob('*') if path.is_file()},
    }
    with (output / 'run_provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'output': str(output), 'analyses': report['analyses']}, indent=2))
    return report
if __name__ == '__main__':
    main()
