"""Independent saved profile/identity/readback; no production clustering import."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    receipt = a.run / 'independent_saved_readback.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    prov = json.loads((a.run / 'run_provenance.json').read_text())
    inputs = prov['input_hashes']
    for (path, h) in {**inputs, **prov['selected_source_hashes']}.items():
        if sha(path) != h:
            raise ValueError('Changed input/source ' + path)
    for (rel, h) in prov['output_hashes'].items():
        if sha(a.run / rel) != h:
            raise ValueError('Changed output ' + rel)

    def read(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    manifest = read(a.manifest)
    uids = (manifest.SampleID + '::' + manifest.NeuronID).tolist()
    summary = read(a.export / 'neuron_summary.csv')
    summary.index = summary.NeuronUID
    ledger = read(a.run / 'all462_feature_eligibility_and_assignments.csv').set_index('NeuronUID')
    targets = read(a.export / 'targets.csv')
    long = read(a.export / 'per_neuron_regional_measures.csv')
    long = long[long.Level == '6']
    if ledger.index.tolist() != uids or len(uids) != 462 or len(set(uids)) != 462:
        raise ValueError('Exact saved identity/order mismatch')
    features = targets[(targets.Level == '6') & (targets.TargetStatus == 'mapped')].TargetID.tolist()
    target_info = targets.set_index('TargetID')
    n = len(uids)
    row = {uid: i for (i, uid) in enumerate(uids)}
    col = {t: i for (i, t) in enumerate(features)}
    x = np.zeros((n, len(features)))
    e = x.copy()
    totx = np.zeros(n)
    tote = np.zeros(n)
    eligible = summary.loc[uids, 'EndpointEligible'].str.lower().eq('true').to_numpy()
    e[~eligible] = np.nan
    tote[~eligible] = np.nan
    for r in long.itertuples(index=False):
        i = row[r.NeuronUID]
        length = float(r.AxonTemplateLengthMm)
        totx[i] += length
        if r.TargetID in col:
            x[i, col[r.TargetID]] = length
        if eligible[i]:
            count = float(r.CandidateEndpointCount)
            tote[i] += count
            if r.TargetID in col:
                e[i, col[r.TargetID]] = count
    for (
        name,
        values,
    ) in [('AxonLengthAllStatusesMm', totx), ('AxonLengthMappedL6Mm', x.sum(axis=1)), ('CandidateEndsAllStatuses', tote), ('CandidateEndsMappedL6', np.where(eligible, np.nansum(e, axis=1), np.nan))]:
        actual = pd.to_numeric(ledger[name].replace('', np.nan)).to_numpy()
        if not np.allclose(values, actual, equal_nan=True, rtol=1e-10, atol=1e-10):
            raise ValueError('Independent profile total mismatch ' + name)
    profile_masks = {'axon': x.sum(axis=1) > 0, 'endpoint': eligible & (np.nansum(e, axis=1) > 0)}
    checked = []
    diagnostics = pd.read_csv(a.run / 'candidate_k_diagnostics.csv')
    for (family, raw) in [('axon', x), ('endpoint', e)]:
        mask = profile_masks[family]
        p = raw[mask] / raw[mask].sum(axis=1, keepdims=True)
        distance = squareform(pdist(np.sqrt(p))) / np.sqrt(2)
        for r in diagnostics[diagnostics.analysis == f'all_{family}_hellinger'].itertuples(index=False):
            column = f'all_{family}_hellinger_k{r.requested_k}'
            labels = pd.to_numeric(ledger.loc[mask, column]).to_numpy()
            realized = len(set(labels))
            if realized != r.realized_k:
                raise ValueError('Realized cut mismatch')
            if realized > 1:
                measured = silhouette_score(distance, labels, metric='precomputed')
                if not np.isclose(measured, r.silhouette, rtol=1e-10, atol=1e-10):
                    raise ValueError('Independent Hellinger silhouette mismatch')
            if ledger.loc[~mask, column].ne('').any():
                raise ValueError('Unassessed source assigned a cluster')
            checked.append({
                'family': family,
                'k': int(r.requested_k),
                'n_eligible': int(mask.sum()),
                'realized_k': realized,
            })
    result = {
        'status': 'independent_saved_profile_readback_passed',
        'selected_UIDs': 462,
        'unique_UIDs': 462,
        'positive_exclusive_L6_targets': len(features),
        'sourceARM0_UIDs': int(manifest.ARMIndex.eq('0').sum()),
        'sourceARM0_axon_eligible': int((manifest.ARMIndex.eq('0').to_numpy() & profile_masks['axon']).sum()),
        'sourceARM0_endpoint_eligible': int((manifest.ARMIndex.eq('0').to_numpy() & profile_masks['endpoint']).sum()),
        'Henry_visual_UIDs': int(summary.loc[
            uids,
            'Henry_coarse_INS_visual_evidence',
        ].str.lower().eq('true').sum()),
        'candidate_end_eligible': int(eligible.sum()),
        'axon_profile_eligible': int(profile_masks['axon'].sum()),
        'endpoint_profile_eligible': int(profile_masks['endpoint'].sum()),
        'independent_Hellinger_cut_checks': checked,
        'source_hashes_reverified': len(prov['selected_source_hashes']),
        'producer_and_saved_outputs_hashes_checked': True,
        'input_manifest_SHA256': sha(a.manifest),
        'run_provenance_SHA256': sha(a.run / 'run_provenance.json'),
        'readback_code_SHA256': sha(__file__),
        'statistical_or_anatomical_acceptance': False,
    }
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
