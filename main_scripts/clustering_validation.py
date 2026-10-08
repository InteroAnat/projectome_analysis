"""Validated FNT dissimilarities and exploratory clustering diagnostics.

Subsampling measures stability of a fixed dissimilarity, not independent
reconstruction accuracy, prediction strength, or proof of discrete cell types.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_distance_matrix(matrix):
    """Reject malformed dissimilarities rather than impute missing pairs."""
    if isinstance(matrix, pd.DataFrame):
        if not matrix.index.is_unique or not matrix.columns.is_unique:
            raise ValueError("Distance labels must be unique")
        if not matrix.index.equals(matrix.columns):
            raise ValueError("Distance row/column identities and order must match")
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 3:
        raise ValueError("Distance matrix must be square with at least 3 neurons")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric")
    if not np.allclose(np.diag(values), 0, rtol=0, atol=1e-12):
        raise ValueError("Distance diagonal must be zero")
    # Only remove numerical roundoff after validation, never repair missing data.
    values = (values + values.T) / 2
    np.fill_diagonal(values, 0)
    return values


def read_fnt_distances(dist_file, joined_fnt_file, symmetrization="max"):
    """Resolve indices from actual joined-FNT markers, never a directory sort.

    Complete triangular and complete directed files are supported. Every
    unordered pair and every self pair must be present. Raw FNT self scores
    are retained in metadata, then explicitly replaced by zero for clustering.
    If both directions are present, the chosen max/mean reduction is recorded.
    """
    if symmetrization not in {"max", "mean"}:
        raise ValueError("symmetrization must be max or mean")
    names = []
    marker = re.compile(r"^\d+\s+Neuron\s+(\S+)\s*$")
    with open(joined_fnt_file, encoding="utf-8-sig") as stream:
        for line in stream:
            match = marker.match(line.strip())
            if match:
                names.append(match.group(1))
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError("Joined FNT must contain at least 3 unique neuron markers")

    frame = pd.read_csv(dist_file, sep="\t", header=None, comment="#")
    if frame.shape[1] < 3:
        raise ValueError("FNT scores require I, J and Score columns")
    if str(frame.iloc[0, 0]).strip().lower() == "i":
        frame = frame.iloc[1:]
    frame = frame.iloc[:, :3].copy()
    frame.columns = ["I", "J", "Score"]
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame.to_numpy(float)
    if not np.isfinite(values).all() or (frame["Score"] < 0).any():
        raise ValueError("FNT indices/scores must be finite; scores nonnegative")
    indices = frame[["I", "J"]].to_numpy(float)
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("FNT indices must be integers")
    if (indices < 0).any() or (indices >= len(names)).any():
        raise ValueError("FNT indices do not match joined-FNT neuron count")
    frame[["I", "J"]] = frame[["I", "J"]].astype(int)
    if frame.duplicated(["I", "J"]).any():
        raise ValueError("Duplicate directed FNT pairs")
    directed = frame.pivot(index="I", columns="J", values="Score").reindex(
        index=range(len(names)), columns=range(len(names))
    ).to_numpy(float)
    available = np.isfinite(directed)
    missing = ~(available | available.T)
    if missing.any():
        pairs = np.argwhere(np.triu(missing))[:5].tolist()
        raise ValueError(f"Missing FNT pairs (including self pairs): {pairs}")
    both = available & available.T
    asymmetry = np.abs(directed - directed.T)[both]
    # Fill only from the observed reverse direction. Neither-direction missing
    # is already an error above, so an absent pair never becomes distance zero.
    forward = np.where(available, directed, directed.T)
    reverse = np.where(available.T, directed.T, directed)
    symmetric = (np.maximum(forward, reverse) if symmetrization == "max"
                 else (forward + reverse) / 2)
    diagonal = np.diag(symmetric).copy()
    np.fill_diagonal(symmetric, 0)
    matrix = pd.DataFrame(symmetric, index=names, columns=names)
    validate_distance_matrix(matrix)
    metadata = {
        "distance_file": str(Path(dist_file).resolve()),
        "distance_sha256": file_sha256(dist_file),
        "joined_fnt_file": str(Path(joined_fnt_file).resolve()),
        "joined_fnt_sha256": file_sha256(joined_fnt_file),
        "n_neurons": len(names), "n_input_pairs": len(frame),
        "symmetrization": symmetrization,
        "max_directional_difference": float(asymmetry.max()),
        "raw_self_score_min": float(diagonal.min()),
        "raw_self_score_max": float(diagonal.max()),
        "diagonal_policy": "raw self scores recorded; clustering diagonal set to zero",
        "identity_source": "joined FNT Neuron markers in file order",
    }
    return matrix, metadata


def safe_linkage(matrix, method="average"):
    """Use linkage methods defined for arbitrary precomputed dissimilarities."""
    if method not in {"average", "complete", "single"}:
        raise ValueError("Use average, complete or single for FNT dissimilarities; "
                         "Ward requires independently verified Euclidean geometry")
    return linkage(squareform(validate_distance_matrix(matrix)), method=method)


def _c_index(values, labels):
    upper = np.triu_indices(len(values), 1)
    distances = values[upper]
    same = labels[upper[0]] == labels[upper[1]]
    count = int(same.sum())
    if count == 0 or count == len(distances):
        return np.nan
    ordered = np.sort(distances)
    low, high = ordered[:count].sum(), ordered[-count:].sum()
    if high == low:
        return np.nan  # A flat distance has no separation to optimize.
    return float((distances[same].sum() - low) / (high - low))


def _jaccard_matches(reference, resampled):
    """Best overlap of each represented reference cluster after subsetting."""
    result = {}
    for cluster in np.unique(reference):
        members = reference == cluster
        if members.sum() < 2:
            continue  # Singleton retention gives uninformative perfect scores.
        result[int(cluster)] = max(
            float(np.logical_and(members, resampled == other).sum()
                  / np.logical_or(members, resampled == other).sum())
            for other in np.unique(resampled)
        )
    return result


def evaluate_k(matrix, k_values, repeats=100, fraction=0.8, seed=42,
               groups=None, method="average"):
    """Report separation, subset stability and leave-one-group-out sensitivity.

    Group holdouts recluster the remaining cohort and compare its labels to the
    full-cohort partition on those same neurons. They do not classify unseen
    animals. No stability cutoff is treated as biological acceptance.
    """
    values = validate_distance_matrix(matrix)
    n = len(values)
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    subset_n = max(3, int(np.floor(n * fraction)))
    if subset_n >= n:
        raise ValueError("Need enough neurons for a proper subsample")
    requested = list(k_values)
    if not requested or any(not isinstance(k, (int, np.integer)) for k in requested):
        raise ValueError("k_values must contain integers")
    if any(k < 2 or k >= subset_n for k in requested):
        raise ValueError(f"Each k must be >=2 and <subsample size {subset_n}")
    if len(set(requested)) != len(requested):
        raise ValueError("k_values must be unique")
    group_values = None
    if groups is not None:
        if isinstance(groups, pd.Series) and isinstance(matrix, pd.DataFrame):
            if not groups.index.equals(matrix.index):
                raise ValueError("Group index must match distance identities/order")
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or len(group_values) != n or pd.isna(group_values).any():
            raise ValueError("Each neuron needs one nonmissing group label")
        group_values = group_values.astype(str)
    full_tree = safe_linkage(values, method)
    labels_by_k = {int(k): fcluster(full_tree, t=k, criterion="maxclust")
                   for k in requested}
    rng = np.random.default_rng(seed)
    samples = [np.sort(rng.choice(n, subset_n, replace=False)) for _ in range(repeats)]
    # Reuse one tree per subset for all candidate cuts.
    trees = [safe_linkage(values[np.ix_(idx, idx)], method) for idx in samples]
    rows, stability, holdouts = [], {}, []
    for k in requested:
        labels = labels_by_k[k]
        cluster_ids, sizes = np.unique(labels, return_counts=True)
        realized = len(cluster_ids)
        ari, overlaps = [], {int(c): [] for c in cluster_ids}
        for idx, tree in zip(samples, trees):
            sampled_labels = fcluster(tree, t=k, criterion="maxclust")
            ari.append(adjusted_rand_score(labels[idx], sampled_labels))
            for cluster, score in _jaccard_matches(labels[idx], sampled_labels).items():
                overlaps[cluster].append(score)
        cluster_rows = []
        for cluster, size in zip(cluster_ids, sizes):
            scores = overlaps[int(cluster)]
            cluster_rows.append({
                "cluster": int(cluster), "n": int(size),
                "jaccard_mean": float(np.mean(scores)) if scores else np.nan,
                "jaccard_p05": float(np.quantile(scores, .05)) if scores else np.nan,
                "n_resamples_observed": len(scores),
            })
        stability[k] = pd.DataFrame(cluster_rows)
        row = {
            "requested_k": k, "realized_k": realized,
            "silhouette": (float(silhouette_score(values, labels, metric="precomputed"))
                           if 1 < realized < n else np.nan),
            "c_index": _c_index(values, labels),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
            "largest_cluster_fraction": float(sizes.max() / n),
            "singleton_clusters": int((sizes == 1).sum()),
            "subsample_ari_mean": float(np.mean(ari)),
            "subsample_ari_p05": float(np.quantile(ari, .05)),
            "worst_cluster_jaccard_mean": stability[k]["jaccard_mean"].min(),
            "jaccard_assessed_clusters": int(stability[k]["jaccard_mean"].notna().sum()),
            "jaccard_unassessed_clusters": int(stability[k]["jaccard_mean"].isna().sum()),
            "n_subsample_repeats": repeats, "subsample_n": subset_n,
        }
        if group_values is not None:
            row["sample_cluster_ari"] = adjusted_rand_score(group_values, labels)
            for group in sorted(set(group_values)):
                idx = np.flatnonzero(group_values != group)
                record = {"requested_k": k, "held_out_group": group,
                          "remaining_n": len(idx), "ari": np.nan,
                          "realized_k": np.nan, "status": "insufficient_remaining_neurons"}
                if len(idx) > k and len(idx) >= 3:
                    sublabels = fcluster(safe_linkage(values[np.ix_(idx, idx)], method),
                                         t=k, criterion="maxclust")
                    record.update(ari=adjusted_rand_score(labels[idx], sublabels),
                                  realized_k=len(np.unique(sublabels)), status="evaluated")
                holdouts.append(record)
        rows.append(row)
    return pd.DataFrame(rows), labels_by_k, stability, pd.DataFrame(holdouts)
