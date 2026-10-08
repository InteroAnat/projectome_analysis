"""Exploratory FNT clustering with explicit identity and stability checks.

Raw FNT dissimilarity with average linkage is the default. Spearman mode
clusters distance-to-cohort profiles and must not be called direct morphology
clustering. Type penalties are an explicit supervised taxonomy option; type
separation in that option cannot independently validate biological enrichment.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

try:
    from .clustering_validation import (
        read_fnt_distances, validate_distance_matrix, safe_linkage, evaluate_k,
    )
except ImportError:
    from clustering_validation import (
        read_fnt_distances, validate_distance_matrix, safe_linkage, evaluate_k,
    )

DIST_FILE = r"D:\projectome_analysis\main_scripts\processed_neurons\251637\fnt_processed\ins\ins_dist.txt"
TYPE_FILE = r"D:\projectome_analysis\main_scripts\neuron_tables\251637_results.xlsx"
FNT_FOLDER = str(Path(DIST_FILE).parent)  # retained for older callers
USE_SPEARMAN = False
USE_PENALTY = False
PENALTY_STRENGTH = 1.5
OUTPUT_SUFFIX = "_clustered"


def _plot_dependencies():
    global plt, sns, mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches


def infer_joined_fnt(dist_file):
    path = Path(dist_file)
    if not path.name.endswith("_dist.txt"):
        raise ValueError("Pass --joined-fnt explicitly when distance file is not named *_dist.txt")
    return path.with_name(path.name[:-len("_dist.txt")] + "_joined.fnt")


def _read_annotations(type_file, sheet_name=None):
    path = Path(type_file)
    if path.suffix.lower() == ".xlsx":
        book = pd.ExcelFile(path)
        sheet = sheet_name or ("Summary" if "Summary" in book.sheet_names else "Projection_Strength_L3")
        if sheet not in book.sheet_names:
            raise ValueError(f"Annotation sheet {sheet!r} missing in {path}; choose --sheet")
        frame = pd.read_excel(book, sheet_name=sheet)
    elif path.suffix.lower() == ".csv":
        sheet = None
        frame = pd.read_csv(path)
    else:
        raise ValueError("Annotation table must be .xlsx or .csv")
    if frame.empty or "Neuron_Type" not in frame.columns:
        raise ValueError("Annotation table needs nonempty rows and Neuron_Type")
    frame.attrs["annotation_sheet"] = sheet
    return frame


def _align_annotations(frame, matrix_names):
    names = list(map(str, matrix_names))
    expected = set(names)
    candidates = []
    for col in ("NeuronUID", "NeuronID"):
        if col in frame:
            candidates.append((col, frame[col].astype("string")))
    if "NeuronID" in frame:
        candidates.append(("NeuronID_stem", frame["NeuronID"].astype("string").str.replace(r"\.swc$", "", regex=True)))
    if {"SampleID", "NeuronID"}.issubset(frame.columns):
        sample = frame["SampleID"].astype("string").str.replace(r"\.0$", "", regex=True)
        neuron = frame["NeuronID"].astype("string").str.replace(r"\.swc$", "", regex=True)
        candidates.append(("SampleID_NeuronID", sample + "_" + neuron))
    for source, identities in candidates:
        if expected.issubset(set(identities.dropna())):
            used = identities.isin(expected)
            if identities[used].duplicated().any():
                raise ValueError(f"Duplicate annotation identities in {source}")
            aligned = frame.loc[used].copy()
            aligned.index = identities[used].astype(str).to_numpy()
            aligned = aligned.loc[names]
            if aligned["Neuron_Type"].isna().any():
                raise ValueError("Missing Neuron_Type for a clustered neuron")
            return aligned, source
    covered = set().union(*(set(v.dropna()) for _, v in candidates)) if candidates else set()
    missing = sorted(expected - covered)
    raise ValueError(f"Missing annotation identities for FNT neurons: {missing[:10]} (no rows silently dropped)")


def load_data(dist_file, type_file, fnt_folder=None, joined_fnt_file=None,
              sheet_name=None, return_metadata=False):
    """Load identity from joined FNT markers, never directory sort or truncation.

    fnt_folder is accepted for legacy callers but is not used to infer order.
    Extra annotation rows are recorded; every distance-matrix neuron is required.
    """
    joined = Path(joined_fnt_file) if joined_fnt_file else infer_joined_fnt(dist_file)
    matrix, metadata = read_fnt_distances(dist_file, joined, symmetrization="max")
    frame = _read_annotations(type_file, sheet_name)
    aligned, identity_source = _align_annotations(frame, matrix.index)
    type_map = aligned["Neuron_Type"].astype(str).to_dict()
    metadata = dict(metadata)
    metadata.update({"annotation_identity_source": identity_source,
                     "annotation_rows": len(frame), "annotation_rows_used": len(aligned),
                     "annotation_extra_rows": len(frame) - len(aligned),
                     "annotation_file": str(Path(type_file).resolve()),
                     "annotation_sheet": frame.attrs["annotation_sheet"]})
    if return_metadata:
        return matrix, type_map, metadata, aligned
    return matrix, type_map


def process_matrix(raw_matrix, type_map, use_spearman=False,
                   use_penalty=False, penalty_strength=1.5, mode=None):
    """Validate input then transform, with an explicit supervised option.

    The legacy true boolean selects profile Spearman. Otherwise raw FNT is
    the default; log1p is available only through an explicit mode selection.
    """
    values = validate_distance_matrix(raw_matrix)
    mode = mode or ("spearman-profile" if use_spearman else "raw")
    if mode == "spearman-profile":
        if np.any(np.ptp(values, axis=1) == 0):
            raise ValueError("Constant FNT distance profile has undefined Spearman correlation")
        ranked = pd.DataFrame(values).rank(axis=1, method="average").to_numpy()
        corr = np.corrcoef(ranked)
        if not np.isfinite(corr).all():
            raise ValueError("Undefined Spearman correlation; profiles cannot be imputed to zero")
        transformed = np.clip(1.0 - corr, 0.0, 2.0)
    elif mode == "log1p":
        transformed = np.log1p(values)
    elif mode == "raw":
        transformed = values.copy()
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    np.fill_diagonal(transformed, 0)
    if use_penalty:
        if not np.isfinite(penalty_strength) or penalty_strength < 0:
            raise ValueError("Penalty strength must be finite and nonnegative")
        missing = [name for name in raw_matrix.index if name not in type_map or pd.isna(type_map[name])]
        if missing:
            raise ValueError(f"Supervised penalty needs every type label: {missing[:10]}")
        types = np.array([type_map[name] for name in raw_matrix.index])
        transformed[types[:, None] != types[None, :]] += transformed.max() * penalty_strength
    result = pd.DataFrame(transformed, index=raw_matrix.index, columns=raw_matrix.columns)
    validate_distance_matrix(result)
    return result


def compute_linkage(dist_matrix, method="average"):
    """Use linkage defined for dissimilarities; reject unsupported Ward geometry."""
    return safe_linkage(dist_matrix, method=method)


def c_index_value(dist_matrix, labels):
    values = validate_distance_matrix(dist_matrix)
    labels = np.asarray(labels)
    if len(labels) != len(values):
        raise ValueError("Cluster labels do not align with distance matrix")
    pairs = values[np.triu_indices(len(values), 1)]
    same = (labels[:, None] == labels[None, :])[np.triu_indices(len(values), 1)]
    n_intra = int(same.sum())
    if n_intra == 0 or n_intra == len(pairs):
        return np.nan
    ordered = np.sort(pairs)
    lower, upper = ordered[:n_intra].sum(), ordered[-n_intra:].sum()
    return np.nan if upper == lower else float((pairs[same].sum() - lower) / (upper - lower))


def c_index_diagnostics(dist_matrix, linkage_matrix, min_k=2, max_k=65):
    """Bound cuts to n-1 and record requested versus realized cluster count."""
    n = len(validate_distance_matrix(dist_matrix))
    if min_k < 2 or max_k < min_k or min_k >= n:
        raise ValueError("Require 2 <= min_k <= max_k and min_k < n")
    rows, observed = [], set()
    for k in range(min_k, min(max_k, n - 1) + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        actual = int(len(np.unique(labels)))
        duplicate = actual in observed
        observed.add(actual)
        rows.append({"requested_k": k, "realized_k": actual,
                     "duplicate_realized_k": duplicate,
                     "c_index": c_index_value(dist_matrix, labels)})
    return pd.DataFrame(rows)


def calculate_c_index(dist_matrix, linkage_matrix, max_k=65):
    """Legacy API: suggest a cut from C-index, without claiming a true subtype k."""
    diagnostics = c_index_diagnostics(dist_matrix, linkage_matrix, max_k=max_k)
    eligible = diagnostics[(diagnostics.realized_k >= 2) & ~diagnostics.duplicate_realized_k]
    eligible = eligible.dropna(subset=["c_index"])
    if eligible.empty:
        raise ValueError("C-index provides no valid cluster suggestion")
    return int(eligible.sort_values(["c_index", "requested_k"]).iloc[0].requested_k)


def assign_clusters_and_save(dist_matrix, linkage_matrix, type_map, k,
                             type_file, use_penalty=False, use_spearman=False,
                             output_suffix="_clustered", output_dir=None,
                             sheet_name=None, mode=None):
    """Export selected rows to a separate workbook; never replace source data."""
    if isinstance(k, bool) or int(k) != k or not 2 <= k < len(dist_matrix):
        raise ValueError("Cluster k must be an integer in [2, n-1]")
    labels = fcluster(linkage_matrix, t=int(k), criterion="maxclust")
    results = pd.DataFrame({"NeuronID": dist_matrix.index,
                            "Bio_Type": [type_map[n] for n in dist_matrix.index],
                            "Morph_Cluster": labels})
    frame, _ = _align_annotations(_read_annotations(type_file, sheet_name), dist_matrix.index)
    new_frame = frame.copy()
    new_frame["Morph_Cluster"] = labels
    if "NeuronID" in new_frame:
        cols = list(new_frame.columns)
        cols.remove("Morph_Cluster")
        cols.insert(cols.index("NeuronID") + 1, "Morph_Cluster")
        new_frame = new_frame[cols]
    mode = mode or ("spearman-profile" if use_spearman else "raw")
    out_dir = Path(output_dir) if output_dir else Path(type_file).parent / "clustering_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(type_file).resolve()
    out = out_dir / f"{source.stem}{output_suffix}_k{k}_{mode}{'_penalty' if use_penalty else ''}.xlsx"
    if out.resolve() == source:
        raise ValueError("Cluster export must not overwrite annotation source")
    new_frame.to_excel(out, index=False)
    results.to_csv(out_dir / "cluster_assignments.csv", index=False)
    return results, new_frame


def _annotate_dendrogram_clusters(g, results, Z):
    """
    Annotate cluster sizes (n=XX) on the row dendrogram.
    Draws a bracket + label at the vertical span of each cluster's leaves.
    """
    ax_dendro = g.ax_row_dendrogram
    reordered_idx = g.dendrogram_row.reordered_ind

    # Map reordered positions to cluster labels
    ordered_labels = results['Morph_Cluster'].values[reordered_idx]

    # Find contiguous runs of each cluster in dendrogram order
    cluster_runs = []
    current_label = ordered_labels[0]
    start = 0
    for i in range(1, len(ordered_labels)):
        if ordered_labels[i] != current_label:
            cluster_runs.append((current_label, start, i - 1))
            current_label = ordered_labels[i]
            start = i
    cluster_runs.append((current_label, start, len(ordered_labels) - 1))

    xlim = ax_dendro.get_xlim()
    x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.03

    n_total = len(ordered_labels)
    base_font = max(5, min(7, 350 // max(n_total, 1)))

    for cid, row_start, row_end in cluster_runs:
        n_neurons = row_end - row_start + 1
        y_bot = row_start * 10 + 5
        y_top = row_end * 10 + 5
        y_mid = (y_bot + y_top) / 2.0

        bracket_x = x_pos + (xlim[1] - xlim[0]) * 0.01
        ax_dendro.plot([bracket_x, bracket_x], [y_bot, y_top],
                       color='black', lw=0.6, clip_on=False)
        ax_dendro.plot([bracket_x, bracket_x + (xlim[1] - xlim[0]) * 0.015],
                       [y_bot, y_bot], color='black', lw=0.6, clip_on=False)
        ax_dendro.plot([bracket_x, bracket_x + (xlim[1] - xlim[0]) * 0.015],
                       [y_top, y_top], color='black', lw=0.6, clip_on=False)

        ax_dendro.text(
            x_pos, y_mid,
            f"C{cid} n={n_neurons}",
            fontsize=base_font, fontweight='bold',
            color='black',
            ha='left', va='center',
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                      edgecolor='none', alpha=0.8)
        )


def plot_heatmap(dist_matrix, Z, results, type_map,
                 use_spearman=True, use_penalty=False, output_dir=None, mode=None):
    """Publication-ready clustermap with cluster-size annotations on dendrogram."""
    _plot_dependencies()
    output_dir = Path(output_dir or "clustering_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Heatmap...")

    neuron_labels = dist_matrix.index
    bio_types = [type_map.get(n, 'Unknown') for n in neuron_labels]
    unique_types = sorted(set(bio_types))

    lut = dict(zip(unique_types, sns.color_palette("tab20", len(unique_types))))
    row_colors = pd.Series(bio_types, index=neuron_labels).map(lut)

    mode = mode or ("spearman-profile" if use_spearman else "raw")
    mode_str = {"spearman-profile": "FNT distance-profile Spearman", "log1p": "Log1p FNT", "raw": "Raw FNT"}[mode]
    pen_str = "+ Penalty" if use_penalty else ""
    cbar_label = {"spearman-profile": "Dissimilarity (1 - profile correlation)", "log1p": "Log1p FNT score", "raw": "FNT score"}[mode]

    n_clusters = results['Morph_Cluster'].nunique()

    g = sns.clustermap(
        dist_matrix,
        row_linkage=Z, col_linkage=Z,
        row_colors=row_colors,
        cmap='mako' if use_spearman else 'viridis_r',
        xticklabels=False, yticklabels=False,
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={'label': cbar_label, 'orientation': 'vertical'},
        figsize=(13, 13),
        rasterized=True
    )

    g.ax_heatmap.set_xlabel("Neurons", fontsize=12, labelpad=10)
    g.ax_heatmap.set_ylabel("Neurons", fontsize=12, labelpad=10)
    g.ax_heatmap.tick_params(axis='both', which='both', length=0, labelsize=0)

    g.fig.suptitle(
        f"{mode_str} {pen_str} | Clusters: {n_clusters}",
        y=0.98, fontsize=16, fontweight='bold'
    )

    _annotate_dendrogram_clusters(g, results, Z)

    handles = [mpatches.Patch(color=lut[t], label=t) for t in unique_types]
    g.fig.legend(
        handles=handles, title="Biological Type",
        loc="center right", bbox_to_anchor=(0.98, 0.8),
        borderaxespad=0., frameon=True,
        fontsize=8, title_fontsize=9,
        handlelength=0.8, handleheight=0.7,
        labelspacing=0.25, handletextpad=0.3,
        ncol=1 if len(unique_types) <= 15 else 2
    )
    plt.savefig(output_dir / "fnt_dist_Clusters.png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_cluster_sizes(results, output_dir=output_dir)


def plot_cluster_sizes(results, output_dir=None):
    """Stacked bar chart showing neuron count per cluster, coloured by biological type."""
    _plot_dependencies()
    output_dir = Path(output_dir or "clustering_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Cluster Size Chart...")

    ct = pd.crosstab(results['Morph_Cluster'], results['Bio_Type'])
    ct = ct.sort_index()

    unique_types = sorted(results['Bio_Type'].unique())
    palette = dict(zip(unique_types, sns.color_palette("tab20", len(unique_types))))

    fig, ax = plt.subplots(figsize=(max(6, len(ct) * 0.45), 5))

    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))

    for bio_type in unique_types:
        vals = ct[bio_type].values if bio_type in ct.columns else np.zeros(len(ct))
        ax.bar(x, vals, bottom=bottom, color=palette[bio_type],
               label=bio_type, edgecolor='white', linewidth=0.4)
        bottom += vals

    totals = ct.sum(axis=1).values
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals) * 0.01, str(total),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in ct.index], fontsize=8, rotation=45, ha='right')
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Neurons", fontsize=12)
    ax.set_title("Neurons per Cluster (by Biological Type)", fontsize=13, fontweight='bold')
    ax.legend(title="Bio Type", bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, title_fontsize=9, frameon=True,
              handlelength=1.0, labelspacing=0.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(totals) * 1.12)

    fig.tight_layout()
    plt.savefig(output_dir / "fnt_dist_Clustersizes.png", dpi=300, bbox_inches="tight")
    plt.show()



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-file", default=DIST_FILE)
    parser.add_argument("--type-file", default=TYPE_FILE)
    parser.add_argument("--joined-fnt")
    parser.add_argument("--sheet")
    parser.add_argument("--mode", choices=["raw", "log1p", "spearman-profile"], default="raw")
    parser.add_argument("--linkage", choices=["average", "complete", "single"], default="average")
    parser.add_argument("--supervised-penalty", action="store_true")
    parser.add_argument("--penalty-strength", type=float, default=PENALTY_STRENGTH)
    parser.add_argument("--k", type=int, help="Explicit exploratory cut; otherwise use diagnostic suggestion")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-col", default="SampleID")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir or Path(args.type_file).parent / "clustering_runs" /
                   datetime.now().strftime("%Y%m%d_%H%M%S"))
    raw, types, provenance, annotations = load_data(
        args.dist_file, args.type_file, joined_fnt_file=args.joined_fnt,
        sheet_name=args.sheet, return_metadata=True)
    distance = process_matrix(raw, types, mode=args.mode,
                              use_penalty=args.supervised_penalty,
                              penalty_strength=args.penalty_strength)
    n = len(distance)
    if not 0 < args.fraction < 1:
        raise ValueError("Resampling fraction must be between zero and one")
    highest = min(n - 1, int(np.floor(n * args.fraction)) - 1, args.max_k)
    if args.min_k < 2 or highest < args.min_k:
        raise ValueError("Need at least two non-singleton resample cluster cuts; lower min-k or increase n/fraction")
    if args.k is not None and not args.min_k <= args.k <= highest:
        raise ValueError(f"Explicit k must be in diagnostic range {args.min_k}..{highest}")
    z = compute_linkage(distance, args.linkage)
    groups = annotations[args.group_col].to_numpy() if args.group_col in annotations else None
    if groups is not None and pd.isna(groups).any():
        raise ValueError(f"Missing {args.group_col} annotations; group holdouts cannot be computed")
    candidates = list(range(args.min_k, highest + 1))
    diagnostics, labels_by_k, stability_by_k, holdouts = evaluate_k(
        distance, candidates, repeats=args.repeats, fraction=args.fraction,
        seed=args.seed, groups=groups, method=args.linkage)
    cdiag = c_index_diagnostics(distance, z, min_k=args.min_k, max_k=highest)
    # Silhouette and C-index are diagnostics, not evidence for discrete cell types.
    silhouette_cols = [c for c in diagnostics if "silhouette" in c.lower() and "null" not in c.lower()]
    if not silhouette_cols:
        raise ValueError("Clustering diagnostics did not return silhouette scores")
    k_col = "requested_k" if "requested_k" in diagnostics else "k"
    available = diagnostics.dropna(subset=[silhouette_cols[0]])
    if "realized_k" in available:
        available = available[available.realized_k >= 2]
    if available.empty:
        raise ValueError("No nondegenerate cut has a finite silhouette; report a continuum")
    suggested = int(available.sort_values([silhouette_cols[0], k_col], ascending=[False, True]).iloc[0][k_col])
    selected = args.k if args.k is not None else suggested
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_export = pd.DataFrame({"FNT_NeuronID": distance.index})
    for k in candidates:
        candidates_export[f"k_{k}"] = labels_by_k[k]
    candidates_export.to_csv(out_dir / "candidate_cluster_assignments.csv", index=False)
    diagnostics.to_csv(out_dir / "clustering_diagnostics.csv", index=False)
    cdiag.to_csv(out_dir / "c_index_diagnostics.csv", index=False)
    holdouts.to_csv(out_dir / "group_holdout_diagnostics.csv", index=False)
    for k, stability in stability_by_k.items():
        stability.to_csv(out_dir / f"cluster_stability_k{k}.csv", index=False)
    results, _ = assign_clusters_and_save(
        distance, z, types, selected, args.type_file, mode=args.mode,
        use_penalty=args.supervised_penalty, output_dir=out_dir, sheet_name=args.sheet)
    if not np.array_equal(results.Morph_Cluster.to_numpy(), labels_by_k[selected]):
        raise ValueError("Exported labels differ from diagnostic linkage")
    metadata = {
        **provenance, "n_neurons": n, "mode": args.mode, "linkage": args.linkage,
        "type_penalty": args.supervised_penalty, "penalty_strength": args.penalty_strength,
        "suggested_k": suggested, "selected_k": selected,
        "suggested_realized_k": int(len(np.unique(labels_by_k[suggested]))),
        "selected_realized_k": int(len(np.unique(labels_by_k[selected]))),
        "selected_k_source": "explicit" if args.k is not None else "silhouette_diagnostic",
        "k_status": "exploratory; no validated subtype count", "seed": args.seed,
        "repeats": args.repeats, "fraction": args.fraction,
        "group_col": args.group_col if groups is not None else None,
        "annotation_sha256": hashlib.sha256(Path(args.type_file).read_bytes()).hexdigest(),
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scipy": scipy.__version__,
                              "scikit_learn": sklearn.__version__},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["fnt_dist_clustering.py", "clustering_validation.py"]},
        "group_holdout_scope": "Recluster remaining neurons after leaving one animal out; not prediction on an unseen animal",
        "stability_scope": "Subsampling a fixed distance matrix; not reconstruction reproducibility or prediction strength",
        "claim_limits": "Type-guided clustering cannot independently test type enrichment" if args.supervised_penalty else
                        "Unsupervised exploratory clusters require biological and animal-level validation",
    }
    (out_dir / "clustering_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Exported exploratory clustering: n={n}, selected k={selected}, suggested k={suggested}, output={out_dir}")
    if not args.no_plots:
        plot_heatmap(distance, z, results, types, mode=args.mode,
                     use_spearman=args.mode == "spearman-profile",
                     use_penalty=args.supervised_penalty, output_dir=out_dir)
    return metadata


if __name__ == "__main__":
    main()
