"""Read-only numerical receipts for the 2026-10-09 methods/consumer audit.

Writes only alongside this script. Does not run R, refit published inference,
or alter source workbooks, historical outputs, cohorts, labels or maps.
"""
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    path = ROOT / path
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


workbook = ROOT / "group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx"
stats = ROOT / "group_analysis/R_analysis/outputs/combined_primary_v2/stats"
s = pd.read_excel(workbook, sheet_name="Summary", dtype={"SampleID": str}).set_index("NeuronUID")
meta = ["SampleID", "NeuronID", "Neuron_Type"]
tables = {}
for name in ("Projection_Strength_ipsi", "Projection_Strength_L3_ipsi", "Projection_Length_ipsi"):
    tables[name] = pd.read_excel(workbook, sheet_name=name).set_index("NeuronUID").drop(columns=meta).fillna(0)
strength, length = tables["Projection_Strength_ipsi"], tables["Projection_Length_ipsi"]
common = s.index.intersection(strength.index).intersection(tables["Projection_Strength_L3_ipsi"].index)
side = s.Soma_Side_Final.where(s.Soma_Side_Final.isin(["L", "R"]), s.Soma_Side)
current = s.loc[common[side.loc[common].isin(["L", "R"])]]
old = pd.read_csv(stats / "05_per_neuron_ibias.csv", dtype={"SampleID": str})
old_ids = old.NeuronUID.tolist()
insula = ["Ial", "Iai", "Iapl", "Iam/Iapm", "Ia/Id", "Ig", "Pi", "Ri"]
log_profile = strength.div(strength.sum(axis=1), axis=0).fillna(0)
length_profile = length.div(length.sum(axis=1), axis=0).fillna(0)
per_neuron = pd.DataFrame({
    "NeuronUID": old_ids,
    "normalized_log_strength_intra_INS_share": log_profile.loc[old_ids, insula].sum(axis=1).to_numpy(),
    "legacy_raw_voxel_length_intra_INS_share": length_profile.loc[old_ids, insula].sum(axis=1).to_numpy(),
})
per_neuron.to_csv(OUT / "historical306_logstrength_vs_legacy_voxellength.csv", index=False)
regressions = []
saved_regressions = pd.read_csv(stats / "04_interoceptive_gradient_regressions.csv")
for group in ("L", "R", "all"):
    ids = [uid for uid in old_ids if group == "all" or side.loc[uid] == group]
    fit = linregress(s.loc[ids, "Soma_NII_Y"], log_profile.loc[ids, "Ig"])
    saved = saved_regressions[(saved_regressions.target == "Ig") & (saved_regressions.side == group)].iloc[0]
    regressions.append({"target": "Ig", "side": group, "n": len(ids),
        "independent_OLS_slope": float(fit.slope), "saved_slope": float(saved.slope),
        "independent_OLS_t_p": float(fit.pvalue), "saved_slope_p": float(saved.slope_p),
        "slope_matches": bool(np.isclose(fit.slope, saved.slope, rtol=1e-12, atol=1e-16)),
        "p_matches": bool(np.isclose(fit.pvalue, saved.slope_p, rtol=1e-12, atol=0))})

key_path = ROOT / "atlas/ARM_key_all.txt"
key = pd.read_csv(key_path, sep="\t")
atlas_path = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
atlas = nib.load(atlas_path)
level3 = np.asarray(atlas.dataobj[:, :, :, 0, 2])
level6 = np.asarray(atlas.dataobj[:, :, :, 0, 5])
labels = dict(zip(key.Index, key.Abbreviation))
overlap = []
for target in insula:
    ids = key[key.Abbreviation.map(lambda x: re.sub(r"^[CS][LR]_", "", x) == target)].Index.tolist()
    mask = np.isin(level6, ids)
    parent, counts = np.unique(level3[mask], return_counts=True)
    overlap.append({"stripped_L6_target": target, "actual_label_ids": ids, "voxel_count": int(mask.sum()),
        "actual_level3_footprints": [{"label": labels.get(int(p), "UNKNOWN"), "voxels": int(n)}
                                     for p, n in zip(parent, counts)]})

fnt_path = ROOT / "group_analysis/fnt/multi_monkey_INS_dist.txt"
fnt = pd.read_csv(fnt_path, sep="\t")
indices = sorted(set(fnt.I) | set(fnt.J))
pair_sets = {(min(i, j), max(i, j)) for i, j in zip(fnt.I, fnt.J)}
joined = ROOT / "group_analysis/fnt/multi_monkey_INS_joined.fnt"
names = re.findall(r"^\d+\s+Neuron\s+(\S+)\s*$", joined.read_text(), re.M)
coverage = []
for folder in (ROOT / "R_analysis", ROOT / "group_analysis/R_analysis"):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".r", ".rmd"):
            text = path.read_text(encoding="utf-8-sig")
            coverage.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                "line_count": len(text.splitlines()), "coverage": "full-text static scan; not R execution",
                "test_tokens": sorted(set(re.findall(r"wilcox\.test|fisher\.test|adonis2|mantel|p\.adjust|lm\(", text))),
                "join_tokens": sorted(set(re.findall(r"left_join|inner_join|full_join|merge\(", text)))})

source_evidence = json.loads((ROOT / "group_analysis/evolution_20261008/references/fmost_full_methods_evidence_20261009.json").read_text())
fulltexts = []


def inspect_fulltext(value):
    if isinstance(value, dict):
        if "cache_text_path" in value:
            path = Path(value["cache_text_path"])
            if path.exists():
                expected = value.get("cached_text_bytes_sha256")
                content = path.read_text(encoding="utf-8")
                fulltexts.append({"attachment": value.get("zotero_attachment_key"), "temporary_path": str(path),
                    "bytes_sha256": digest(path), "recorded_hash_matches": expected == digest(path),
                    "parsed_line_count": len(content.splitlines()), "fulltext_copied_to_repository": False})
        for child in value.values():
            inspect_fulltext(child)
    elif isinstance(value, list):
        for child in value:
            inspect_fulltext(child)


inspect_fulltext(source_evidence)
files = ["group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx",
    "group_analysis/R_analysis/v2_combined_primary_pipeline.R", "group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd",
    "group_analysis/R_analysis/outputs/combined_primary_v2/README.md",
    "group_analysis/R_analysis/outputs/combined_primary_v2/stats/05_per_neuron_ibias.csv",
    "group_analysis/R_analysis/outputs/combined_primary_v2/stats/04_interoceptive_gradient_regressions.csv",
    "group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md",
    "group_analysis/evolution_20261008/references/fmost_full_methods_evidence_20261009.json",
    "references/analysis-code_gou_etal_2025/monkeyrec/zz1data/src/zz1data.jl",
    "references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl",
    "main_scripts/region_analysis/population.py", "main_scripts/region_analysis/neuron_analysis.py"]
report = {
    "status": "independent_read_only_consumer_methods_audit", "source_mutation": False,
    "R_execution": "not performed; Rscript unavailable on PATH; independent NumPy/Pandas/SciPy receipts",
    "source_files": [receipt(path) for path in files], "coverage": coverage,
    "fulltext_cache_readbacks": fulltexts,
    "current_cohort": {"valid_common_unique_UIDs": len(current),
        "context": current.assign(Side=side.loc[current.index]).groupby(["Soma_Region_Refined", "Side"]).size().reset_index(name="n").to_dict("records"),
        "IDD5_IDM_sample_side_counts": current[current.Soma_Region_Refined.isin(["IDD5", "IDM"])].assign(
            Side=side.loc[current[current.Soma_Region_Refined.isin(["IDD5", "IDM"])].index]).groupby(
                ["SampleID", "Side"]).size().reset_index(name="n").to_dict("records")},
    "historical_cohort": {"saved_UIDs": len(old), "current_only_UIDs": sorted(set(current.index) - set(old_ids)),
        "saved_only_UIDs": sorted(set(old_ids) - set(current.index))},
    "strength_contract": {"formula": "round(log10(legacy_length+1),4)",
        "max_absolute_formula_residual": float(np.max(abs(np.log10(length.to_numpy()+1)-strength.to_numpy()))),
        "historical306_mean_normalized_log_strength_intra_INS_share": float(per_neuron.normalized_log_strength_intra_INS_share.mean()),
        "historical306_mean_legacy_raw_voxel_length_intra_INS_share": float(per_neuron.legacy_raw_voxel_length_intra_INS_share.mean()),
        "biological_status": "neither measure establishes reviewed axonal terminal fields; legacy lengths retain unverified terminal-target, compartment and spatial semantics"},
    "OLS_pvalue_receipts": regressions,
    "hybrid_parent_child_footprint_overlap": overlap,
    "FNT_actual_completeness": {"rows": len(fnt), "indices": len(indices),
        "contiguous_zero_based_indices": indices == list(range(len(indices))),
        "one_row_per_unordered_pair_including_diagonal": len(pair_sets) == len(fnt) == len(indices)*(len(indices)+1)//2,
        "joined_names": len(names), "unique_joined_names": len(set(names)),
        "missing_pair_zero_fill_seen": False, "registration_acceptance": False},
    "receipt_files": ["historical306_logstrength_vs_legacy_voxellength.csv"],
    "limitations": ["Static scan does not establish full R execution correctness", "No new inferential p-values or animal-level model selected",
        "Source-code changes after this snapshot require a fresh hash comparison", "Voxel-footprint overlap is software geometry, not native anatomical acceptance"]}
with (OUT / "independent_consumer_measurements_20261009.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({"coverage_files": len(coverage), "current_UIDs": len(current), "historical_UIDs": len(old),
    "mean_log_share": report["strength_contract"]["historical306_mean_normalized_log_strength_intra_INS_share"],
    "mean_legacy_length_share": report["strength_contract"]["historical306_mean_legacy_raw_voxel_length_intra_INS_share"],
    "all_OLS_receipts_match": all(row["slope_matches"] and row["p_matches"] for row in regressions)}, indent=2))
