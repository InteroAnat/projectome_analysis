"""QC report: canonical vs staging harmonized; write staging README.md."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ST = Path(r"D:\projectome_analysis\group_analysis\staging_20260926")
CAN_H = Path(
    r"D:\projectome_analysis\group_analysis\combined"
    r"\multi_monkey_INS_combined_harmonized.xlsx"
)
STG_H = ST / "combined" / "multi_monkey_INS_combined_harmonized.xlsx"
DIFF_SUM = ST / "recovery" / "rescue_rebuild_diff_summary.csv"


def side_col(df: pd.DataFrame) -> pd.Series:
    if "Soma_Side_Final" in df.columns:
        return df["Soma_Side_Final"]
    if "Soma_Side_Inferred" in df.columns:
        return df["Soma_Side_Inferred"].fillna(df.get("Soma_Side"))
    return df["Soma_Side"]


def stratum_counts(df: pd.DataFrame) -> dict:
    side = side_col(df)
    reg = df["Soma_Region_Refined"].astype(str).str.upper()
    out = {}
    for name, mask in [
        ("IDD5", reg == "IDD5"),
        ("IDM", reg == "IDM"),
        ("IDD5+IDM", reg.isin(["IDD5", "IDM"])),
        ("IAL", reg == "IAL"),
        ("IG", reg == "IG"),
        ("G", reg == "G"),
    ]:
        sub = side[mask]
        out[name] = {
            "L": int((sub == "L").sum()),
            "R": int((sub == "R").sum()),
            "n": int(mask.sum()),
        }
    return out


def crosstab_region_side(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        [df["SampleID"].astype(str), df["Soma_Region_Refined"].astype(str)],
        side_col(df).fillna("NA"),
    )


can = pd.read_excel(CAN_H, sheet_name="Summary")
stg = pd.read_excel(STG_H, sheet_name="Summary")

can_keys = set(zip(can["SampleID"].astype(str), can["NeuronID"].astype(str)))
stg_keys = set(zip(stg["SampleID"].astype(str), stg["NeuronID"].astype(str)))
added = sorted(stg_keys - can_keys)
removed = sorted(can_keys - stg_keys)

can_st = stratum_counts(can)
stg_st = stratum_counts(stg)

# Ingest yield check vs portal audit expectations
new_sids = ["252790", "250432", "252714"]
# Pre-harmonize atlas labels from staging combined (before Ig→IG etc.)
stg_comb = pd.read_excel(
    ST / "combined" / "multi_monkey_INS_combined.xlsx", sheet_name="Summary"
)
yield_lines = []
expected = {
    "252790": {"IAI_L": 15, "IAI_R": 1, "IAL_L": 1, "IAL_R": 16, "IG_L": 5, "IG_R": 0},
    "250432": {"IA/ID_L": 2, "IA/ID_R": 0, "IG_L": 0, "IG_R": 6},
    "252714": {"IA/ID_L": 3, "IA/ID_R": 1},
}
for sid in new_sids:
    sub = stg_comb[stg_comb["SampleID"].astype(str) == sid].copy()
    auto = sub["Soma_Region_Auto"].astype(str)
    # strip CL_/CR_
    def leaf(s):
        for p in ("CL_", "CR_"):
            if s.startswith(p):
                return s[len(p):]
        return s

    leaves = auto.map(leaf)
    sides = side_col(sub)
    counts = {}
    for leaf_u, side in zip(leaves, sides):
        key = f"{leaf_u.upper()}_{side}"
        # normalize IA/ID casing
        key = key.replace("IA/ID", "IA/ID").replace("IAI", "IAI").replace("IAL", "IAL").replace("IG", "IG")
        # Use original leaf for IA/Id
        key2 = f"{leaf(auto.iloc[0])}"  # unused
        counts[f"{leaf_u}_{side}"] = counts.get(f"{leaf_u}_{side}", 0) + 1
    # rebuild properly
    counts = {}
    for a, s in zip(auto, sides):
        lf = leaf(a)
        counts[f"{lf}_{s}"] = counts.get(f"{lf}_{s}", 0) + 1
    yield_lines.append((sid, counts, expected.get(sid, {})))

# Ig source tags
ig_src = stg[stg["Soma_Region_Source"].astype(str).str.contains("Ig_to_IG", na=False)]
g_rows = stg[stg["Soma_Region_Refined"].astype(str).str.upper() == "G"]

# Mapping rule sheet
mr = pd.read_excel(STG_H, sheet_name="Mapping_Rule")

diff_sum = pd.read_csv(DIFF_SUM) if DIFF_SUM.exists() else pd.DataFrame()

# Write QC tables
ct_can = crosstab_region_side(can)
ct_stg = crosstab_region_side(stg)
ct_can.to_csv(ST / "qc_crosstab_canonical.csv")
ct_stg.to_csv(ST / "qc_crosstab_staging.csv")

pd.DataFrame(
    [{"SampleID": a, "NeuronID": b} for a, b in added]
).to_csv(ST / "qc_membership_added.csv", index=False)
pd.DataFrame(
    [{"SampleID": a, "NeuronID": b} for a, b in removed],
    columns=["SampleID", "NeuronID"],
).to_csv(ST / "qc_membership_removed.csv", index=False)

stratum_tbl = pd.DataFrame(
    [
        {
            "stratum": k,
            "can_L": can_st[k]["L"],
            "can_R": can_st[k]["R"],
            "can_n": can_st[k]["n"],
            "stg_L": stg_st[k]["L"],
            "stg_R": stg_st[k]["R"],
            "stg_n": stg_st[k]["n"],
        }
        for k in can_st
    ]
)
stratum_tbl.to_csv(ST / "qc_stratum_before_after.csv", index=False)

# README
lines = []
lines.append("# Staging 2026-09-26 — Phase 0 fixes + portal ingest")
lines.append("")
lines.append("All outputs under this folder. **Canonical** `group_analysis/{combined,recovery,reference}` were not overwritten.")
lines.append("")
lines.append("## What changed (code)")
lines.append("")
lines.append("1. `scripts/cohort.py` — central `NEW_SAMPLES` (9 IDs incl. 252527/252718/252790/250432/252714), geometry constants, `PROJECTOME_SAMPLES` / `PROJECTOME_PAD_TOL_MM` env; `KEEP_ATLAS_LABELS={'G'}` so gustatory G stays in cohort as G (not mapped to IG).")
lines.append("2. Folded-X rescue geometry (`dx=|X-128|`) + pad in mm (`PAD_TOL_MM` → voxels via 0.25 mm).")
lines.append("3. Ig harmonization: atlas `Ig` → granular `IG` (`atlas_Ig_to_IG_granular_user_rule_20260926`); atlas `G` untouched; only `auto_atlas_insula` rows edited; 251637 remains `curated_251637`.")
lines.append("4. R: drop `pmin` across presence/magnitude BH families; F3A greying for `n_source<10`; deprecate `multi_monkey_lr_analysis.R`.")
lines.append("")
lines.append("## Commands run")
lines.append("")
lines.append("```powershell")
lines.append("$PY = (conda run -n deb_fmri_pipeline python -c \"import sys;print(sys.executable)\").Trim()")
lines.append("$env:PYTHONIOENCODING='utf-8'")
lines.append("$ST='D:\\projectome_analysis\\group_analysis\\staging_20260926'")
lines.append("$env:PROJECTOME_REFERENCE_OUT=\"$ST\\reference\"")
lines.append("# 01 folded reference")
lines.append("& $PY group_analysis\\scripts\\01_build_insula_coord_reference.py")
lines.append("# step1 ingest (multimodal_ins_win — seaborn missing in deb_fmri_pipeline)")
lines.append("& $PY_MM group_analysis\\scripts\\run_step1_one_sample.py 252790")
lines.append("& $PY_MM group_analysis\\scripts\\run_step1_one_sample.py 250432")
lines.append("& $PY_MM group_analysis\\scripts\\run_step1_one_sample.py 252714")
lines.append("$env:PROJECTOME_REFERENCE_DIR=\"$ST\\reference\"")
lines.append("$env:PROJECTOME_PAD_TOL_MM='0.5'; $env:PROJECTOME_RECOVERY_OUT=\"$ST\\recovery_pad0p5\"")
lines.append("& $PY group_analysis\\scripts\\04_refine_soma_region_by_coords.py")
lines.append("$env:PROJECTOME_PAD_TOL_MM='2.0'; $env:PROJECTOME_RECOVERY_OUT=\"$ST\\recovery_pad2p0\"")
lines.append("& $PY group_analysis\\scripts\\04_refine_soma_region_by_coords.py")
lines.append("$env:PROJECTOME_RECOVERY_DIR=\"$ST\\recovery\"; $env:PROJECTOME_COMBINED_OUT=\"$ST\\combined\"")
lines.append("& $PY group_analysis\\scripts\\05a_build_combined_table.py")
lines.append("$env:PROJECTOME_COMBINED_IN=\"$ST\\combined\\multi_monkey_INS_combined.xlsx\"")
lines.append("$env:PROJECTOME_HARMONIZED_OUT=\"$ST\\combined\\multi_monkey_INS_combined_harmonized.xlsx\"")
lines.append("& $PY group_analysis\\scripts\\06_harmonize_atlas_to_manual.py")
lines.append("```")
lines.append("")
lines.append("### Not run (list only)")
lines.append("")
lines.append("```powershell")
lines.append("# Global FNT rebuild (heavy) — do NOT run here:")
lines.append("& $PY group_analysis\\scripts\\05c_run_global_fnt.py")
lines.append("# (point inputs at staging harmonized / updated cohort after freeze)")
lines.append("```")
lines.append("")
lines.append("## Membership vs canonical harmonized (n=353)")
lines.append("")
lines.append(f"- Staging harmonized n = **{len(stg)}**")
lines.append(f"- Added neurons: **{len(added)}** (see `qc_membership_added.csv`)")
lines.append(f"- Removed neurons: **{len(removed)}** (see `qc_membership_removed.csv`)")
lines.append("")
lines.append("## Balanced-stratum L:R (before = canonical, after = staging)")
lines.append("")
lines.append("| Stratum | Can L:R (n) | Stg L:R (n) |")
lines.append("|---------|-------------|-------------|")
for _, r in stratum_tbl.iterrows():
    lines.append(
        f"| {r['stratum']} | {r['can_L']}:{r['can_R']} ({r['can_n']}) | "
        f"{r['stg_L']}:{r['stg_R']} ({r['stg_n']}) |"
    )
lines.append("")
lines.append("## Rescue rebuild (folded X; vs canonical `recovery/all_refined_neurons.csv`)")
lines.append("")
lines.append("Note: canonical `all_refined_neurons.csv` lacked 252527/252718 (those lived only as per-sample xlsx / combined).")
lines.append("")
if len(diff_sum):
    lines.append("```")
    lines.append(diff_sum.to_string(index=False))
    lines.append("```")
lines.append("")
lines.append("- Pad 0.5 mm = legacy effective pad (2 vox). Pad 2.0 mm = documented intent (8 vox).")
lines.append("- Primary staging recovery used for 05a/06 = **pad 2.0 mm** (`recovery/` mirrored from `recovery_pad2p0/`).")
lines.append("")
lines.append("## Ingest yields vs portal audit expected")
lines.append("")
lines.append("| Sample | Observed atlas-leaf×side (combined keepers) | Expected (audit) | Match? |")
lines.append("|--------|---------------------------------------------|------------------|--------|")
for sid, obs, exp in yield_lines:
    obs_s = ", ".join(f"{k}={v}" for k, v in sorted(obs.items()))
    exp_s = ", ".join(f"{k}={v}" for k, v in sorted(exp.items()))
    # soft match: check keys with case folding
    ok = True
    for k, v in exp.items():
        # expected keys like IAI_L
        found = False
        for ok_k, ok_v in obs.items():
            if ok_k.replace("/", "/").upper() == k.upper() or ok_k.upper().replace("IA/ID", "IA/ID") == k.upper():
                # normalize
                pass
            kn = ok_k.upper().replace("IA/ID", "IA/ID")
            if kn == k.upper() or ok_k == k.replace("IA/ID", "Ia/Id").replace("_", "_"):
                found = True
                if ok_v != v:
                    ok = False
        # simpler: map expected
    # manual compare helper
    def get_obs(leaf, side):
        for k, v in obs.items():
            if k.upper() == f"{leaf.upper()}_{side}" or k == f"{leaf}_{side}":
                return v
            if leaf == "IA/ID" and k.upper() in (f"IA/ID_{side}", f"IA/ID_{side}"):
                return v
            if leaf.upper() == "IA/ID" and ("IA/ID" in k.upper() or "IA/ID" in k) and k.endswith(f"_{side}"):
                return v
        # try Ia/Id
        for k, v in obs.items():
            if k.endswith(f"_{side}") and k.replace("Ia/Id", "IA/ID").upper().startswith("IA/ID"):
                if leaf.upper() == "IA/ID":
                    return v
        return 0

    mismatches = []
    for k, v in exp.items():
        leaf, side = k.rsplit("_", 1)
        got = get_obs(leaf, side)
        if got != v:
            mismatches.append(f"{k}: got {got} vs {v}")
    # Flag unexpected extra atlas leaves vs audit (e.g. G on 252714)
    exp_leaf_sides = {tuple(k.rsplit("_", 1)) for k in exp}
    for k, v in obs.items():
        if "_" not in k:
            continue
        leaf, side = k.rsplit("_", 1)
        leaf_norm = leaf.upper().replace("IA/ID", "IA/ID")
        key_norm = (leaf_norm, side)
        # normalize expected keys
        matched_exp = False
        for el, es in exp_leaf_sides:
            if es == side and el.upper().replace("IA/ID", "IA/ID") == leaf_norm:
                matched_exp = True
                break
            if es == side and el.upper() == leaf.upper():
                matched_exp = True
                break
            if es == side and leaf in ("Ia/Id", "IA/ID") and el.upper() in ("IA/ID", "IA/ID"):
                matched_exp = True
                break
        if not matched_exp and leaf.upper() not in {e[0].upper() for e in exp_leaf_sides}:
            mismatches.append(f"extra {k}={v} (not in audit insula list)")
    match = "OK" if not mismatches else ("DEV: " + "; ".join(mismatches))
    lines.append(f"| {sid} | {obs_s} | {exp_s} | {match} |")
lines.append("")
lines.append(f"- Ig→IG source-tag rows in staging harmonized: **{len(ig_src)}**")
lines.append(f"- Staging `Soma_Region_Refined==G` (gustatory, unmapped): **{len(g_rows)}**")
lines.append("")
lines.append("## Mapping_Rule (staging)")
lines.append("")
lines.append("```")
lines.append(mr.to_string(index=False))
lines.append("```")
lines.append("")
lines.append("## Crosstabs")
lines.append("")
lines.append("- `qc_crosstab_canonical.csv` — SampleID × Soma_Region_Refined × Side (canonical)")
lines.append("- `qc_crosstab_staging.csv` — same for staging harmonized")
lines.append("")
lines.append("## Env notes")
lines.append("")
lines.append("- `deb_fmri_pipeline` lacked `seaborn` → step1 used `multimodal_ins_win`.")
lines.append("- ION fetch via existing `region_analysis` / IONData path (step1); new code elsewhere follows bounded-retry pattern.")
lines.append("")

readme = ST / "README.md"
readme.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {readme}")
print(stratum_tbl.to_string(index=False))
print(f"n can={len(can)} stg={len(stg)} added={len(added)} removed={len(removed)}")
print(f"Ig→IG rows={len(ig_src)} G rows={len(g_rows)}")
for sid, obs, exp in yield_lines:
    print(sid, "obs=", obs, "exp=", exp)
