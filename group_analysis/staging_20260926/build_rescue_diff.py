"""Build rescue_rebuild_diff.csv comparing canonical vs folded pad 0.5/2.0."""
from __future__ import annotations

import os
import pandas as pd

ST = r"D:\projectome_analysis\group_analysis\staging_20260926"
CAN = r"D:\projectome_analysis\group_analysis\recovery\all_refined_neurons.csv"
p05 = os.path.join(ST, "recovery_pad0p5", "all_refined_neurons.csv")
p20 = os.path.join(ST, "recovery_pad2p0", "all_refined_neurons.csv")
out = os.path.join(ST, "recovery", "rescue_rebuild_diff.csv")
os.makedirs(os.path.join(ST, "recovery"), exist_ok=True)

old = pd.read_csv(CAN)
n05 = pd.read_csv(p05)
n20 = pd.read_csv(p20)


def key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        SampleID=df["SampleID"].astype(str),
        NeuronID=df["NeuronID"].astype(str),
    )


old, n05, n20 = key(old), key(n05), key(n20)


def slim(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    cols = ["SampleID", "NeuronID", "Soma_Region_Refined", "Soma_Region_Source"]
    s = df[cols].copy()
    return s.rename(
        columns={
            "Soma_Region_Refined": f"label_{tag}",
            "Soma_Region_Source": f"source_{tag}",
        }
    )


o = slim(old, "old")
a = slim(n05, "pad0p5")
b = slim(n20, "pad2p0")

keys = pd.concat(
    [
        o[["SampleID", "NeuronID"]],
        a[["SampleID", "NeuronID"]],
        b[["SampleID", "NeuronID"]],
    ]
).drop_duplicates()

m = keys.merge(o, how="left").merge(a, how="left").merge(b, how="left")


def flag_row(r) -> str:
    in_old = pd.notna(r["label_old"])
    in_05 = pd.notna(r["label_pad0p5"])
    in_20 = pd.notna(r["label_pad2p0"])
    flags = []
    if in_old and not in_05:
        flags.append("lost_at_0p5")
    if in_old and not in_20:
        flags.append("lost_at_2p0")
    if (not in_old) and in_05:
        flags.append("gained_at_0p5")
    if (not in_old) and in_20:
        flags.append("gained_at_2p0")
    if in_old and in_05 and r["label_old"] != r["label_pad0p5"]:
        flags.append("label_change_0p5")
    if in_old and in_20 and r["label_old"] != r["label_pad2p0"]:
        flags.append("label_change_2p0")
    if in_05 and in_20 and r["label_pad0p5"] != r["label_pad2p0"]:
        flags.append("pad0p5_vs_2p0_label_diff")
    if not flags:
        if in_old and in_05 and in_20:
            flags.append("unchanged")
        else:
            flags.append("partial")
    return ";".join(flags)


m["change_flag"] = m.apply(flag_row, axis=1)
m.to_csv(out, index=False)
print(f"wrote {out} n={len(m)}")

rows = []
for sid, sub in m.groupby("SampleID"):
    old_n = sub["label_old"].notna().sum()
    n05_n = sub["label_pad0p5"].notna().sum()
    n20_n = sub["label_pad2p0"].notna().sum()
    gained_05 = ((sub["label_old"].isna()) & sub["label_pad0p5"].notna()).sum()
    lost_05 = (sub["label_old"].notna() & sub["label_pad0p5"].isna()).sum()
    gained_20 = ((sub["label_old"].isna()) & sub["label_pad2p0"].notna()).sum()
    lost_20 = (sub["label_old"].notna() & sub["label_pad2p0"].isna()).sum()

    def n_rescued(col_src: str) -> int:
        return int(sub[col_src].fillna("").astype(str).str.startswith("coord_inferred").sum())

    rows.append(
        dict(
            SampleID=sid,
            n_old=int(old_n),
            n_pad0p5=int(n05_n),
            n_pad2p0=int(n20_n),
            gained_pad0p5=int(gained_05),
            lost_pad0p5=int(lost_05),
            gained_pad2p0=int(gained_20),
            lost_pad2p0=int(lost_20),
            rescued_old=n_rescued("source_old"),
            rescued_pad0p5=n_rescued("source_pad0p5"),
            rescued_pad2p0=n_rescued("source_pad2p0"),
        )
    )

sum_df = pd.DataFrame(rows).sort_values("SampleID")
sum_path = os.path.join(ST, "recovery", "rescue_rebuild_diff_summary.csv")
sum_df.to_csv(sum_path, index=False)
print(sum_df.to_string(index=False))
print("\nchange_flag value counts:")
print(m["change_flag"].value_counts().to_string())
