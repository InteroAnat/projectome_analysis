from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .paths import COMBINED_XLSX, PROJECT, REF_INS_XLSX, STEP1_DIR

SCRIPTS = PROJECT / "group_analysis" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from insula_label_set import build_insula_label_set, normalize_label  # noqa: E402


def find_step1_xlsx(fmost_id: str | int) -> Path | None:
    sid = str(int(fmost_id)) if pd.notna(fmost_id) else ""
    if not sid:
        return None
    pattern = str(
        STEP1_DIR / f"{sid}_*_region_analysis" / "tables" / f"{sid}_results_*.xlsx"
    )
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _format_distribution(counts: Counter) -> str:
    if not counts:
        return "—"
    return " ".join(f"{k}:{v}" for k, v in counts.most_common())


def count_auto_insula(summary: pd.DataFrame, region_col: str = "Soma_Region") -> int:
    if summary.empty or region_col not in summary.columns:
        return 0
    insula_labels, _ = build_insula_label_set()
    n = 0
    for val in summary[region_col].fillna(""):
        if normalize_label(val) in insula_labels:
            n += 1
    return n


def load_step1_stats(fmost_ids: list[str | int]) -> pd.DataFrame:
    rows = []
    for fid in fmost_ids:
        sid = str(int(fid)) if pd.notna(fid) else ""
        xlsx = find_step1_xlsx(sid)
        if xlsx is None:
            rows.append(
                dict(
                    fmost_id=sid,
                    step1_xlsx="",
                    step1_all_n=0,
                    step1_auto_insula_n=0,
                    step1_auto_distribution="—",
                )
            )
            continue
        summary = pd.read_excel(xlsx, sheet_name="Summary")
        counts: Counter = Counter()
        insula_labels, _ = build_insula_label_set()
        for val in summary.get("Soma_Region", pd.Series(dtype=str)).fillna(""):
            lab = normalize_label(val)
            if lab in insula_labels:
                counts[lab] += 1
        rows.append(
            dict(
                fmost_id=sid,
                step1_xlsx=str(xlsx.relative_to(PROJECT)),
                step1_all_n=len(summary),
                step1_auto_insula_n=sum(counts.values()),
                step1_auto_distribution=_format_distribution(counts),
            )
        )
    return pd.DataFrame(rows)


def load_combined_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-sample summary, long distribution table)."""
    if not COMBINED_XLSX.exists():
        empty = pd.DataFrame(
            columns=[
                "fmost_id",
                "combined_insula_n",
                "combined_distribution",
                "combined_lr",
            ]
        )
        return empty, pd.DataFrame(columns=["fmost_id", "subregion", "n", "side"])

    df = pd.read_excel(COMBINED_XLSX, sheet_name="Summary")
    region_col = "Soma_Region_Refined" if "Soma_Region_Refined" in df.columns else "Soma_Region_Auto"
    side_col = "Soma_Side" if "Soma_Side" in df.columns else None

    long_rows = []
    summary_rows = []
    for sid, sub in df.groupby("SampleID"):
        sid = str(int(sid))
        vc = sub[region_col].fillna("NA").value_counts()
        lr = "—"
        if side_col and side_col in sub.columns:
            l = int((sub[side_col].astype(str).str.upper() == "L").sum())
            r = int((sub[side_col].astype(str).str.upper() == "R").sum())
            lr = f"L{l}/R{r}"
        summary_rows.append(
            dict(
                fmost_id=sid,
                combined_insula_n=len(sub),
                combined_distribution=_format_distribution(Counter(vc.to_dict())),
                combined_lr=lr,
            )
        )
        for region, n in vc.items():
            long_rows.append(dict(fmost_id=sid, subregion=str(region), n=int(n), side="all"))
        if side_col:
            for (region, side), n in sub.groupby([region_col, side_col]).size().items():
                long_rows.append(
                    dict(
                        fmost_id=sid,
                        subregion=str(region),
                        n=int(n),
                        side=str(side),
                    )
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def load_reference_distribution() -> str:
    if not REF_INS_XLSX.exists():
        return "—"
    df = pd.read_excel(REF_INS_XLSX, sheet_name="Summary")
    col = "Soma_Region" if "Soma_Region" in df.columns else df.columns[0]
    vc = df[col].value_counts()
    return _format_distribution(Counter(vc.to_dict()))
