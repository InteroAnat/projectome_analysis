"""
Phase 2: Run step1 region analysis on the 4 new monkey samples.

This is a thin wrapper around PopulationRegionAnalysis that redirects all
outputs into group_analysis/step1_results/{sample_id}_<timestamp>/ instead
of the default ./output location.

DOES NOT touch 251637 (per user instruction).

Outputs: per-sample timestamped folder with tables/, reports/, plots/.
"""
from __future__ import annotations

import os
import sys
import nibabel as nib
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = r"D:\projectome_analysis"
MAIN_SCRIPTS = os.path.join(PROJECT_ROOT, "main_scripts")
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
for p in (MAIN_SCRIPTS, NEUROVIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from region_analysis import PopulationRegionAnalysis  # noqa: E402

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]

ATLAS_PATH = os.path.join(PROJECT_ROOT, "atlas", "ARM_in_NMT_v2.1_sym.nii.gz")
TABLE_PATH = os.path.join(PROJECT_ROOT, "atlas", "ARM_key_all.txt")
TEMPLATE = os.path.join(PROJECT_ROOT, "atlas", "NMT_v2.1_sym",
                        "NMT_v2.1_sym", "NMT_v2.1_sym_SS.nii.gz")
CTX_HIER_CSV = os.path.join(PROJECT_ROOT, "atlas", "CHARM_key_table_v2.csv")
SUBCTX_HIER_CSV = os.path.join(PROJECT_ROOT, "atlas", "SARM_key_table_v2.csv")

OUT_BASE = os.path.join(PROJECT_ROOT, "group_analysis", "step1_results")
os.makedirs(OUT_BASE, exist_ok=True)


def run_one(sample_id: str, atlas_data, atlas_table, template):
    print("\n" + "=" * 70)
    print(f"[Phase 2] Sample {sample_id}")
    print("=" * 70)

    pop = PopulationRegionAnalysis(
        sample_id=sample_id,
        atlas=atlas_data,
        atlas_table=atlas_table,
        template_img=template,
        arm_key_path=TABLE_PATH,
        cortex_hierarchy_csv=CTX_HIER_CSV,
        subcortical_hierarchy_csv=SUBCTX_HIER_CSV,
        output_base=OUT_BASE,
        create_output_folder=True,
        show_plots=False,
    )
    pop.process(limit=None, level=6)
    if pop.plot_dataframe.empty:
        print(f"[WARN] {sample_id}: no neurons processed.")
        return None

    output_path = pop.save_all(
        include_strength_levels=[3],
        generate_plots=True,
    )

    # Rename default results workbook for traceability (mirrors the existing wrapper).
    if output_path:
        tables_dir = os.path.join(output_path, "tables")
        default_results = os.path.join(tables_dir, f"{sample_id}_results.xlsx")
        if os.path.exists(default_results):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped = os.path.join(
                tables_dir, f"{sample_id}_results_{ts}.xlsx"
            )
            os.replace(default_results, timestamped)
            print(f"[RENAMED] {timestamped}")

    counts = pop.get_neuron_count()
    print(f"\n[{sample_id}] Processed: {counts['processed']} / "
          f"{counts['total']} ({counts['completion_rate']}%)")
    return output_path


def main(samples=None) -> int:
    samples = samples or NEW_SAMPLES
    for f in (ATLAS_PATH, TABLE_PATH, TEMPLATE, CTX_HIER_CSV, SUBCTX_HIER_CSV):
        if not os.path.exists(f):
            print(f"[ERROR] Missing required atlas file: {f}")
            return 1

    print(f"[Phase 2] Loading atlas: {ATLAS_PATH}")
    atlas_data = nib.load(ATLAS_PATH).get_fdata()
    atlas_table = pd.read_csv(TABLE_PATH, delimiter="\t")
    template = nib.load(TEMPLATE)
    print(f"  atlas shape: {atlas_data.shape}, atlas rows: {len(atlas_table)}")

    summary_rows = []
    for sid in samples:
        try:
            out = run_one(sid, atlas_data, atlas_table, template)
            summary_rows.append(dict(sample_id=sid, ok=out is not None,
                                     output_path=str(out) if out else ""))
        except Exception as e:
            print(f"[ERROR] {sid}: {e}")
            summary_rows.append(dict(sample_id=sid, ok=False, output_path="",
                                     error=str(e)))

    # Summary manifest
    df = pd.DataFrame(summary_rows)
    manifest = os.path.join(OUT_BASE, "phase2_run_summary.csv")
    df.to_csv(manifest, index=False)
    print(f"\n[saved manifest] {manifest}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
