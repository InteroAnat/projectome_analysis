"""
bulk_visual_multi_monkey.py

Wraps step3.1.bulk_visual_data.py logic to run on ALL new monkey samples
(251730, 252383, 252384, 252385) for three groups:
  - INS     : any atlas-derived insula sub-region (Ial, Iai, Iam/Iapm, Iapl,
              lat_Ia, Ia/Id, Ig, Pi, Ins, Ins/Pi, etc.)
  - PrCO    : auto-labeled PrCO
  - Unknown : Soma_Region empty / contains "Unknown"

Output goes to W:\\fMOST\\visual\\{SAMPLE_ID}\\Region_{group}\\{HighRes,LowRes}\\...

Skips 251637 entirely.
"""
from __future__ import annotations

import os
import sys
import glob
import traceback
import pandas as pd
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
MAIN_SCRIPTS = os.path.join(PROJECT_ROOT, "main_scripts")
GROUP_SCRIPTS = os.path.join(PROJECT_ROOT, "group_analysis", "scripts")
for p in (NEUROVIS, MAIN_SCRIPTS, GROUP_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports must happen after path setup
from Visual_toolkit import Visual_toolkit  # noqa: E402
import IONData as IT  # noqa: E402
from insula_label_set import build_insula_label_set, normalize_label, strip_prefix  # noqa: E402

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]
STEP1_DIR   = os.path.join(PROJECT_ROOT, "group_analysis", "step1_results")
PARENT_OUT  = r"W:\fMOST\visual"

GENERATE_HIGH_RES = True
GENERATE_LOW_RES  = True

INSULA_LABELS, _ = build_insula_label_set()
DATESTAMP = datetime.now().strftime("%Y%m%d")


def sanitize(name: str) -> str:
    if not name:
        return name
    return (name.replace("/", "-").replace("\\", "-").replace(":", "-")
                 .replace(" ", "_"))


def find_results_xlsx(sample_id: str) -> str | None:
    pattern = os.path.join(STEP1_DIR, f"{sample_id}_*_region_analysis",
                           "tables", f"{sample_id}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def assign_group(soma_region_raw: str) -> str | None:
    """Return 'INS' | 'PrCO' | 'Unknown' | None (skip)."""
    if soma_region_raw is None or (isinstance(soma_region_raw, float)
                                     and pd.isna(soma_region_raw)):
        return "Unknown"
    raw = str(soma_region_raw).strip()
    if not raw or "Unknown" in raw:
        return "Unknown"
    clean_upper = normalize_label(raw)  # uppercase, prefix stripped
    if clean_upper in INSULA_LABELS:
        return "INS"
    # Strict PrCO match (case-insensitive)
    if strip_prefix(raw).upper() == "PRCO":
        return "PrCO"
    return None


def safe_fetch_raw_swc(ion, sample_id, neuron_id, max_retries=3, timeout_s=15):
    """Fetch raw SWC with bounded retries; bypasses IONData's infinite-retry
    loop in downloadfile() for neurons whose raw SWC is missing on the server.
    Returns a NeuronTree or None on failure."""
    import requests as _req
    from SwcLoader import NeuronTree
    raw_dir = os.path.join("../resource/swc_raw", sample_id)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, neuron_id)

    # Short-circuit if cached locally and non-empty
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 100:
        try:
            tree = NeuronTree()
            tree.readFile(raw_path)
            return tree
        except Exception:
            pass  # fall through to fetch

    # Try SSH path first via IONData metadata
    info = ion.getSampleInfo(sample_id)
    if not info:
        return None
    project_id = info[0].get("project_id")
    url = f"http://10.10.31.31/swc/newswc/{project_id}/{sample_id}/swc_raw/{neuron_id}"
    for attempt in range(max_retries):
        try:
            r = _req.get(url, timeout=timeout_s)
            if r.status_code != 200 or not r.text or len(r.text) < 200:
                return None  # SWC not available
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            tree = NeuronTree()
            tree.readFile(raw_path)
            return tree
        except _req.RequestException:
            if attempt == max_retries - 1:
                return None
            continue
    return None


def process_neuron(toolkit, ion, sample_id, neuron_id, soma_region, dirs):
    tree = safe_fetch_raw_swc(ion, sample_id, neuron_id)
    if tree is None:
        raise ValueError(f"raw SWC not available for {sample_id}/{neuron_id} (skipped after retries)")
    soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

    if GENERATE_HIGH_RES:
        vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
        toolkit.export_data(
            vol_h, org_h, res_h, neuron_id,
            suffix="SomaBlock", soma_region=soma_region,
            soma_coords=soma_xyz, output_dir=dirs["high_nii"],
        )
        toolkit.plot_soma_block(
            vol_h, org_h, res_h, soma_xyz, neuron_id,
            suffix="SomaBlock", soma_region=soma_region,
            output_dir=dirs["high_img"],
        )

    if GENERATE_LOW_RES:
        vol_l, org_l, res_l = toolkit.get_low_res_widefield(
            soma_xyz, width_um=8000, height_um=8000, depth_um=30
        )
        toolkit.export_data(
            vol_l, org_l, res_l, neuron_id,
            suffix="WideField", soma_region=soma_region,
            soma_coords=soma_xyz, output_dir=dirs["low_nii"],
        )
        toolkit.plot_widefield_context(
            vol_l, org_l, res_l, soma_xyz, neuron_id,
            bg_intensity=2.0, swc_tree=tree,
            soma_region=soma_region, output_dir=dirs["low_img"],
        )


def process_sample(sample_id: str, ion: IT.IONData) -> dict:
    print("\n" + "=" * 70)
    print(f"[bulk_visual] Sample {sample_id}")
    print("=" * 70)
    xlsx = find_results_xlsx(sample_id)
    if not xlsx:
        print(f"[skip] {sample_id}: no results.xlsx")
        return dict(sample_id=sample_id, total=0, ins=0, prco=0,
                     unknown=0, ok=0, fail=0)
    print(f"  source: {xlsx}")

    summ = pd.read_excel(xlsx, sheet_name="Summary")
    summ["group"] = summ["Soma_Region"].map(assign_group)
    keep = summ[summ["group"].isin(["INS", "PrCO", "Unknown"])].copy()
    counts = keep["group"].value_counts().to_dict()
    print(f"  filter: total={len(summ)}  kept={len(keep)} "
          f"(INS={counts.get('INS', 0)}, "
          f"PrCO={counts.get('PrCO', 0)}, "
          f"Unknown={counts.get('Unknown', 0)})")

    if not len(keep):
        return dict(sample_id=sample_id, total=len(summ), ins=0, prco=0,
                     unknown=0, ok=0, fail=0)

    success = 0
    fail = 0
    failed_neurons = []
    toolkit = Visual_toolkit(sample_id)

    try:
        # Process per group so the folder structure mirrors the existing pipeline
        for group_name, group_df in keep.groupby("group"):
            print(f"\n  -- Group {group_name} ({len(group_df)} neurons) --")
            base = os.path.join(
                PARENT_OUT, sample_id,
                f"cube_data_{sample_id}_INS_PrCO_Unknown_{DATESTAMP}",
                f"Region_{sanitize(group_name)}"
            )
            dirs = {
                "high_img": os.path.join(base, "HighRes", "Plots"),
                "high_nii": os.path.join(base, "HighRes", "Data"),
                "low_img":  os.path.join(base, "LowRes",  "Plots"),
                "low_nii":  os.path.join(base, "LowRes",  "Data"),
            }
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)

            for _, row in tqdm(group_df.iterrows(), total=len(group_df),
                                  desc=f"{sample_id}/{group_name}"):
                neuron_id = str(row["NeuronID"])
                soma_region = str(row.get("Soma_Region")) if pd.notnull(
                    row.get("Soma_Region")) else "Unknown"
                try:
                    process_neuron(toolkit, ion, sample_id, neuron_id,
                                    soma_region, dirs)
                    success += 1
                except Exception as e:
                    fail += 1
                    failed_neurons.append(f"{neuron_id}: {e}")
                    print(f"\n    [ERR] {neuron_id}: {e}")
    finally:
        try:
            toolkit.close()
        except Exception:
            pass

    if failed_neurons:
        log = os.path.join(PARENT_OUT, sample_id,
                            f"failed_neurons_{DATESTAMP}.txt")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_neurons))
        print(f"  failed list -> {log}")

    return dict(sample_id=sample_id, total=len(summ),
                 ins=counts.get("INS", 0), prco=counts.get("PrCO", 0),
                 unknown=counts.get("Unknown", 0),
                 ok=success, fail=fail)


def main():
    if not os.path.exists(PARENT_OUT):
        os.makedirs(PARENT_OUT, exist_ok=True)

    ion = IT.IONData()
    rows = []
    for sid in NEW_SAMPLES:
        try:
            r = process_sample(sid, ion)
        except Exception as e:
            r = dict(sample_id=sid, total=0, ins=0, prco=0, unknown=0,
                     ok=0, fail=0, error=str(e))
            print(f"[ERROR] {sid}: {e}")
            traceback.print_exc()
        rows.append(r)

    df = pd.DataFrame(rows)
    out = os.path.join(PARENT_OUT, f"bulk_visual_summary_{DATESTAMP}.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 70)
    print("MULTI-MONKEY BULK VISUAL FINISHED")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
