"""Run bulk widefield and/or soma visual QC for one fMOST sample.

Exports the filtered step1 Summary table into each run folder under ``tables/``.

Examples:
  python run_bulk_visual_sample.py --sample 252385 --group INS --png-only
  python run_bulk_visual_sample.py --sample 252385 --groups INS,PrCO --with-soma --run-stamp 20260707
  python run_bulk_visual_sample.py --sample 252527 --potential-ins --with-soma --grid-radius 2
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
MAIN_SCRIPTS = os.path.join(PROJECT_ROOT, "main_scripts")
GROUP_SCRIPTS = os.path.join(PROJECT_ROOT, "group_analysis", "scripts")
for p in (NEUROVIS, MAIN_SCRIPTS, GROUP_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from Visual_toolkit import Visual_toolkit  # noqa: E402
import IONData as IT  # noqa: E402
from bulk_visual_multi_monkey import (  # noqa: E402
    PARENT_OUT,
    assign_group,
    find_results_xlsx,
    safe_fetch_raw_swc,
    sanitize,
)

COMBINED_DIR = os.path.join(PROJECT_ROOT, "group_analysis", "combined")
HARMONIZED_XLSX = os.path.join(COMBINED_DIR, "multi_monkey_INS_combined_harmonized.xlsx")
COMBINED_XLSX = os.path.join(COMBINED_DIR, "multi_monkey_INS_combined.xlsx")


def _resolve_potential_ins_table() -> str:
    """Latest potential-INS cohort table (harmonized preferred)."""
    if os.path.isfile(HARMONIZED_XLSX):
        return HARMONIZED_XLSX
    if os.path.isfile(COMBINED_XLSX):
        return COMBINED_XLSX
    raise FileNotFoundError(
        f"Missing potential-INS workbook: {HARMONIZED_XLSX} or {COMBINED_XLSX}"
    )


def load_potential_ins_neurons(sample_id: str) -> tuple[pd.DataFrame, dict]:
    """Load refined potential-INS keepers for one sample (atlas + PrCO rescue)."""
    path = _resolve_potential_ins_table()
    meta = {"step1_xlsx": path, "source": "potential_ins"}
    summ = pd.read_excel(path, sheet_name="Summary")
    keep = summ[summ["SampleID"].astype(str) == str(sample_id)].copy()
    if not len(keep):
        raise ValueError(f"No potential-INS rows for {sample_id} in {path}")
    if "Soma_Region_Refined" in keep.columns:
        keep["Soma_Region"] = keep["Soma_Region_Refined"]
    elif "Soma_Region" not in keep.columns:
        raise ValueError(f"No Soma_Region / Soma_Region_Refined in {path}")
    keep["group"] = "INS"
    print(f"[source] potential-INS table: {path} ({len(keep)} rows for {sample_id})")
    return keep, meta


def load_neurons(
    sample_id: str,
    group: str,
    combined_fallback: bool,
    potential_ins: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if potential_ins:
        if group != "INS":
            raise ValueError("--potential-ins only applies to group INS")
        return load_potential_ins_neurons(sample_id)

    meta = {"step1_xlsx": None, "source": None}
    xlsx = find_results_xlsx(sample_id)
    if xlsx:
        meta["step1_xlsx"] = xlsx
        meta["source"] = "step1"
        summ = pd.read_excel(xlsx, sheet_name="Summary")
        summ["group"] = summ["Soma_Region"].map(assign_group)
        keep = summ[summ["group"] == group].copy()
        if len(keep):
            return keep, meta

    if not combined_fallback:
        raise FileNotFoundError(
            f"No step1 Summary for {sample_id} and --no-combined-fallback"
        )

    combined = COMBINED_XLSX
    if not os.path.isfile(combined):
        raise FileNotFoundError(f"Missing combined workbook: {combined}")

    meta["source"] = "combined"
    summ = pd.read_excel(combined, sheet_name="Summary")
    sid = str(sample_id)
    keep = summ[summ["SampleID"].astype(str) == sid].copy()
    if "Soma_Region" not in keep.columns and "Soma_Region_Refined" in keep.columns:
        keep["Soma_Region"] = keep["Soma_Region_Refined"]
    keep["group"] = keep["Soma_Region"].map(assign_group)
    keep = keep[keep["group"] == group].copy()
    if not len(keep):
        raise ValueError(f"No {group} neurons for {sample_id} in step1 or combined table")
    print(f"[source] combined fallback: {combined} ({len(keep)} {group} rows)")
    return keep, meta


def export_step1_table(
    neurons: pd.DataFrame,
    run_base: str,
    sample_id: str,
    group: str,
    meta: dict,
) -> str:
    tables_dir = os.path.join(run_base, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, f"step1_{sample_id}_{group}_summary.csv")
    neurons.to_csv(csv_path, index=False)

    src = meta.get("step1_xlsx")
    if src and os.path.isfile(src):
        xlsx_name = os.path.basename(src)
        dest_xlsx = os.path.join(tables_dir, xlsx_name)
        if os.path.abspath(src) != os.path.abspath(dest_xlsx):
            shutil.copy2(src, dest_xlsx)
        readme = os.path.join(tables_dir, "README_step1_source.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"Filtered rows: {len(neurons)} ({group})\n")
            f.write(f"Source xlsx: {src}\n")
            f.write(f"Filtered csv: {csv_path}\n")

    print(f"[tables] {csv_path} ({len(neurons)} rows)")
    return csv_path


def _plot_exists(plot_dir: str, sample_id: str, neuron_id: str, suffix: str) -> bool:
    if not os.path.isdir(plot_dir):
        return False
    token = f"{sample_id}_{neuron_id}_"
    return any(
        f.startswith(token) and f.endswith(f"_{suffix}_Plot.png")
        for f in os.listdir(plot_dir)
    )


def run(args: argparse.Namespace) -> dict:
    datestamp = args.run_stamp or datetime.now().strftime("%Y%m%d")
    neurons, meta = load_neurons(
        args.sample,
        args.group,
        args.combined_fallback,
        potential_ins=args.potential_ins,
    )
    if args.smoke:
        nid = args.smoke if args.smoke.endswith(".swc") else f"{args.smoke}.swc"
        neurons = neurons[
            neurons["NeuronID"].astype(str).str.replace(".swc", "", regex=False)
            == nid.replace(".swc", "")
        ]
        if neurons.empty:
            neurons = pd.DataFrame([{"NeuronID": nid, "Soma_Region": args.group}])

    do_widefield = not args.soma_only
    do_soma = args.with_soma or args.soma_only

    import bulk_visual_multi_monkey as bvm

    bvm.GENERATE_HIGH_RES = do_soma and not args.png_only and not args.soma_only
    bvm.GENERATE_LOW_RES = do_widefield and not args.soma_only

    ion = IT.IONData()
    toolkit = Visual_toolkit(args.sample)
    run_base = os.path.join(
        PARENT_OUT,
        args.sample,
        f"cube_data_{args.sample}_{args.group}_{datestamp}",
    )
    base = os.path.join(run_base, f"Region_{sanitize(args.group)}")
    dirs = {
        "high_img": os.path.join(base, "HighRes", "Plots"),
        "high_nii": os.path.join(base, "HighRes", "Data"),
        "low_img": os.path.join(base, "LowRes", "Plots"),
        "low_nii": os.path.join(base, "LowRes", "Data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    table_path = export_step1_table(neurons, run_base, args.sample, args.group, meta)

    ok = fail = 0
    failed = []
    desc = f"{args.sample}/{args.group}" + (" smoke" if args.smoke else "")
    try:
        for _, row in tqdm(neurons.iterrows(), total=len(neurons), desc=desc):
            neuron_id = str(row["NeuronID"])
            if not neuron_id.endswith(".swc"):
                neuron_id = f"{neuron_id}.swc"
            soma_region = (
                str(row.get("Soma_Region"))
                if pd.notnull(row.get("Soma_Region"))
                else args.group
            )
            try:
                skip_wf = (
                    args.skip_existing
                    and do_widefield
                    and _plot_exists(dirs["low_img"], args.sample, neuron_id, "WideField")
                )
                skip_soma = (
                    args.skip_existing
                    and do_soma
                    and _plot_exists(dirs["high_img"], args.sample, neuron_id, "SomaBlock")
                )
                if skip_wf and (not do_soma or skip_soma):
                    ok += 1
                    continue

                tree = safe_fetch_raw_swc(ion, args.sample, neuron_id)
                if tree is None:
                    raise ValueError(f"raw SWC not available for {neuron_id}")
                soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

                if do_soma and not skip_soma:
                    vol_h, org_h, res_h = toolkit.get_high_res_block(
                        soma_xyz, grid_radius=args.grid_radius
                    )
                    if float(vol_h.max()) <= 0:
                        raise ValueError("soma block volume is all zeros")
                    if not args.png_only:
                        toolkit.export_data(
                            vol_h,
                            org_h,
                            res_h,
                            neuron_id,
                            suffix="SomaBlock",
                            soma_region=soma_region,
                            soma_coords=soma_xyz,
                            output_dir=dirs["high_nii"],
                        )
                    toolkit.plot_soma_block(
                        vol_h,
                        org_h,
                        res_h,
                        soma_xyz,
                        neuron_id,
                        suffix="SomaBlock",
                        soma_region=soma_region,
                        output_dir=dirs["high_img"],
                    )

                if do_widefield and not skip_wf:
                    vol_l, org_l, res_l = toolkit.get_low_res_widefield(
                        soma_xyz, width_um=8000, height_um=8000, depth_um=30
                    )
                    if float(vol_l.max()) <= 0:
                        raise ValueError("widefield volume is all zeros")
                    if not args.png_only:
                        toolkit.export_data(
                            vol_l,
                            org_l,
                            res_l,
                            neuron_id,
                            suffix="WideField",
                            soma_region=soma_region,
                            soma_coords=soma_xyz,
                            output_dir=dirs["low_nii"],
                        )
                    toolkit.plot_widefield_context(
                        vol_l,
                        org_l,
                        res_l,
                        soma_xyz,
                        neuron_id,
                        bg_intensity=2.0,
                        swc_tree=tree,
                        soma_region=soma_region,
                        output_dir=dirs["low_img"],
                    )
                ok += 1
            except Exception as exc:
                fail += 1
                failed.append(f"{neuron_id}: {exc}")
                print(f"\n[ERR] {neuron_id}: {exc}")
    finally:
        try:
            toolkit.close()
        except Exception:
            pass

    if failed:
        log = os.path.join(run_base, f"failed_neurons_{args.group}.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))

    result = dict(
        sample=args.sample,
        group=args.group,
        total=len(neurons),
        ok=ok,
        fail=fail,
        table=table_path,
        soma_png=dirs["high_img"] if do_soma else None,
        widefield_png=dirs["low_img"] if do_widefield else None,
    )
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", default="INS", choices=["INS", "PrCO", "Unknown"])
    parser.add_argument(
        "--groups",
        help="Comma-separated groups to run in sequence (overrides --group), e.g. INS,PrCO",
    )
    parser.add_argument("--png-only", action="store_true", help="PNG only (no NIfTI export)")
    parser.add_argument(
        "--with-soma",
        action="store_true",
        help="Also render high-res soma-block PNGs (HTTP 0.65um cubes)",
    )
    parser.add_argument(
        "--soma-only",
        action="store_true",
        help="Soma PNGs only (skip widefield)",
    )
    parser.add_argument(
        "--grid-radius",
        type=int,
        default=1,
        help="High-res soma block radius (1=1 cube, 2=3x3x3 cubes)",
    )
    parser.add_argument(
        "--run-stamp",
        help="Reuse output folder date stamp, e.g. 20260707",
    )
    parser.add_argument("--smoke", help="Neuron ID for single-neuron smoke test")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip neurons whose target PNG already exists",
    )
    parser.add_argument(
        "--no-combined-fallback",
        action="store_true",
        help="Do not fall back to multi_monkey_INS_combined.xlsx",
    )
    parser.add_argument(
        "--potential-ins",
        action="store_true",
        help=(
            "Use latest potential-INS cohort table "
            "(harmonized preferred: atlas + PrCO-rescue keepers), not step1 auto labels"
        ),
    )
    args = parser.parse_args()
    args.combined_fallback = not args.no_combined_fallback
    if not os.path.exists(PARENT_OUT):
        os.makedirs(PARENT_OUT, exist_ok=True)
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else [args.group]
    try:
        results = []
        for group in groups:
            args.group = group
            results.append(run(args))
        if len(results) > 1:
            print("\nALL GROUPS:", results)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
