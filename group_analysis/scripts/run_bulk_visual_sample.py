"""Run the bulk Soma/Context visual QC for one fMOST sample.

Version: 1.9.1 (released 2867eef; VisualToolkit package wrapper)
Last updated: 2026-10-17
Change history: main_scripts/Visual_toolkit/CHANGELOG.md (package changes) + notes below

Exports the filtered step1 Summary table into each run folder under ``tables/``.

Examples:
  python run_bulk_visual_sample.py --sample 252385 --group INS --png-only
  python run_bulk_visual_sample.py --sample 252385 --groups INS,PrCO --with-soma --run-stamp 20260707
  python run_bulk_visual_sample.py --sample 252527 --potential-ins --with-soma --grid-radius 2

2026-10-16 Codex (local, unreleased): add optional ``--cohort`` routing through
the active selected workbook and refined shared-output layout; legacy mode and
Soma MIP gamma remain unchanged.

2026-10-17 Codex (local, unreleased): the processed-DS sagittal panel no longer
depends on the widefield branch. A missing 5 um source records
``panel_errors.context`` for that neuron and the combined review figure is
written from the panels that exist (TriPlot with three panels, PairPlot with
two), instead of failing the neuron and losing the Sagittal view.
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
for p in (PROJECT_ROOT, NEUROVIS, MAIN_SCRIPTS, GROUP_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from Visual_toolkit import (  # noqa: E402
    Visual_toolkit, REVIEW_SETTINGS, apply_setting_overrides,
    review_settings_snapshot,
)
import IONData as IT  # noqa: E402
from bulk_visual_multi_monkey import (  # noqa: E402
    PARENT_OUT,
    load_selected_mega,
    selected_cohort,
    refined_sample_paths,
    ensure_refined_targets_empty,
    write_refined_sample_sources,
    write_refined_run_manifest,
    append_soma_crop_qc,
    load_registered_ds_swc,
    prepare_native_sagittal_panel,
    raw_swc_cache_path,
    neuron_visual_provenance,
    review_figure_suffix,
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
    cohort: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if cohort:
        all_sample_rows, meta = load_selected_mega(sample_id)
        keep = selected_cohort(all_sample_rows, cohort)
        meta["source"] = "active_selected_mega"
        meta["sample_rows"] = all_sample_rows
        meta["cohort"] = cohort
        keep["group"] = cohort
        print(
            f"[source] active selected-neuron workbook: {meta['workbook']} "
            f"({len(keep)} {cohort} rows for {sample_id})"
        )
        return keep, meta
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
    if meta.get("source") == "active_selected_mega":
        source_meta = {key: value for key, value in meta.items() if key != "sample_rows"}
        saved = write_refined_sample_sources(
            sample_id,
            meta["run_stamp"],
            meta["sample_rows"],
            source_meta,
            output_root=meta.get("output_root"),
        )
        print(f"[source] {saved['source_table']} ({len(meta['sample_rows'])} rows)")
        return saved["source_table"]
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
        cohort=args.cohort,
    )
    if args.cohort:
        meta["run_stamp"] = datestamp
        meta["output_root"] = args.output_root
        meta["soma_grid_radius"] = args.grid_radius
        if args.skip_existing:
            raise ValueError("Refined cohort runs cannot use --skip-existing; record every selected neuron explicitly")
    if args.context_sagittal:
        if not args.cohort:
            raise ValueError("--context-sagittal requires --cohort")
        from Visual_toolkit.toolkit import SAMPLE_PATHS
        if args.sample not in SAMPLE_PATHS:
            raise ValueError(f"No processed DS sagittal source is registered for {args.sample}")
    if args.smoke:
        nid = args.smoke if args.smoke.endswith(".swc") else f"{args.smoke}.swc"
        neurons = neurons[
            neurons["NeuronID"].astype(str).str.replace(".swc", "", regex=False)
            == nid.replace(".swc", "")
        ]
        if neurons.empty:
            if args.cohort:
                raise ValueError(
                    f"Smoke neuron {nid} is not in {args.sample}/{args.cohort} "
                    "in the active selected-neuron workbook"
                )
            neurons = pd.DataFrame([{"NeuronID": nid, "Soma_Region": args.group}])

    do_widefield = not args.soma_only
    do_soma = bool(args.cohort) or args.with_soma or args.soma_only

    import bulk_visual_multi_monkey as bvm

    bvm.GENERATE_HIGH_RES = do_soma and not args.png_only and not args.soma_only
    bvm.GENERATE_LOW_RES = do_widefield and not args.soma_only

    if args.cohort:
        run_base, dirs = refined_sample_paths(
            args.sample, datestamp, args.cohort, output_root=args.output_root
        )
        if not args.dry_run:
            ensure_refined_targets_empty(run_base, [args.cohort])
    else:
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
    if args.dry_run:
        counts = (
            meta["sample_rows"]["Atlas_INS"].map(bool).value_counts().to_dict()
            if args.cohort else {}
        )
        print({"sample": args.sample, "cohort": args.cohort, "rows": len(neurons),
               "atlas_ins_counts": counts, "output": run_base, "dry_run": True})
        return {"sample": args.sample, "cohort": args.cohort, "total": len(neurons),
                "ok": 0, "fail": 0, "dry_run": True}
    sagittal_source = None
    sagittal_trees = {}
    if args.context_sagittal:
        from Visual_toolkit.toolkit import prepare_sagittal_source
        for value in neurons["NeuronID"].astype(str):
            neuron_id = value if value.endswith(".swc") else f"{value}.swc"
            try:
                sagittal_trees[neuron_id] = load_registered_ds_swc(args.sample, neuron_id)
            except Exception:
                # The per-neuron loop records any missing source as an explicit failure.
                continue
        if sagittal_trees:
            xyz = [(tree.root.x, tree.root.y, tree.root.z)
                   for tree in sagittal_trees.values()]
            sagittal_source = prepare_sagittal_source(args.sample, xyz)
    ion = IT.IONData()
    toolkit = Visual_toolkit(args.sample)
    for d in dirs.values():
        if d is not None:
            os.makedirs(d, exist_ok=True)

    table_path = export_step1_table(neurons, run_base, args.sample, args.group, meta)

    ok = fail = partial = 0
    failed = []
    partial_neurons = []
    neuron_records = []
    desc = f"{args.sample}/{args.group}" + (" smoke" if args.smoke else "")
    try:
        for _, row in tqdm(neurons.iterrows(), total=len(neurons), desc=desc):
            neuron_id = str(row["NeuronID"])
            if not neuron_id.endswith(".swc"):
                neuron_id = f"{neuron_id}.swc"
            region_fields = (
                ("Soma_Region_Auto", "Soma_Region_Refined", "OwnRoot_Atlas_Label", "review_region")
                if args.cohort else ("Soma_Region",)
            )
            soma_region = next(
                (str(row[field]) for field in region_fields
                 if field in row and pd.notnull(row[field]) and str(row[field]).strip()),
                args.cohort or args.group,
            )
            try:
                skip_wf = (
                    args.skip_existing
                    and do_widefield
                    and _plot_exists(dirs["low_img"], args.sample, neuron_id, "Context")
                )
                skip_soma = (
                    args.skip_existing
                    and do_soma
                    and _plot_exists(dirs["high_img"], args.sample, neuron_id, "Soma")
                )
                if skip_wf and (not do_soma or skip_soma):
                    ok += 1
                    continue

                tree = (sagittal_trees.get(neuron_id) if args.context_sagittal
                        else safe_fetch_raw_swc(ion, args.sample, neuron_id))
                if tree is None:
                    raise ValueError(f"raw SWC not available for {neuron_id}")
                soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

                soma_nifti_path = None
                soma_panel = None
                widefield_panel = None
                styled_panel = None
                sagittal_meta = None
                panel_errors = {}
                if do_soma and not skip_soma:
                    if args.cached_only:
                        from Visual_toolkit import toolkit as toolkit_module
                        attempts = toolkit_module.HTTP_ATTEMPTS
                        toolkit_module.HTTP_ATTEMPTS = 0
                        try:
                            vol_h, org_h, res_h = toolkit.get_high_res_soma_window(soma_xyz)
                        finally:
                            toolkit_module.HTTP_ATTEMPTS = attempts
                    else:
                        vol_h, org_h, res_h = toolkit.get_high_res_soma_window(soma_xyz)
                    if float(vol_h.max()) <= 0:
                        raise ValueError("soma block volume is all zeros")
                    if not args.png_only:
                        soma_nifti_path = toolkit.export_data(
                            vol_h,
                            org_h,
                            res_h,
                            neuron_id,
                            suffix="SomaBlock",
                            soma_region=soma_region,
                            soma_coords=soma_xyz,
                            output_dir=dirs["high_nii"],
                        )
                    soma_panel = toolkit.plot_soma_block(
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
                    # A missing 5 um source, an empty crop or an incomplete section
                    # stack drops only the Context panel: the Soma and Sagittal panels
                    # and the combined review figure are still written, and the reason
                    # is recorded for this neuron instead of failing it.
                    try:
                        context = REVIEW_SETTINGS['context']
                        vol_l, org_l, res_l = toolkit.get_low_res_widefield(
                            soma_xyz, width_um=context['width_um'],
                            height_um=context['height_um'], depth_um=context['depth_um'])
                        if float(vol_l.max()) <= 0:
                            raise ValueError("widefield volume is all zeros")
                        if not args.png_only and not args.cohort:
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
                        widefield_panel = toolkit.plot_widefield_context(
                            vol_l,
                            org_l,
                            res_l,
                            soma_xyz,
                            neuron_id,
                            swc_tree=tree,
                            soma_region=soma_region,
                            output_dir=dirs["low_img"],
                        )
                    except Exception as exc:
                        panel_errors["context"] = str(exc)
                        print(f"  > [warn] Context panel unavailable: {exc}")
                if args.context_sagittal:
                    prepared = prepare_native_sagittal_panel(
                        args.sample, neuron_id, tree, source=sagittal_source
                    )
                    sagittal_meta = prepared["provenance"]
                    styled_panel = toolkit.plot_sagittal_context(
                        prepared["panel"], neuron_id, soma_region=soma_region,
                        soma_coords=soma_xyz, output_dir=dirs["low_img"],
                    )
                    sagittal_meta["styled_png"] = styled_panel["output_png"]

                panels = [soma_panel, widefield_panel, styled_panel]
                figure_suffix = review_figure_suffix(panels)
                if figure_suffix and sagittal_meta is not None:
                    figure_path = toolkit.plot_tri_plot(
                        panels, neuron_id, suffix=figure_suffix,
                        soma_region=soma_region, soma_coords=soma_xyz,
                        output_dir=dirs.get("trio") or dirs["low_img"],
                    )
                    sagittal_meta["tri_plot_png"] = figure_path
                    sagittal_meta["tri_plot_sha256"] = bvm._sha256(figure_path)
                    sagittal_meta["review_figure_suffix"] = figure_suffix
                    sagittal_meta["review_figure_panels"] = [
                        panel["role"] for panel in panels if panel]
                if args.cohort:
                    raw_path = (str(bvm.native_ds_swc_path(args.sample, neuron_id))
                                if args.context_sagittal
                                else raw_swc_cache_path(args.sample, neuron_id))
                    record = neuron_visual_provenance(
                        toolkit, args.sample, neuron_id, soma_region, dirs,
                        raw_path, soma_nifti_path, context_sagittal=sagittal_meta,
                    )
                    if panel_errors:
                        record["panel_errors"] = panel_errors
                    neuron_records.append(record)
                if panel_errors:
                    partial += 1
                    partial_neurons.append(
                        f"{neuron_id}: " + "; ".join(
                            f"{name}: {reason}" for name, reason in panel_errors.items()))
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

    if failed and not args.cohort:
        log = os.path.join(run_base, f"failed_neurons_{args.group}.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))

    result = dict(
        sample=args.sample,
        group=args.cohort or args.group,
        total=len(neurons),
        ok=ok,
        fail=fail,
        partial=partial,
        partial_neurons=partial_neurons,
        source_readme=table_path if args.cohort else None,
        soma_png=dirs["high_img"] if do_soma else None,
        widefield_png=dirs["low_img"] if do_widefield else None,
    )
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    if args.cohort:
        group_counts = meta["sample_rows"]["Atlas_INS"].map(bool).value_counts().to_dict()
        manifest_meta = {key: value for key, value in meta.items() if key != "sample_rows"}
        # The resolved display/window settings travel with the run, not just the defaults.
        manifest_meta["review_settings"] = review_settings_snapshot()
        append_soma_crop_qc(
            table_path,
            [{"cohorts": [{"cohort": args.cohort, "neurons": neuron_records}]}],
        )
        manifest = write_refined_run_manifest(
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            datestamp,
            manifest_meta,
            [{"sample_id": args.sample,
              "atlas_neurons": int(group_counts.get(True, 0)),
              "candidate_neurons": int(group_counts.get(False, 0)),
              "rendered_cohort": args.cohort,
              "rendered_count": len(neurons), "ok": ok, "fail": fail,
              "failed_neurons": failed,
                   "partial": partial,
                   "partial_neurons": partial_neurons,
              "source_readme": table_path,
              "neurons": neuron_records,
              "output_dirs": dirs}],
            output_root=args.output_root,
        )
        print(f"[manifest] {manifest}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", default="INS", choices=["INS", "PrCO", "Unknown"])
    parser.add_argument(
        "--cohort", choices=["Atlas_neurons", "Candidate_neurons"],
        help="Use the current active 462-neuron workbook and the source Atlas_INS split",
    )
    parser.add_argument(
        "--output-root",
        help="Override the refined output root (for bounded local pilots); defaults to the fMOST_Plots server folder",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show selected counts and paths without rendering or writing")
    parser.add_argument("--context-sagittal", action="store_true",
                        help="Also render the configured processed-DS sagittal PNG into Context/")
    parser.add_argument("--cached-only", action="store_true",
                        help="Do not request missing Soma blocks over HTTP")
    parser.add_argument(
        "--groups",
        help="Comma-separated groups to run in sequence (overrides --group), e.g. INS,PrCO",
    )
    parser.add_argument("--png-only", action="store_true", help="PNG only (no NIfTI export)")
    parser.add_argument(
        "--with-soma",
        action="store_true",
        help="Also render the Soma panel PNGs (HTTP 0.65 µm blocks)",
    )
    parser.add_argument(
        "--soma-only",
        action="store_true",
        help="Soma panel only (skip Context)",
    )
    parser.add_argument(
        "--grid-radius",
        type=int,
        default=1,
        help="Legacy block-grid radius, recorded in the manifest only: the Soma "
             "panel uses the soma-centred multi-cube window from "
             "REVIEW_SETTINGS['soma_window'] (get_high_res_soma_window)",
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
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="Override one REVIEW_SETTINGS leaf for this run, e.g. "
                             "--set widefield.gain=1.6 --set marker.crosshair_px=80 "
                             "(repeatable; recorded in the manifest)")
    args = parser.parse_args()
    try:
        applied = apply_setting_overrides(args.set)
    except (KeyError, ValueError) as exc:
        parser.error(exc.args[0] if exc.args else str(exc))
    if applied:
        print(f"[settings] overrides applied: {applied}")
    args.combined_fallback = not args.no_combined_fallback
    if not args.cohort and not os.path.exists(PARENT_OUT):
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
