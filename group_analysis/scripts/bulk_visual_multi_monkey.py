"""
bulk_visual_multi_monkey.py

Version: 1.8.0 (local/unreleased; VisualToolkit package wrapper)
Last updated: 2026-10-17
Change history: main_scripts/Visual_toolkit/CHANGELOG.md (package changes) + notes below

Wraps step3.1.bulk_visual_data.py logic to run on ALL new monkey samples
(251730, 252383, 252384, 252385) for three groups:
  - INS     : any atlas-derived insula sub-region (Ial, Iai, Iam/Iapm, Iapl,
              lat_Ia, Ia/Id, Ig, Pi, Ins, Ins/Pi, etc.)
  - PrCO    : auto-labeled PrCO
  - Unknown : Soma_Region empty / contains "Unknown"

Output goes to \\\\10.102.8.200\\microscopy_data\\fMOST\\visual\\{SAMPLE_ID}\\...

Skips 251637 entirely.

2026-10-16 Codex (local, unreleased): add opt-in ``--selected-cohorts`` output
from the active 462-row workbook, split by source Atlas_INS, into the refined
Soma/Soma_nifti/Context layout with one shared provenance manifest. Legacy
runner/default paths remain unchanged; high-resolution gamma is unchanged.

2026-10-17 Codex (local, unreleased): the optional sagittal context now uses the
Visual_toolkit legacy panel style (same black-background style as the Soma and
Widefield plots) and each neuron also gets one tri-plot review PNG combining
Soma, WideField and Sagittal with a single shared legend. The previous PIL
sagittal renderer stays available in main_scripts/Visual_toolkit/sagittal.py as the
historical full-plane product. A/B raw-SWC mapping candidates remain unresolved.

2026-10-17 Codex (local, unreleased): an unavailable Context panel (a sample with
no 5 um resampled source, or an empty/incomplete crop) no longer fails the
neuron. The reason is recorded per neuron under ``panel_errors`` and the
combined review figure is written from the panels that exist (``TriPlot`` for
three panels, ``PairPlot`` for two), so Soma and Sagittal panels are never lost
to a missing Context source. 252383 and 252718 have no 5 um resampled source on
the share (checked 2026-10-17).
"""
from __future__ import annotations

import os
import sys
import glob
import traceback
import json
import hashlib
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
MAIN_SCRIPTS = os.path.join(PROJECT_ROOT, "main_scripts")
GROUP_SCRIPTS = os.path.join(PROJECT_ROOT, "group_analysis", "scripts")
for p in (PROJECT_ROOT, NEUROVIS, MAIN_SCRIPTS, GROUP_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports must happen after path setup
from Visual_toolkit import (  # noqa: E402
    Visual_toolkit, REVIEW_SETTINGS, apply_setting_overrides,
    review_settings_snapshot,
)
import IONData as IT  # noqa: E402
from insula_label_set import build_insula_label_set, normalize_label, strip_prefix  # noqa: E402
from cohort import NEW_SAMPLES  # noqa: E402

STEP1_DIR   = os.path.join(PROJECT_ROOT, "group_analysis", "step1_results")
FMOST_SHARE_ROOT = r"\\10.102.8.200\microscopy_data\fMOST"
PARENT_OUT = os.path.join(FMOST_SHARE_ROOT, "visual")
REFINED_OUT = os.path.join(FMOST_SHARE_ROOT, "fMOST_Plots")
DATA_PROGRESS_DIR = os.path.join(PROJECT_ROOT, "group_analysis", "data_progress")
ACTIVE_PROGRESS_JSON = os.path.join(DATA_PROGRESS_DIR, "active.json")

GENERATE_HIGH_RES = True
GENERATE_LOW_RES  = True

INSULA_LABELS, _ = build_insula_label_set()
DATESTAMP = datetime.now().strftime("%Y%m%d")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path: str | None) -> dict:
    if not path:
        return {"path": None, "sha256": None, "status": "not produced"}
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return {"path": path, "sha256": None, "status": "missing at receipt time"}
    return {"path": path, "sha256": _sha256(path), "bytes": os.path.getsize(path),
            "status": "hashed"}


def raw_swc_cache_path(sample_id: str, neuron_id: str) -> str:
    return os.path.abspath(os.path.join("../resource/swc_raw", str(sample_id), str(neuron_id)))


def newest_plot(folder: str, sample_id: str, neuron_id: str, region_suffix: str,
                role: str) -> str | None:
    """Newest timestamped plot for one role (names keep the role suffix last)."""
    pattern = os.path.join(folder, f"{sample_id}_{neuron_id}{region_suffix}_*_{role}_Plot.png")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def review_figure_suffix(panels) -> str | None:
    """Suffix for the combined review figure, or None when fewer than two panels exist."""
    available = [panel for panel in panels if panel]
    return {3: "TriPlot", 2: "PairPlot"}.get(len(available))


def neuron_visual_provenance(toolkit, sample_id: str, neuron_id: str,
                             soma_region: str, dirs: dict,
                             raw_swc_path: str, soma_nifti_path: str | None,
                             context_sagittal: dict | None = None) -> dict:
    """Bind each rendered neuron to its raw graph, acquisitions and outputs."""
    safe_region = toolkit._sanitize_filename(soma_region) if soma_region else ""
    suffix = f"_{safe_region}" if safe_region else ""
    highres = toolkit.last_high_res_metadata or {}
    requested = [tuple(index) for index in highres.get("requested", [])]
    loaded = {tuple(index) for index in highres.get("loaded", [])}
    cache_blocks = []
    for x, y, z in requested:
        cached_path = os.path.join(toolkit.cache_http_dir, str(z), f"{x}_{y}_{z}.tif")
        cache_blocks.append({
            "index_xyz": [x, y, z],
            "loaded_by_acquisition": (x, y, z) in loaded,
            "cache_file": file_provenance(cached_path),
        })
    center = highres.get("center_xyz_um")
    origin = highres.get("origin_xyz_um")
    spacing = highres.get("spacing_xyz_um")
    shape = highres.get("shape_zyx")
    crop_position = None
    if center and origin and spacing and shape:
        x_px = (center[0] - origin[0]) / spacing[0]
        y_px = (center[1] - origin[1]) / spacing[1]
        width, height = int(shape[2]), int(shape[1])
        crop_position = {
            "image_size_xy_px": [width, height],
            "soma_marker_xy_px": [round(x_px, 3), round(y_px, 3)],
            "edge_clearance_px": {
                "left": round(x_px, 3), "right": round(width - 1 - x_px, 3),
                "top": round(y_px, 3), "bottom": round(height - 1 - y_px, 3),
                "minimum": round(min(x_px, y_px, width - 1 - x_px, height - 1 - y_px), 3),
            },
            "interpretation": "Source-root marker position only; no acceptance threshold. A complete source cube does not establish that the soma is fully visible in the 2D projection.",
        }
    return {
        "sample_id": str(sample_id), "neuron_id": str(neuron_id),
        "source_region_label": str(soma_region),
        "soma_xyz_um": list(toolkit.last_high_res_metadata.get("center_xyz_um", []))
        if toolkit.last_high_res_metadata else None,
        "source_acquisition_dates": "unknown in the registered source records",
        "raw_swc": file_provenance(raw_swc_path),
        "high_res_acquisition": highres,
        "high_res_cache_blocks": cache_blocks,
        "soma_mip_crop_position_qc": crop_position,
        "context_acquisition": toolkit.last_low_res_metadata,
        "outputs": {
            "soma_png": file_provenance(
                newest_plot(dirs["high_img"], sample_id, neuron_id, suffix, "Soma")),
            "soma_nifti": file_provenance(soma_nifti_path),
            "soma_nifti_metadata": file_provenance(
                f"{soma_nifti_path}.json" if soma_nifti_path else None),
            "context_png": file_provenance(
                newest_plot(dirs["low_img"], sample_id, neuron_id, suffix, "Context")),
            "sagittal_png": file_provenance(
                (context_sagittal or {}).get("styled_png")),
            "tri_plot_png": file_provenance(
                (context_sagittal or {}).get("tri_plot_png")),
        },
        "sagittal": context_sagittal,
    }


def load_selected_mega(sample_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load the current selected-neuron table through the active pointer.

    The workbook hash is checked against active.json before its exact
    SampleID/NeuronID rows are used. Source annotations and region fields stay
    in the returned table unchanged.
    """
    with open(ACTIVE_PROGRESS_JSON, "r", encoding="utf-8") as stream:
        active = json.load(stream)
    workbook = os.path.normpath(os.path.join(DATA_PROGRESS_DIR, active["workbook"]))
    if not os.path.isfile(workbook):
        raise FileNotFoundError(f"Active selected-neuron workbook is missing: {workbook}")
    expected = active.get("artifacts", {}).get(active["workbook"])
    actual = _sha256(workbook)
    if not expected or actual.lower() != expected.lower():
        raise ValueError(
            f"Active workbook SHA256 mismatch: expected {expected}, observed {actual}"
        )
    neurons = pd.read_excel(
        workbook, sheet_name="neurons", dtype={"SampleID": str, "NeuronID": str}
    )
    required = {"SampleID", "NeuronID", "Atlas_INS", "Soma_Region_Auto",
                "Soma_Region_Refined", "EvidenceStratum"}
    missing = sorted(required.difference(neurons.columns))
    if missing:
        raise ValueError(f"Active workbook is missing required fields: {missing}")
    if neurons.duplicated(["SampleID", "NeuronID"]).any():
        raise ValueError("Active workbook has duplicate (SampleID, NeuronID) identities")
    if neurons["Atlas_INS"].isna().any():
        raise ValueError("Atlas_INS contains unknown values; refusing implicit classification")
    if sample_id is not None:
        neurons = neurons[neurons["SampleID"].astype(str).eq(str(sample_id))].copy()
        if neurons.empty:
            raise ValueError(f"No selected-neuron rows for sample {sample_id} in {workbook}")
    meta = {
        "active_json": ACTIVE_PROGRESS_JSON,
        "workbook": workbook,
        "workbook_sha256": actual,
        "source_date": active.get("source_date"),
        "updated_date": active.get("updated_date"),
        "generated_at": active.get("generated_at"),
        "total_rows": int(len(neurons)),
    }
    return neurons, meta


def selected_cohort(neurons: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Apply the source ARM-insula boolean; Candidate is its exact complement."""
    if cohort not in {"Atlas_neurons", "Candidate_neurons"}:
        raise ValueError(f"Unsupported selected cohort: {cohort}")
    atlas = neurons["Atlas_INS"].map(_parse_source_bool)
    keep = atlas if cohort == "Atlas_neurons" else ~atlas
    return neurons.loc[keep].copy()


def _parse_source_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Atlas_INS contains a value other than true/false: {value!r}")


def refined_sample_paths(sample_id: str, run_stamp: str, cohort: str,
                         output_root: str | None = None) -> tuple[str, dict]:
    """Return the requested sample/cohort path and Soma/Context Data/Plots."""
    root = output_root or REFINED_OUT
    sample_root = os.path.join(root, f"{sample_id}_{run_stamp}")
    cohort_root = os.path.join(sample_root, cohort)
    dirs = {
        "high_img": os.path.join(cohort_root, "Soma"),
        "high_nii": os.path.join(cohort_root, "Soma_nifti"),
        "low_img": os.path.join(cohort_root, "Context"),
        "trio": os.path.join(cohort_root, "TriPlots"),
        "low_nii": None,
    }
    return sample_root, dirs


def ensure_refined_targets_empty(sample_root: str, cohorts: list[str]) -> None:
    """Refuse to replace any existing sample-level receipt or rendered file."""
    if os.path.isfile(os.path.join(sample_root, "README.md")):
        raise FileExistsError(f"Refusing to replace existing sample README: {sample_root}")
    for cohort in cohorts:
        target = os.path.join(sample_root, cohort)
        if not os.path.isdir(target):
            continue
        existing = next((os.path.join(root, name)
                         for root, _, files in os.walk(target) for name in files), None)
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing bulk visual output: {existing}")


def native_ds_swc_path(sample_id: str, neuron_id: str):
    """Resolve the sagittal raw SWC: registered cache first, then the legacy cache."""
    from Visual_toolkit.sagittal import SAMPLE_PATHS, resolve_swc_path

    sample_id = str(sample_id)
    if sample_id not in SAMPLE_PATHS:
        raise ValueError(f"No processed DS sagittal source is registered for {sample_id}")
    bare_id = str(neuron_id).removesuffix(".swc")
    swc_path, _source = resolve_swc_path(sample_id, bare_id,
                                         SAMPLE_PATHS.get(sample_id) or {})
    if swc_path is None:
        raise FileNotFoundError(
            f"Registered cached raw SWC is unavailable for {sample_id}/{bare_id}")
    return swc_path


def load_registered_ds_swc(sample_id: str, neuron_id: str):
    """Load the adapter-bound cached raw SWC for rendering and soma matching."""
    from SwcLoader import NeuronTree
    swc_path = native_ds_swc_path(sample_id, neuron_id)
    tree = NeuronTree()
    tree.readFile(str(swc_path))
    return tree


def prepare_native_sagittal_panel(sample_id: str, neuron_id: str, tree, source=None,
                                  crop_half_width_mm: float | None = None) -> dict:
    """Prepare the native DS sagittal panel for the shared legacy review style."""
    from Visual_toolkit.sagittal import prepare_sagittal_panel

    bare_id = str(neuron_id).removesuffix(".swc")
    swc_path = native_ds_swc_path(sample_id, neuron_id)
    crop = {} if crop_half_width_mm is None else {"crop_half_width_mm": crop_half_width_mm}
    return prepare_sagittal_panel(
        sample_id, bare_id, (tree.root.x, tree.root.y, tree.root.z),
        swc_path=swc_path, source=source, **crop,
    )


def write_refined_sample_sources(sample_id: str, run_stamp: str,
                                 neurons: pd.DataFrame, meta: dict,
                                 output_root: str | None = None) -> dict:
    """Save one source table and README at the sample/date root."""
    root = output_root or REFINED_OUT
    sample_root = os.path.join(root, f"{sample_id}_{run_stamp}")
    os.makedirs(sample_root, exist_ok=True)
    rows = neurons.copy()
    rows["visual_group"] = rows["Atlas_INS"].map(
        lambda value: "Atlas_neurons" if _parse_source_bool(value) else "Candidate_neurons"
    )
    counts = rows["visual_group"].value_counts().to_dict()
    readme_path = os.path.join(sample_root, "README.md")
    with open(readme_path, "w", encoding="utf-8") as stream:
        stream.write(f"# {sample_id} bulk visual source\n\n")
        stream.write(f"Production date: {run_stamp}\n\n")
        stream.write("## Source\n\n")
        stream.write(f"- Active pointer: `{meta['active_json']}`\n")
        stream.write(f"- Workbook: `{meta['workbook']}`\n")
        stream.write(f"- Workbook SHA256: `{meta['workbook_sha256']}`\n")
        stream.write(f"- Declared source date: `{meta['source_date']}`\n")
        stream.write(f"- Export updated date: `{meta['updated_date']}`\n")
        stream.write(f"- Workbook generated_at: `{meta['generated_at']}`\n")
        stream.write("- Source acquisition date: unknown in the active workbook.\n")
        stream.write("- Identity key: exact `(SampleID, NeuronID)` from the `neurons` sheet.\n")
        stream.write("- Classification source table: the compact table below is derived from the selected workbook; exact IDs and source fields are retained here, while the source workbook remains unmodified.\n\n")
        if meta.get("manifest_filename"):
            stream.write(f"- Run provenance: [../{meta['manifest_filename']}](../{meta['manifest_filename']})\n\n")
        stream.write("## Classification and counts\n\n")
        stream.write("`Atlas_neurons` uses source workbook `Atlas_INS == true`, copied from the registered full-inventory `atlas_INS` field. `Candidate_neurons` is the exact selected-row complement (`Atlas_INS == false`). `EvidenceStratum`, region labels, ARM/source fields, and review fields are retained as source context and do not override this split. This is a display grouping, not anatomical acceptance.\n\n")
        stream.write(f"- Atlas_neurons: {counts.get('Atlas_neurons', 0)}\n")
        stream.write(f"- Candidate_neurons: {counts.get('Candidate_neurons', 0)}\n\n")
        stream.write("## Methods and provenance\n\n")
        stream.write("| Product | Native source | Display/export and limits |\n| --- | --- | --- |\n")
        soma_window = REVIEW_SETTINGS['soma_window']
        context_settings = REVIEW_SETTINGS['context']
        stream.write(f"| Soma | High-resolution cube cache; nominal 0.65 × 0.65 × 3 µm. One soma-centred window per neuron: {soma_window['cubes_xy']:g} cubes wide, using the largest step whose cubes could all be loaded (per-neuron cube paths/hashes, window_centered flag and centre offset are in the shared manifest). | PNG: legacy MIP, 0.5–99.5 percentile, gamma 0.5, cyan target box + cross. Sibling NIfTI preserves native XYZ micrometre geometry with JSON sidecar. |\n")
        stream.write(f"| Widefield Context | Native 5 × 5 × 3 µm TIFF sections from the configured sample path. Exact paths and loaded coverage are recorded; full TIFF hashes are not computed. | {context_settings['width_um']:,} × {context_settings['height_um']:,} × {context_settings['depth_um']:,} µm display crop, gain {REVIEW_SETTINGS['widefield']['gain']:g}, threshold {REVIEW_SETTINGS['widefield']['threshold']:g}, cyan target box + cross. PNG only; no Context NIfTI/volume export. |\n")
        stream.write("| Sagittal Context | Manifest-selected processed DS00 N4 image; path/hash are in the shared manifest. Nominal native grid 100 × 90 × 100 µm. | Legacy panel style (same as Soma/Context), whole constant-i plane so both candidate somas stay in view; the shared cyan box + cross marks the exact candidate-A target position (drawn larger than scale, not to scale); A/B candidate traces dashed. SWC-to-image axes, anatomical laterality, and plane acceptance remain unresolved; no resampling. |\n")
        stream.write("| Combined review figure | The PNGs above, redrawn side by side in one figure (no legend band; each panel title carries its source dataset). | One check per neuron instead of three separate PNGs; per-neuron detail PNGs remain the authoritative single views. Suffix `TriPlot` when all three panels exist, `PairPlot` when one panel's source is unavailable (the missing panel and reason stay in the per-neuron receipt). |\n\n")
        stream.write("Panel titles use `sample | neuron | <panel> | src_dataset: <source dataset>`, i.e. `Soma Plot | src_dataset: 0.65µm_raw`, `Context Plot | src_dataset: 5µm_localserver`, `Sagittal Context Plot | src_dataset: DS00 N4`. Files are named `<sample>_<neuron>_<region>_<YYYYMMDD-HHMMSS>_Soma_Plot.png` (same pattern for `_Context_Plot`, `_Sagittal_Plot`, `_TriPlot` / `_PairPlot`); the timestamp is per figure and the role suffix stays last, so suffix readers still match. The soma NIfTI sibling is `_Soma.nii.gz`. Legacy `SomaBlock`/`WideField` names are the internal call-site suffixes only.\n\n")
        stream.write("Source acquisition dates are unknown. The manifest also records raw SWC and rendered-output hashes; widefield TIFF content hashes were not computed this run.\n\n")
        stream.write("## Output layout\n\n")
        stream.write("- [Atlas_neurons/Soma/](Atlas_neurons/Soma/): high-resolution soma PNGs\n")
        stream.write("- [Atlas_neurons/Soma_nifti/](Atlas_neurons/Soma_nifti/): high-resolution soma NIfTIs\n")
        stream.write("- [Atlas_neurons/Context/](Atlas_neurons/Context/): context PNGs plus the optional sagittal panel; no context volumes\n")
        stream.write("- [Atlas_neurons/TriPlots/](Atlas_neurons/TriPlots/): one combined review PNG per neuron (`TriPlot`, or `PairPlot` when a panel source is unavailable)\n")
        stream.write("- [Candidate_neurons/Soma/](Candidate_neurons/Soma/): high-resolution soma PNGs\n")
        stream.write("- [Candidate_neurons/Soma_nifti/](Candidate_neurons/Soma_nifti/): high-resolution soma NIfTIs\n")
        stream.write("- [Candidate_neurons/Context/](Candidate_neurons/Context/): context PNGs plus the optional sagittal panel; no context volumes\n")
        stream.write("- [Candidate_neurons/TriPlots/](Candidate_neurons/TriPlots/): one combined review PNG per neuron (`TriPlot`, or `PairPlot` when a panel source is unavailable)\n\n")
        columns = [
            name for name in (
                "SampleID", "NeuronUID", "NeuronID", "visual_group", "Atlas_INS", "EvidenceStratum",
                "Soma_Region_Auto", "Soma_Region_Refined", "Soma_Region_Source",
                "OwnRoot_Atlas_Label", "Soma_Side", "review_region", "review_status",
                "review_notes", "review_reviewer", "review_date", "review_hemisphere",
                "review_side", "review_method",
            ) if name in rows.columns
        ]
        stream.write("## Source rows\n\n")
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in rows[columns].iterrows():
            cells = []
            for name in columns:
                value = row[name]
                if pd.isna(value):
                    value = ""
                value = str(value).replace("|", "\\|").replace("\n", " ")
                cells.append(value)
            stream.write("| " + " | ".join(cells) + " |\n")
    return {"sample_root": sample_root, "source_table": readme_path,
            "readme": readme_path, "counts": counts}


def write_refined_run_manifest(run_id: str, run_stamp: str, meta: dict,
                               sample_records: list[dict], output_root: str | None = None) -> str:
    root = output_root or REFINED_OUT
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, f"bulk_visual_manifest_{run_id}.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "production_date": run_stamp,
        "source": meta,
        "execution": meta.get("run_options", {}),
        "classification": {
            "atlas_group": "Atlas_INS == true (source inventory atlas_INS)",
            "candidate_group": "Atlas_INS == false (exact complement within selected workbook rows)",
        },
        "rendering": {
            "soma": "Visual_toolkit.plot_soma_block on a soma-centred multi-cube window (get_high_res_soma_window, REVIEW_SETTINGS['soma_window'], no resampling); legacy gamma unchanged; NIfTI saved beside PNGs in sibling Soma_nifti/",
            "soma_window": "centred on the fractional soma position; per-neuron window_centered flag, window_cubes_xy and window_centre_offset_px are in the receipt",
            "high_res_gamma": "legacy 0.5 p0.5-p99.5 MIP",
            "context": f"Visual_toolkit.plot_widefield_context; gain {REVIEW_SETTINGS['widefield']['gain']:g} and threshold {REVIEW_SETTINGS['widefield']['threshold']:g} via REVIEW_SETTINGS['widefield']; {REVIEW_SETTINGS['context']['width_um']}x{REVIEW_SETTINGS['context']['height_um']}x{REVIEW_SETTINGS['context']['depth_um']} um display crop only",
            "context_volumes": "not exported",
            "marker": "one cyan/turquoise open box plus a cross through the exact point in every panel (REVIEW_SETTINGS['marker']); the drawn size is enlarged for visibility and is not to scale",
            "sagittal": "Visual_toolkit.sagittal.prepare_sagittal_panel + Visual_toolkit.plot_sagittal_context; whole constant-i plane, legacy panel style, A/B candidate traces",
            "tri_plot": "Visual_toolkit.plot_tri_plot; one combined review PNG per neuron (Soma | Context | Sagittal), legend off by default; suffix TriPlot for three panels, PairPlot when one panel's source is unavailable (named per neuron under panel_errors)",
        },
        "samples": sample_records,
        "anatomical_acceptance": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest_path


def append_soma_crop_qc(readme_path: str, sample_records: list[dict]) -> None:
    """Append source-root marker positions, explicitly separate from visual acceptance."""
    with open(readme_path, "a", encoding="utf-8") as stream:
        stream.write("## Soma MIP crop-position QC\n\n")
        stream.write("Marker coordinates are derived from the raw root and the recorded cube origin/spacing. A complete cube acquisition does not guarantee that the soma is fully visible in the 2D MIP. No pixel acceptance threshold is applied; review each crop visually.\n\n")
        stream.write("| Cohort | NeuronID | Marker x,y (px) | Minimum edge clearance (px) | Visual crop review |\n")
        stream.write("| --- | --- | ---: | ---: | --- |\n")
        for sample in sample_records:
            for cohort in sample.get("cohorts", []):
                for neuron in cohort.get("neurons", []):
                    qc = neuron.get("soma_mip_crop_position_qc")
                    if not qc:
                        continue
                    x, y = qc["soma_marker_xy_px"]
                    clearance = qc["edge_clearance_px"]["minimum"]
                    review = qc.get("visual_review_status", "not individually reviewed by this run")
                    stream.write(f"| {cohort['cohort']} | {neuron['neuron_id'].removesuffix('.swc')} | {x:.1f}, {y:.1f} | {clearance:.1f} | {review} |\n")
        stream.write("\n")


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


def process_neuron(toolkit, ion, sample_id, neuron_id, soma_region, dirs,
                   export_context_volume=True, context_sagittal=False,
                   cached_only=False, sagittal_source=None, preloaded_tree=None):
    tree = (preloaded_tree if preloaded_tree is not None else
            load_registered_ds_swc(sample_id, neuron_id) if context_sagittal
            else safe_fetch_raw_swc(ion, sample_id, neuron_id))
    if tree is None:
        raise ValueError(f"raw SWC not available for {sample_id}/{neuron_id} (skipped after retries)")
    soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
    soma_nifti_path = None
    sagittal_meta = None
    soma_panel = None
    widefield_panel = None
    styled_panel = None
    panel_errors = {}

    if GENERATE_HIGH_RES:
        if cached_only:
            from Visual_toolkit import toolkit as toolkit_module
            attempts = toolkit_module.HTTP_ATTEMPTS
            toolkit_module.HTTP_ATTEMPTS = 0
            try:
                vol_h, org_h, res_h = toolkit.get_high_res_soma_window(soma_xyz)
            finally:
                toolkit_module.HTTP_ATTEMPTS = attempts
        else:
            vol_h, org_h, res_h = toolkit.get_high_res_soma_window(soma_xyz)
        soma_nifti_path = toolkit.export_data(
            vol_h, org_h, res_h, neuron_id,
            suffix="SomaBlock", soma_region=soma_region,
            soma_coords=soma_xyz, output_dir=dirs["high_nii"],
        )
        soma_panel = toolkit.plot_soma_block(
            vol_h, org_h, res_h, soma_xyz, neuron_id,
            suffix="SomaBlock", soma_region=soma_region,
            output_dir=dirs["high_img"],
        )

    if GENERATE_LOW_RES:
        # A missing 5 um source, an empty crop or an incomplete section stack drops
        # only the Context panel: the Soma and Sagittal panels plus the combined
        # review figure are still written and the reason is recorded per neuron.
        try:
            context = REVIEW_SETTINGS['context']
            vol_l, org_l, res_l = toolkit.get_low_res_widefield(
                soma_xyz, width_um=context['width_um'], height_um=context['height_um'],
                depth_um=context['depth_um'])
            if export_context_volume and dirs.get("low_nii"):
                toolkit.export_data(
                    vol_l, org_l, res_l, neuron_id,
                    suffix="WideField", soma_region=soma_region,
                    soma_coords=soma_xyz, output_dir=dirs["low_nii"],
                )
            widefield_panel = toolkit.plot_widefield_context(
                vol_l, org_l, res_l, soma_xyz, neuron_id,
                swc_tree=tree,
                soma_region=soma_region, output_dir=dirs["low_img"],
            )
        except Exception as exc:
            panel_errors["context"] = str(exc)
            print(f"  > [warn] Context panel unavailable for {sample_id}/{neuron_id}: {exc}")
    if context_sagittal:
        prepared = prepare_native_sagittal_panel(
            sample_id, neuron_id, tree, source=sagittal_source
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
        sagittal_meta["tri_plot_sha256"] = _sha256(figure_path)
        sagittal_meta["review_figure_suffix"] = figure_suffix
        sagittal_meta["review_figure_panels"] = [panel["role"] for panel in panels if panel]
    if not export_context_volume:
        raw_path = (str(native_ds_swc_path(sample_id, neuron_id)) if context_sagittal
                    else raw_swc_cache_path(sample_id, neuron_id))
        record = neuron_visual_provenance(
            toolkit, sample_id, neuron_id, soma_region, dirs, raw_path,
            soma_nifti_path, context_sagittal=sagittal_meta,
        )
        if panel_errors:
            record["panel_errors"] = panel_errors
        return record


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


def process_selected_cohorts(sample_ids: list[str] | None, cohorts: list[str],
                             run_stamp: str, output_root: str | None,
                             dry_run: bool, context_sagittal: bool = False,
                             cached_only: bool = False) -> dict:
    """Export current selected rows into the source-Atlas_INS folder layout."""
    all_rows, meta = load_selected_mega()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta["manifest_filename"] = f"bulk_visual_manifest_{run_id}.json"
    meta["run_options"] = {
        "selected_samples": sample_ids,
        "cohorts": cohorts,
        "production_date": run_stamp,
        "high_resolution_cached_only": bool(cached_only),
        "sagittal_context": bool(context_sagittal),
        "context_volumes_exported": False,
        "soma_window": "soma-centred multi-cube window (get_high_res_soma_window); per-neuron window_centered flag, cube count and centre offset recorded",
    }
    meta["provenance_limits"] = {
        "source_acquisition_dates": "unknown in the active source records",
        "widefield_source_tiff_sha256": "not computed this run; exact paths and loaded section indices are recorded per neuron",
    }
    # The resolved display/window settings travel with the run, not just the defaults.
    meta["review_settings"] = review_settings_snapshot()
    requested = sorted(all_rows["SampleID"].astype(str).unique())
    if sample_ids:
        absent = sorted(set(sample_ids).difference(requested))
        if absent:
            raise ValueError(f"Samples absent from the active selected workbook: {absent}")
        requested = sample_ids
    if context_sagittal:
        from Visual_toolkit.sagittal import SAMPLE_PATHS
        unsupported = sorted(set(requested).difference(SAMPLE_PATHS))
        if unsupported:
            raise ValueError(
                f"No processed-DS sagittal source is registered for samples {unsupported}"
            )
    records = []
    for sample_id in requested:
        sample_rows = all_rows[all_rows["SampleID"].astype(str).eq(sample_id)].copy()
        sagittal_source = None
        sagittal_trees = {}
        sagittal_failures = {}
        if context_sagittal and not dry_run:
            from Visual_toolkit.sagittal import prepare_sagittal_source

            for neuron_id in sample_rows["NeuronID"].astype(str):
                if not neuron_id.endswith(".swc"):
                    neuron_id += ".swc"
                try:
                    tree = load_registered_ds_swc(sample_id, neuron_id)
                    sagittal_trees[neuron_id] = tree
                except Exception as exc:
                    sagittal_failures[neuron_id] = f"raw SWC unavailable: {exc}"
            if sagittal_trees:
                xyz = [(tree.root.x, tree.root.y, tree.root.z)
                       for tree in sagittal_trees.values()]
                sagittal_source = prepare_sagittal_source(sample_id, xyz)
        sample_root = os.path.join(output_root or REFINED_OUT, f"{sample_id}_{run_stamp}")
        if not dry_run:
            ensure_refined_targets_empty(sample_root, cohorts)
        sources = write_refined_sample_sources(
            sample_id, run_stamp, sample_rows, meta, output_root=output_root
        ) if not dry_run else None
        source_atlas = sample_rows["Atlas_INS"].map(_parse_source_bool)
        item = {
            "sample_id": sample_id,
            "atlas_neurons": int(source_atlas.sum()),
            "candidate_neurons": int((~source_atlas).sum()),
            "source_readme": sources["readme"] if sources else None,
            "cohorts": [],
        }
        for cohort in cohorts:
            neurons = selected_cohort(sample_rows, cohort)
            _, dirs = refined_sample_paths(
                sample_id, run_stamp, cohort, output_root=output_root
            )
            cohort_record = {"cohort": cohort, "selected": int(len(neurons)),
                             "output_dirs": dirs, "ok": 0, "fail": 0,
                             "partial": 0, "partial_neurons": [], "neurons": []}
            if dry_run:
                item["cohorts"].append(cohort_record)
                continue
            for path in dirs.values():
                if path:
                    os.makedirs(path, exist_ok=True)
            toolkit = Visual_toolkit(sample_id)
            ion = IT.IONData()
            failures = []
            try:
                for _, row in tqdm(neurons.iterrows(), total=len(neurons),
                                   desc=f"{sample_id}/{cohort}"):
                    neuron_id = str(row["NeuronID"])
                    if not neuron_id.endswith(".swc"):
                        neuron_id += ".swc"
                    region = next(
                        (str(row[name]) for name in
                         ("Soma_Region_Auto", "Soma_Region_Refined", "OwnRoot_Atlas_Label", "review_region")
                         if name in row and pd.notnull(row[name]) and str(row[name]).strip()),
                        cohort,
                    )
                    try:
                        if neuron_id in sagittal_failures:
                            raise FileNotFoundError(sagittal_failures[neuron_id])
                        neuron_provenance = process_neuron(
                            toolkit, ion, sample_id, neuron_id, region, dirs,
                            export_context_volume=False,
                            context_sagittal=context_sagittal,
                            cached_only=cached_only,
                            sagittal_source=sagittal_source,
                            preloaded_tree=sagittal_trees.get(neuron_id),
                        )
                        cohort_record["neurons"].append(neuron_provenance)
                        cohort_record["ok"] += 1
                        if neuron_provenance.get("panel_errors"):
                            cohort_record["partial"] += 1
                            cohort_record["partial_neurons"].append(
                                f"{neuron_id}: " + "; ".join(
                                    f"{name}: {reason}" for name, reason in
                                    neuron_provenance["panel_errors"].items()))
                    except Exception as exc:
                        cohort_record["fail"] += 1
                        failures.append(f"{neuron_id}: {exc}")
                        print(f"[ERR] {sample_id}/{cohort}/{neuron_id}: {exc}")
            finally:
                try:
                    toolkit.close()
                except Exception:
                    pass
            cohort_record["failed_neurons"] = failures
            item["cohorts"].append(cohort_record)
        records.append(item)
    manifest = None
    if not dry_run:
        for item in records:
            append_soma_crop_qc(item["source_readme"], [item])
        manifest = write_refined_run_manifest(
            run_id, run_stamp, meta, records,
            output_root=output_root,
        )
    return {"samples": records, "manifest": manifest, "dry_run": dry_run}


def main():
    parser = argparse.ArgumentParser(description="Legacy bulk visual runner or selected-cohort export")
    parser.add_argument("--selected-cohorts", action="store_true",
                        help="Use active selected workbook and Atlas_INS source split")
    parser.add_argument("--samples", help="Comma-separated selected sample IDs; default is all active samples")
    parser.add_argument("--cohorts", default="Atlas_neurons,Candidate_neurons",
                        help="Comma-separated refined cohorts")
    parser.add_argument("--run-stamp", help="Production-date folder stamp YYYYMMDD")
    parser.add_argument("--output-root", help="Override server root for bounded pilots")
    parser.add_argument("--dry-run", action="store_true", help="Report exact source counts and output paths only")
    parser.add_argument("--context-sagittal", action="store_true",
                        help="Also render the processed-DS sagittal panel plus the one-figure "
                             "(Soma|Context|Sagittal) review plot into Context/ or TriPlots/")
    parser.add_argument("--cached-only", action="store_true",
                        help="Never request missing Soma blocks over HTTP; report unavailable cache data as failures")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="Override one REVIEW_SETTINGS leaf for this run, e.g. "
                             "--set widefield.gain=1.6 --set marker.crosshair_px=80 "
                             "(repeatable; recorded in the manifest)")
    args = parser.parse_args()
    if args.selected_cohorts:
        try:
            applied = apply_setting_overrides(args.set)
        except (KeyError, ValueError) as exc:
            parser.error(exc.args[0] if exc.args else str(exc))
        if applied:
            print(f"[settings] overrides applied: {applied}")
        samples = [value.strip() for value in args.samples.split(",") if value.strip()] if args.samples else None
        cohorts = [value.strip() for value in args.cohorts.split(",") if value.strip()]
        invalid = sorted(set(cohorts).difference({"Atlas_neurons", "Candidate_neurons"}))
        if invalid or not cohorts:
            raise ValueError(f"Invalid refined cohorts: {invalid or cohorts}")
        result = process_selected_cohorts(
            samples, cohorts, args.run_stamp or DATESTAMP, args.output_root,
            args.dry_run, context_sagittal=args.context_sagittal,
            cached_only=args.cached_only,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

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
