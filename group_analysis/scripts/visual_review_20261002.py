"""Correction-ready native fMOST panels for the copied 2026-10-02 review set.

Selection is the union of atlas insula, PrCO, explicit unknown/unmapped labels,
staging keepers (including gustatory G), adjacent opercular/claustrum labels,
and somata inside the folded 251637 99% insula bboxes. A blank portal region is
not treated as atlas-Unknown. Portal soma coordinates stay in NMT space; image
crops use the raw SWC root in the declared native micrometre frame.

The five samples without a copied 5 µm series are written to the queue and are
not given another animal's images. Human correction columns start blank and are
kept when a reviewer fills either the CSV or the XLSX.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"D:\projectome_analysis")
MAIN_SCRIPTS = PROJECT_ROOT / "main_scripts"
GROUP_SCRIPTS = PROJECT_ROOT / "group_analysis" / "scripts"
for _path in (MAIN_SCRIPTS, GROUP_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cohort import NII_VOXEL_MM, fold_nii_x  # noqa: E402
from fmost_image_geometry import clip_segment, native_affine, native_to_index, percentile_normalize  # noqa: E402
from insula_label_set import build_insula_label_set, normalize_label  # noqa: E402
from swc_validation import parse_swc  # noqa: E402
from fmost_review_lock import exclusive_batch  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "group_analysis" / "visual_review_20261002"
PORTAL_DIR = PROJECT_ROOT / "notes" / "bulk_visual_review_20261002" / "portal_snapshot"
STEP1_DIR = PROJECT_ROOT / "group_analysis" / "step1_results"
STAGING_DIR = PROJECT_ROOT / "group_analysis" / "staging_20260926"
BBOX_CSV = STAGING_DIR / "reference" / "251637_subregion_bboxes_folded.csv"
REFINED_CSV = STAGING_DIR / "recovery" / "all_refined_neurons.csv"
STAGING_XLSX = STAGING_DIR / "combined" / "multi_monkey_INS_combined_harmonized.xlsx"

# Copied CH1 overview series. 251637 is the reference dataset, not a new sample.
AVAILABLE_SAMPLES = ("252384", "252385", "252527", "252714")
PENDING_SAMPLES = ("250432", "251730", "252383", "252718", "252790")
NEW_SAMPLES = AVAILABLE_SAMPLES + PENDING_SAMPLES

# Inventory adjacent screen: retroinsula, claustrum, ProM, opercular neighbors.
# G is kept as its own gustatory category and is not remapped to IG.
ADJACENT_LABELS = {
    "RI", "RETROINSULA", "CL", "CLAUSTRUM", "PROM",
    "SII", "TPT", "AREA_7OP", "AREA_44", "F4", "F5",
}
EXPLICIT_UNKNOWN = {"UNKNOWN", "UNMAPPED", "_UNMAPPED", "INSULAUNKNOWN"}
CATEGORY_RANK = {
    "INS": 0, "G": 1, "possible_insula": 2, "PrCO": 3,
    "adjacent": 4, "Unknown": 5, "metadata_unresolved": 6, "other": 7,
}
CONTEXT_UM = (4000.0, 4000.0, 30.0)  # XYZ; 4 mm field, 30 µm declared depth
TRACE_SLAB_SECTIONS = 5  # odd window centered on the displayed section
SOMA_PACKAGE_FILES = (
    "soma_unmarked.png", "soma_marked.png", "soma_trace.png",
    "soma_ortho.png", "soma_stack.png", "soma.nii.gz",
)
CONTEXT_PACKAGE_FILES = (
    "context_unmarked.png", "context_marked.png", "context_trace.png",
    "context_stack.png", "section_locator.png", "context.nii.gz",
)
DERIVED_CONTEXT_STATES = {"derived_highres", "derived_partial"}
HIGH_GRID_RADIUS = 1
SWC_ATTEMPTS = 3
NII_UM = NII_VOXEL_MM * 1000.0
CANONICAL_FILES = (
    PROJECT_ROOT / "group_analysis" / "combined" / "multi_monkey_INS_combined_harmonized.xlsx",
    PROJECT_ROOT / "group_analysis" / "combined" / "multi_monkey_INS_combined.xlsx",
    PROJECT_ROOT / "R_analysis" / "tables" / "somainfo_Henry_2026.04.03.xlsx",
    PROJECT_ROOT / "neuron_tables" / "251637_INS_HE.xlsx",
)
PANEL_VERSION = 3
PANEL_FILES = (
    "context_unmarked.png", "context_marked.png", "context_trace.png",
    "context_stack.png", "section_locator.png",
    "soma_unmarked.png", "soma_marked.png", "soma_trace.png",
    "soma_ortho.png", "soma_stack.png",
    "context.nii.gz", "soma.nii.gz",
)
OPTIONAL_PANEL_FILES = ("context_coverage.png", "context_coverage.nii.gz", "context_ortho.png")
HUMAN_FIELDS = (
    "human_region", "human_subregion", "human_layer",
    "reviewer", "review_date", "human_notes",
)
# Folded 99% boxes are the screen. Eight voxels is 2 mm at 0.25 mm/voxel and
# is an audit only; it does not add identities.
COORDINATE_SCREEN_PAD_VOX = 0.0
COORDINATE_PAD_AUDIT_VOX = 8.0
CONTEXT_RESUME_FILES = (
    "context_unmarked.png", "context_marked.png", "context_trace.png",
    "context_stack.png", "section_locator.png", "context.nii.gz",
)
NEIGHBOR_GAP_UIDS = (
    "252385|086.swc", "252385|101.swc", "252527|087.swc",
    "252384|079.swc", "252714|071.swc",
)


def display_label(value) -> str:
    """Blank portal cells are absent metadata, not the atlas word Unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "absent"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return "absent"
    return text


def canon_neuron_id(value) -> str:
    """Normalize a neuron id to the portal ``NNN.swc`` spelling when numeric."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    if text.lower().endswith(".swc"):
        stem, suffix = text[:-4], ".swc"
    else:
        stem, suffix = text, ".swc"
    if stem.isdigit():
        return f"{int(stem):03d}{suffix}"
    if not stem or any(ch in stem for ch in "\\/:"):
        raise ValueError(f"Unsafe neuron id: {value!r}")
    return stem + suffix


def portal_base_label(region) -> str:
    """Strip a trailing atlas index. Blank stays blank and is not Unknown."""
    if region is None or (isinstance(region, float) and np.isnan(region)):
        return ""
    text = str(region).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    head, sep, tail = text.rpartition("_")
    if sep and tail.isdigit():
        text = head
    return normalize_label(text)


def historical_base_label(region) -> str:
    """Normalize a step1 or staging label without removing its anatomical number.

    ``CL_area_44`` and ``area_44`` stay AREA_44. A trailing atlas index is a
    portal spelling (``area_44_76``) and is stripped only by ``portal_base_label``.
    """
    if region is None or (isinstance(region, float) and np.isnan(region)):
        return ""
    text = str(region).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return normalize_label(text)


def is_adjacent(base: str) -> bool:
    """`base` is already the normalized uppercase label. Do not strip digits again.

    Area names such as AREA_44 end in their own number.
    """
    if not base:
        return False
    return base in ADJACENT_LABELS or "OPERCUL" in base or "CLAUSTR" in base


def label_reasons(prefix: str, raw, insula_labels: set[str], portal: bool = False) -> list[str]:
    """Map one source label onto review reasons. Blank adds nothing."""
    base = portal_base_label(raw) if portal else historical_base_label(raw)
    if not base:
        return []
    if base in insula_labels:
        return [f"{prefix}_insula"]
    if base == "PRCO":
        return [f"{prefix}_prco"]
    if base == "G":
        return [f"{prefix}_G"]
    if base in EXPLICIT_UNKNOWN or "UNKNOWN" in base:
        return [f"{prefix}_explicit_unknown"]
    if is_adjacent(base):
        return [f"{prefix}_adjacent"]
    return []


def step1_reasons(raw, insula_labels: set[str]) -> list[str]:
    """Historical step1 Unknown includes an empty atlas cell; portal blank does not."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ["step1_unknown"]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"} or "unknown" in text.lower():
        return ["step1_unknown"]
    return label_reasons("step1", text, insula_labels, portal=False)


def review_category(reasons: list[str]) -> str:
    """Highest-priority bin. Every reason is still kept on the row."""
    found = set(reasons)
    if found & {"step1_insula", "portal_insula", "staging_insula"}:
        return "INS"
    if found & {"step1_G", "portal_G", "staging_G"}:
        return "G"
    if "coordinate_possible_insula" in found:
        return "possible_insula"
    if found & {"step1_prco", "portal_prco", "staging_prco"}:
        return "PrCO"
    if found & {"step1_adjacent", "portal_adjacent", "staging_adjacent"}:
        return "adjacent"
    if found & {"step1_unknown", "portal_explicit_unknown", "staging_explicit_unknown"}:
        return "Unknown"
    if "metadata_unresolved_raw_source" in found:
        return "metadata_unresolved"
    return "other"


def exact_project_id(records, sample: str) -> str:
    """Use the record whose fMOST_id matches exactly, never an arbitrary first row."""
    matches = [row for row in records if str(row.get("fMOST_id")) == str(sample)]
    if len(matches) != 1:
        raise ValueError(f"{sample}: expected one exact fMOST_id, found {len(matches)}")
    project = matches[0].get("project_id")
    if not project:
        raise ValueError(f"{sample}: exact sample info has no project_id")
    return str(project)


def image_source_status(sample: str) -> str:
    if sample in AVAILABLE_SAMPLES:
        return "available"
    if sample in PENDING_SAMPLES:
        return "source_pending"
    raise ValueError(f"Sample {sample} is outside the nine-sample review list")


def folded_bbox_hits(nii_x, nii_y, nii_z, boxes, pad_vox: float = 0.0) -> list[str]:
    """Return 251637 subregion names whose folded 99% box contains this NMT point.

    ``pad_vox`` expands the box for the documented 2 mm audit. Selection calls
    this with pad 0.
    """
    if not np.isfinite([nii_x, nii_y, nii_z]).all():
        return []
    folded = fold_nii_x(nii_x)
    pad = float(pad_vox)
    hits = []
    for name, x0, x1, y0, y1, z0, z1 in boxes:
        if (x0 - pad) <= folded <= (x1 + pad) and (y0 - pad) <= nii_y <= (y1 + pad) and (z0 - pad) <= nii_z <= (z1 + pad):
            hits.append(str(name))
    return hits


def classify_identity(portal_region, step1_region, staging_region, in_staging: bool,
                      nii_xyz, boxes, insula_labels: set[str],
                      in_portal: bool, in_step1: bool) -> list[str]:
    """Return selection reasons, or an empty list when the identity is out of scope."""
    reasons = []
    reasons.extend(label_reasons("portal", portal_region, insula_labels, portal=True))
    if in_step1:
        reasons.extend(step1_reasons(step1_region, insula_labels))
    if in_staging:
        reasons.append("staging_keeper")
        reasons.extend(label_reasons("staging", staging_region, insula_labels, portal=False))
    if nii_xyz is not None:
        hits = folded_bbox_hits(*nii_xyz, boxes)
        if hits:
            reasons.append("coordinate_possible_insula")
    if not reasons:
        return []
    if in_portal and not in_step1:
        reasons.append("missing_from_step1")
    if in_step1 and not in_portal:
        reasons.append("missing_from_portal")
    return reasons


def _finite_xyz(values) -> list[float] | None:
    try:
        point = [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(point).all():
        return None
    return point


def _load_boxes(path: Path = BBOX_CSV):
    frame = pd.read_csv(path)
    boxes = []
    for row in frame.itertuples(index=False):
        boxes.append((
            row.sub_region, float(row.X_lo_q005), float(row.X_hi_q995),
            float(row.Y_lo_q005), float(row.Y_hi_q995),
            float(row.Z_lo_q005), float(row.Z_hi_q995),
        ))
    return boxes


def _index_table(frame: pd.DataFrame, sample_col: str, neuron_col: str) -> dict:
    indexed = {}
    if frame is None or frame.empty:
        return indexed
    for offset, row in enumerate(frame.to_dict(orient="records")):
        sample = str(row.get(sample_col, "")).strip()
        neuron = canon_neuron_id(row.get(neuron_col))
        if sample in NEW_SAMPLES and neuron:
            row["_source_row"] = offset + 2
            indexed.setdefault((sample, neuron), row)
    return indexed


def _field_text(row, name: str) -> str:
    if not row:
        return ""
    value = row.get(name)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def staging_origin(row, workbook: str) -> dict:
    """Keep the workbook's own label-origin fields. Do not collapse them into one token."""
    if not row:
        return {"region_source": "", "neuron_uid": "", "row_uid": ""}
    sample = _field_text(row, "SampleID")
    neuron = canon_neuron_id(row.get("NeuronID"))
    source_row = row.get("_source_row", "")
    neuron_uid = _field_text(row, "NeuronUID")
    return {
        "region_source": _field_text(row, "Soma_Region_Source"),
        "neuron_uid": neuron_uid,
        "row_uid": f"{workbook}|{sample}|{neuron}|row{source_row}",
    }


def _latest_step1(sample: str) -> tuple[str, pd.DataFrame]:
    matches = sorted(STEP1_DIR.glob(
        f"{sample}_*_region_analysis/tables/{sample}_results_*.xlsx"))
    if not matches:
        return "", pd.DataFrame()
    path = matches[-1]
    return str(path), pd.read_excel(path, sheet_name="Summary")


def _pick_nii(step1_row, staging_row, portal_soma) -> tuple[list[float] | None, str, list[float] | None]:
    """Prefer analyzed NII voxels. Portal soma is NMT micrometres, never native."""
    candidates = []
    if step1_row:
        point = _finite_xyz([
            step1_row.get("Soma_NII_X"), step1_row.get("Soma_NII_Y"), step1_row.get("Soma_NII_Z")])
        if point:
            candidates.append((point, "step1_Soma_NII"))
    if staging_row:
        point = _finite_xyz([
            staging_row.get("Soma_NII_X"), staging_row.get("Soma_NII_Y"), staging_row.get("Soma_NII_Z")])
        if point:
            candidates.append((point, "staging_Soma_NII"))
    portal_um = None
    if portal_soma:
        portal_um = _finite_xyz([
            portal_soma.get("somax"), portal_soma.get("somay"), portal_soma.get("somaz")])
        if portal_um and not candidates:
            candidates.append(([value / NII_UM for value in portal_um], "portal_soma_um_div_250"))
    if not candidates:
        return None, "", portal_um
    chosen, source = candidates[0]
    return chosen, source, portal_um


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hashes(files=CANONICAL_FILES) -> dict:
    found = {}
    for path in files:
        path = Path(path)
        if path.is_file():
            found[str(path)] = sha256_file(path)
    return found


def _canonical_map(payload) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("Canonical hash baseline is not a map")
    nested = payload.get("canonical")
    if isinstance(nested, dict):
        return nested
    return payload


def guard_canonical_baseline(out_root: Path = OUT_ROOT, files=CANONICAL_FILES) -> dict:
    """Keep the first hash baseline. A mismatch stops before outputs are replaced."""
    current = canonical_hashes(files)
    if not current:
        raise RuntimeError("Canonical baseline inputs are missing")
    path = out_root / "manifest" / "canonical_hashes.json"
    if path.is_file():
        saved = _canonical_map(json.loads(path.read_text(encoding="utf-8")))
        if saved != current:
            changed = [key for key in sorted(set(saved) | set(current)) if saved.get(key) != current.get(key)]
            raise RuntimeError(
                "Canonical baseline mismatch; the stored baseline was not replaced: " + "; ".join(changed))
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def selection_config() -> dict:
    audit_mm = float(COORDINATE_PAD_AUDIT_VOX) * float(NII_VOXEL_MM)
    return {
        "coordinate_screen_pad_vox": COORDINATE_SCREEN_PAD_VOX,
        "coordinate_pad_audit_vox": COORDINATE_PAD_AUDIT_VOX,
        "coordinate_pad_audit_mm": audit_mm,
        "adjacent_labels": sorted(ADJACENT_LABELS),
        "explicit_unknown": sorted(EXPLICIT_UNKNOWN),
        "context_um_xyz": list(CONTEXT_UM),
        "spacing_low_xyz_um": [5.0, 5.0, 3.0],
        "spacing_high_xyz_um": [0.65, 0.65, 3.0],
        "high_grid_radius": HIGH_GRID_RADIUS,
        "available_samples": list(AVAILABLE_SAMPLES),
        "pending_samples": list(PENDING_SAMPLES),
        "panel_version": PANEL_VERSION,
    }


def selection_input_record(step1_paths: dict) -> dict:
    """Hash the tables and portal snapshot that decide membership, plus the screen config."""
    files = {
        "bbox_csv": BBOX_CSV,
        "refined_csv": REFINED_CSV,
        "staging_xlsx": STAGING_XLSX,
    }
    manifest = PORTAL_DIR / "manifest.json"
    if manifest.is_file():
        files["portal_manifest"] = manifest
    for sample in NEW_SAMPLES:
        for kind in ("neurons", "soma", "sample_info"):
            path = PORTAL_DIR / f"{sample}_{kind}.json"
            if path.is_file():
                files[f"portal_{sample}_{kind}"] = path
    hashed = {name: sha256_file(Path(path)) for name, path in files.items() if Path(path).is_file()}
    for sample, path in sorted(step1_paths.items()):
        hashed[f"step1_{sample}"] = sha256_file(Path(path))
    config = selection_config()
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files_sha256": hashed,
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinate_screen": (
            "Unpadded folded 99% boxes. "
            f"An audit pad of {COORDINATE_PAD_AUDIT_VOX:g} voxels "
            f"({float(COORDINATE_PAD_AUDIT_VOX) * float(NII_VOXEL_MM):g} mm) is recorded and not applied."
        ),
    }


def validated_unresolved_sources(report_path: Path | None) -> dict:
    """Bind optional review candidates to fully validated, own-sample raw files.

    An accessible source is a review reason, not an anatomical label. This
    extension is opt-in so an existing producer's input is never silently changed.
    """
    if report_path is None:
        return {}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("state") != "terminal_complete":
        raise ValueError("Unresolved raw-source validation is not terminal")
    result = {}
    for record in report["records"]:
        if record.get("status") != "structurally_valid" or record.get("graph_validated") is not True:
            continue
        sample, neuron = record["uid"].split("|", 1)
        if sample not in NEW_SAMPLES or neuron != canon_neuron_id(neuron):
            raise ValueError("Unresolved source identity is outside this review")
        expected_url = f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}"
        if record["url"] != expected_url or record.get("metadata_status") != "unresolved":
            raise ValueError("Unresolved source route or metadata status differs")
        path = (PROJECT_ROOT / record["local_path"]).resolve()
        if PROJECT_ROOT.resolve() not in path.parents or path.parent.name != sample or path.name != neuron:
            raise ValueError("Unresolved source path is outside its own-sample workspace")
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Unresolved raw-source bytes changed after validation")
        rows = parse_swc(path.read_text(encoding="utf-8-sig"), source=str(path))
        if len(rows) != record["nodes"] or _root_xyz(rows) != record["raw_root_xyz"]:
            raise ValueError("Unresolved raw graph differs from validation record")
        if record["uid"] in result:
            raise ValueError("Duplicate unresolved source UID")
        result[record["uid"]] = record
    return result


def build_manifest(out_root: Path = OUT_ROOT, unresolved_report: Path | None = None) -> pd.DataFrame:
    """Replace a manifest only when regional production has released ownership."""
    out_root = Path(out_root)
    with exclusive_batch(out_root / "manifest" / "regional_context.lock"):
        return _build_manifest(out_root, unresolved_report)


def retained_unresolved_report(out_root: Path, explicit_report: Path | None) -> Path | None:
    """Retain a previously activated extension on ordinary manifest regeneration."""
    if explicit_report is not None:
        return Path(explicit_report)
    saved_path = Path(out_root) / "manifest" / "selection_inputs.json"
    if not saved_path.is_file():
        return None
    extension = json.loads(saved_path.read_text(encoding="utf-8")).get("validated_unresolved_sources")
    if extension is None:
        return None
    path = Path(extension["report_path"])
    if not path.is_file() or sha256_file(path) != extension["report_sha256"]:
        raise ValueError("Activated unresolved source report changed or is missing; manifest not regenerated")
    return path


def _build_manifest(out_root: Path, unresolved_report: Path | None = None) -> pd.DataFrame:
    """Reconcile portal, latest step1 and staging into one identity table."""
    unresolved_report = retained_unresolved_report(out_root, unresolved_report)
    guard_canonical_baseline(out_root)
    unresolved = validated_unresolved_sources(unresolved_report)
    insula_labels, _rescue = build_insula_label_set()
    boxes = _load_boxes()
    refined = _index_table(pd.read_csv(REFINED_CSV), "SampleID", "NeuronID")
    staging_book = pd.read_excel(STAGING_XLSX, sheet_name="Summary")
    staging = _index_table(staging_book, "SampleID", "NeuronID")
    from Visual_toolkit import LOW_RES_SHARE_BY_SAMPLE

    rows = []
    excluded_labels = []
    metadata_queue = []
    pad_only_uids = []
    step1_paths = {}
    for sample in NEW_SAMPLES:
        portal_neurons = json.loads((PORTAL_DIR / f"{sample}_neurons.json").read_text(encoding="utf-8"))
        portal_soma = json.loads((PORTAL_DIR / f"{sample}_soma.json").read_text(encoding="utf-8"))
        info = json.loads((PORTAL_DIR / f"{sample}_sample_info.json").read_text(encoding="utf-8"))
        project_id = exact_project_id(info, sample)
        by_neuron = {}
        for item in portal_neurons:
            neuron = canon_neuron_id(item.get("name"))
            by_neuron.setdefault(neuron, item)
        by_soma = {}
        for item in portal_soma:
            by_soma.setdefault(canon_neuron_id(item.get("name")), item)
        step1_path, step1_frame = _latest_step1(sample)
        if step1_path:
            step1_paths[sample] = step1_path
        step1 = _index_table(step1_frame, "SampleID", "NeuronID") if "SampleID" in step1_frame.columns else {}
        if step1_frame is not None and not step1_frame.empty and not step1:
            # Older summaries are single-sample files and omit SampleID.
            tagged = step1_frame.copy()
            tagged["SampleID"] = sample
            step1 = _index_table(tagged, "SampleID", "NeuronID")
        step1_ids = {neuron for (sid, neuron) in step1 if sid == sample}
        staging_ids = {neuron for (sid, neuron) in set(refined) | set(staging) if sid == sample}
        keys = set(by_neuron) | step1_ids | staging_ids
        low_dir = LOW_RES_SHARE_BY_SAMPLE.get(sample, "")
        for neuron in sorted(keys):
            portal = by_neuron.get(neuron)
            step = step1.get((sample, neuron))
            harmonized_row = staging.get((sample, neuron))
            refined_row = refined.get((sample, neuron))
            stage = harmonized_row or refined_row
            stage_sources = []
            if harmonized_row:
                stage_sources.append("staging_harmonized")
            if refined_row:
                stage_sources.append("all_refined")
            harmonized_origin = staging_origin(harmonized_row, str(STAGING_XLSX))
            refined_origin = staging_origin(refined_row, str(REFINED_CSV))
            origin_values = [item for item in (harmonized_origin["region_source"], refined_origin["region_source"]) if item]
            refined_label = ""
            if stage:
                refined_label = stage.get("Soma_Region_Refined") or stage.get("Soma_Region") or ""
            nii, nii_source, portal_um = _pick_nii(step, stage, by_soma.get(neuron))
            portal_region = "" if portal is None else portal.get("region", "")
            step_region = "" if step is None else step.get("Soma_Region", "")
            reasons = classify_identity(
                portal_region, step_region, refined_label, bool(stage_sources),
                nii, boxes, insula_labels, portal is not None, step is not None)
            audit_hits = []
            if nii is not None:
                audit_hits = folded_bbox_hits(*nii, boxes, pad_vox=COORDINATE_PAD_AUDIT_VOX)
            if not reasons and audit_hits:
                pad_only_uids.append(f"{sample}|{neuron}")
            if not reasons:
                portal_blank = portal is None or portal_base_label(portal_region) == ""
                if portal_blank:
                    metadata_queue.append({
                        "SampleID": sample,
                        "NeuronID": neuron,
                        "UID": f"{sample}|{neuron}",
                        "portal_region": "" if portal is None or portal_region is None else str(portal_region),
                        "step1_soma_region": "" if step is None else _field_text(step, "Soma_Region"),
                        "reason": (
                            "blank portal metadata with no insula, PrCO, unknown, adjacent, staging, "
                            "or coordinate reason; not atlas-Unknown and not accepted as outside-insula"
                        ),
                    })
                if portal_blank and f"{sample}|{neuron}" in unresolved:
                    reasons = ["metadata_unresolved_raw_source"]
                else:
                    excluded_labels.append({
                        "sample": sample, "portal_base": portal_base_label(portal_region) or "<blank>"})
                    continue
            category = review_category(reasons)
            disagreement = False
            if portal_um and nii is not None and nii_source != "portal_soma_um_div_250":
                portal_nii = [value / NII_UM for value in portal_um]
                disagreement = float(np.max(np.abs(np.subtract(portal_nii, nii)))) > 4.0
            rows.append({
                "SampleID": sample,
                "NeuronID": neuron,
                "UID": f"{sample}|{neuron}",
                "review_category": category,
                "category_rank": CATEGORY_RANK[category],
                "selection_reasons": "|".join(reasons),
                "portal_region": "" if portal_region is None else str(portal_region),
                "portal_region_base": portal_base_label(portal_region),
                "portal_region_absent": portal is None or portal_base_label(portal_region) == "",
                "in_portal": portal is not None,
                "step1_soma_region": "" if step_region is None or (isinstance(step_region, float) and np.isnan(step_region)) else str(step_region),
                "in_step1": step is not None,
                "step1_source": step1_path,
                "staging_refined_label": "" if refined_label is None else str(refined_label),
                "staging_sources": "|".join(stage_sources),
                "staging_region_source": "|".join(dict.fromkeys(origin_values)),
                "staging_harmonized_region_source": harmonized_origin["region_source"],
                "staging_harmonized_neuron_uid": harmonized_origin["neuron_uid"],
                "staging_harmonized_row_uid": harmonized_origin["row_uid"],
                "refined_region_source": refined_origin["region_source"],
                "refined_neuron_uid": refined_origin["neuron_uid"],
                "refined_row_uid": refined_origin["row_uid"],
                "in_staging": bool(stage_sources),
                "overview_source_status": "copied_ch1" if sample in AVAILABLE_SAMPLES else "overview_pending",
                "nmt_nii_x": None if nii is None else nii[0],
                "nmt_nii_y": None if nii is None else nii[1],
                "nmt_nii_z": None if nii is None else nii[2],
                "nmt_frame": nii_source,
                "nmt_portal_soma_x_um": None if portal_um is None else portal_um[0],
                "nmt_portal_soma_y_um": None if portal_um is None else portal_um[1],
                "nmt_portal_soma_z_um": None if portal_um is None else portal_um[2],
                "nmt_coordinate_disagreement": disagreement,
                "bbox_hits": "|".join(folded_bbox_hits(*(nii or (np.nan, np.nan, np.nan)), boxes)),
                "project_id": project_id,
                "image_source_status": image_source_status(sample),
                "low_res_dir": low_dir,
                "channel": "CH1",
                "channel_note": "CH1 fluorescence overview; not verified as PI or cytoarchitecture",
                "spacing_status": "nominal repository configuration; not independently calibrated",
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Identity manifest is empty")
    insula_counts = (
        frame[frame["review_category"] == "INS"].groupby("SampleID").size().to_dict())
    frame["sample_insula_count"] = frame["SampleID"].map(lambda sample: int(insula_counts.get(sample, 0)))
    frame = frame.sort_values(
        ["image_source_status", "sample_insula_count", "SampleID", "category_rank", "NeuronID"],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    manifest_dir = out_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_dir / "identity_manifest.csv", index=False)
    pending = frame[frame["image_source_status"] == "source_pending"].copy()
    pending.to_csv(manifest_dir / "source_pending_queue.csv", index=False)
    counts = {
        "rows": int(len(frame)),
        "available_rows": int((frame["image_source_status"] == "available").sum()),
        "pending_rows": int(len(pending)),
        "by_sample_category": (
            frame.groupby(["SampleID", "image_source_status", "review_category"]).size()
            .reset_index(name="n").to_dict(orient="records")),
        "excluded_portal_base_counts": (
            pd.DataFrame(excluded_labels).groupby(["sample", "portal_base"]).size()
            .reset_index(name="n").sort_values("n", ascending=False).head(40)
            .to_dict(orient="records") if excluded_labels else []),
        "coordinate_screen_pad_vox": COORDINATE_SCREEN_PAD_VOX,
        "coordinate_pad_audit_vox": COORDINATE_PAD_AUDIT_VOX,
        "coordinate_pad_audit_mm": float(COORDINATE_PAD_AUDIT_VOX) * float(NII_VOXEL_MM),
        "pad_audit_extra_identities": len(pad_only_uids),
        "pad_audit_extra_uids": pad_only_uids,
        "metadata_resolution_rows": len(metadata_queue),
    }
    pd.DataFrame(metadata_queue).to_csv(manifest_dir / "metadata_resolution_queue.csv", index=False)
    (manifest_dir / "selection_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    selection_inputs = selection_input_record(step1_paths)
    if unresolved_report is not None:
        selection_inputs["validated_unresolved_sources"] = {
            "report_path": str(Path(unresolved_report).resolve()), "report_sha256": sha256_file(Path(unresolved_report)),
            "sources_sha256": {uid: record["sha256"] for uid, record in unresolved.items()},
            "policy": "include blank-metadata validated raw sources for review; no atlas or cohort promotion",
        }
    (manifest_dir / "selection_inputs.json").write_text(
        json.dumps(selection_inputs, indent=2), encoding="utf-8")
    print(f"[manifest] {len(frame)} identities, {len(pending)} source-pending -> {manifest_dir}")
    print(f"[manifest] 2 mm pad audit extra identities: {len(pad_only_uids)}")
    return frame


def load_manifest(out_root: Path = OUT_ROOT) -> pd.DataFrame:
    path = out_root / "manifest" / "identity_manifest.csv"
    if not path.is_file():
        return build_manifest(out_root)
    frame = pd.read_csv(path, dtype={"SampleID": str, "NeuronID": str, "project_id": str})
    frame["NeuronID"] = frame["NeuronID"].map(canon_neuron_id)
    return frame


def _root_xyz(rows) -> list[float]:
    roots = [row for row in rows if row[6] == -1]
    if len(roots) != 1:
        raise ValueError(f"Expected one SWC root, found {len(roots)}")
    return [float(roots[0][2]), float(roots[0][3]), float(roots[0][4])]


def documented_swc_routes(sample: str, neuron: str, project_id: str, out_root: Path):
    """Raw native routes only. The processed getSWC cache is listed so it can be rejected."""
    url = f"http://10.10.31.31/swc/newswc/{project_id}/{sample}/swc_raw/{neuron}"
    raw_caches = [
        out_root / "swc_raw" / sample / neuron,
        PROJECT_ROOT / "resource" / "swc_raw" / sample / neuron,
        PROJECT_ROOT / "neuron-vis" / "resource" / "swc_raw" / sample / neuron,
    ]
    processed = PROJECT_ROOT / "neuron-vis" / "resource" / "swc" / sample / neuron
    return url, raw_caches, processed


def _read_raw_swc(path: Path, sample: str):
    """Accept a file only when it sits in that sample's own swc_raw directory."""
    if path.parent.name != sample or path.parent.parent.name != "swc_raw":
        raise ValueError(f"Refusing SWC outside the own-sample raw cache: {path}")
    if not path.is_file() or path.stat().st_size <= 50:
        return None
    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    rows = parse_swc(text, source=str(path))
    return rows, hashlib.sha256(payload).hexdigest(), payload


def fetch_raw_swc(sample: str, neuron: str, project_id: str, out_root: Path = OUT_ROOT,
                  attempts: int = SWC_ATTEMPTS, timeout: float = 60.0):
    """Bounded raw-SWC fetch. The file is kept only after parse_swc accepts it.

    NMT portal coordinates are not a fallback. The processed neuronbrowser SWC
    is a different product and is not used for native crops.
    """
    dest = out_root / "swc_raw" / sample / neuron
    url, raw_caches, processed = documented_swc_routes(sample, neuron, project_id, out_root)
    checked = []
    for path in raw_caches:
        checked.append(str(path))
        try:
            loaded = _read_raw_swc(path, sample)
        except ValueError:
            if path == dest and path.is_file():
                path.unlink()
            continue
        if loaded is None:
            continue
        rows, digest, payload = loaded
        if path != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        return rows, digest, url, f"own-sample-cache:{path}"
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "projectome-visual-review"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            text = payload.decode("utf-8", errors="replace")
            rows = parse_swc(text, source=url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return rows, hashlib.sha256(payload).hexdigest(), url, "http"
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    processed_note = "present but not used" if processed.is_file() else "absent"
    raise RuntimeError(
        f"raw SWC unavailable for {sample}/{neuron} after {attempts} requests to {url}; "
        f"own-sample raw caches checked: {checked}; "
        f"processed SWC cache {processed} is {processed_note}; "
        f"NMT coordinates were not used. {last_error}"
    )


def _display(plane: np.ndarray) -> np.ndarray:
    shown = percentile_normalize(np.asarray(plane), 0.5, 99.5)
    return np.power(shown, 0.5)


def _save_gray(plane: np.ndarray, path: Path) -> dict:
    shown = _display(plane)
    image = Image.fromarray((np.clip(shown, 0, 1) * 255).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "min": int(np.min(plane)), "max": int(np.max(plane)),
        "nonzero_fraction": float(np.count_nonzero(plane) / plane.size),
        "display": "percentile 0.5-99.5 then gamma 0.5; raw values are in the NIfTI",
    }


def _draw_marked(plane: np.ndarray, path: Path, marker_xy, spacing_xy, title: str,
                 segments=None, scale_um: float = 500.0) -> None:
    shown = _display(plane)
    height, width = shown.shape
    figure_inches = (max(4.0, width / 140.0), max(4.0, height / 140.0))
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=figure_inches, dpi=110, facecolor="black")
    ax.imshow(shown, cmap="gray", vmin=0, vmax=1, origin="upper")
    if segments:
        ax.add_collection(LineCollection(segments, colors="red", linewidths=0.6, alpha=0.8))
    if marker_xy is not None:
        ax.scatter([marker_xy[0]], [marker_xy[1]], s=40, marker="s",
                   facecolors="none", edgecolors="cyan", linewidths=0.8)
    bar = scale_um / float(spacing_xy[0])
    x0, y0 = 24, height - 24
    ax.plot([x0, x0 + bar], [y0, y0], color="white", linewidth=2)
    ax.text(x0, y0 - 12, f"{scale_um:.0f} um", color="white", fontsize=8, va="bottom")
    ax.set_title(title, color="white", fontsize=9)
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def slab_index_bounds(local_z, depth: int, sections: int = TRACE_SLAB_SECTIONS) -> tuple[int, int]:
    """Inclusive-exclusive Z window of ``sections`` planes around the displayed index."""
    center = int(np.floor(float(local_z)))
    half = int(sections) // 2
    z0 = max(0, center - half)
    z1 = min(int(depth), z0 + int(sections))
    z0 = max(0, z1 - int(sections))
    return z0, z1


def _segments_in_view(rows, origin, spacing, shape_zyx, z_bounds=None) -> list:
    by_id = {row[0]: row for row in rows}
    depth, height, width = shape_zyx
    if z_bounds is None:
        lower_z, upper_z = -0.5, depth - 0.5
    else:
        lower_z, upper_z = float(z_bounds[0]) - 0.5, float(z_bounds[1]) - 0.5
    lower = [-0.5, -0.5, lower_z]
    upper = [width - 0.5, height - 0.5, upper_z]
    segments = []
    for node in rows:
        parent_id = node[6]
        if parent_id == -1 or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        start = native_to_index([parent[2], parent[3], parent[4]], origin, spacing)
        stop = native_to_index([node[2], node[3], node[4]], origin, spacing)
        clipped = clip_segment(start, stop, lower, upper)
        if clipped is not None:
            segments.append([clipped[0][:2], clipped[1][:2]])
    return segments


def trace_slab(rows, origin, spacing, shape_zyx, local_z, source_plane_ids=None,
               sections: int = TRACE_SLAB_SECTIONS) -> dict:
    """Clip SWC edges to the same Z slab that the trace MIP displays.

    Edges wholly outside that slab are counted and omitted. A single central
    section is not used as the trace background.
    """
    z0, z1 = slab_index_bounds(local_z, int(shape_zyx[0]), sections)
    full = _segments_in_view(rows, origin, spacing, shape_zyx)
    kept = _segments_in_view(rows, origin, spacing, shape_zyx, z_bounds=(z0, z1))
    plane_ids = list(source_plane_ids[z0:z1]) if source_plane_ids is not None and len(source_plane_ids) >= z1 else list(range(z0, z1))
    depth_um = (z1 - z0) * float(spacing[2])
    return {
        "z_start": z0,
        "z_stop_exclusive": z1,
        "sections": z1 - z0,
        "nominal_depth_um": depth_um,
        "source_plane_ids": plane_ids,
        "background": "maximum projection of this slab",
        "edges_in_slab": len(kept),
        "edges_in_full_crop": len(full),
        "edges_outside_slab": len(full) - len(kept),
        "segments": kept,
    }


def _save_trace_panel(volume, local, spacing, rows, origin, path: Path, title: str,
                      scale_um: float, source_plane_ids=None) -> dict:
    """MIP and SWC overlay share one narrow slab. The unmarked plane stays single-section."""
    slab = trace_slab(rows, origin, spacing, volume.shape, local[2], source_plane_ids)
    z0, z1 = slab["z_start"], slab["z_stop_exclusive"]
    background = np.max(volume[z0:z1], axis=0) if z1 > z0 else volume[int(np.floor(local[2]))]
    caption = (
        f"{title}\n"
        f"trace slab {z0}:{z1} ({slab['sections']} sections, nominal {slab['nominal_depth_um']:.0f} um)\n"
        f"source planes {slab['source_plane_ids']}\n"
        f"SWC edges in slab {slab['edges_in_slab']}; excluded outside slab {slab['edges_outside_slab']}"
    )
    _draw_marked(background, path, (local[0], local[1]), spacing[:2], caption, slab["segments"], scale_um)
    return {key: value for key, value in slab.items() if key != "segments"}


def _place_export(exported: str, final: Path, center_xyz) -> None:
    """Move a crop to its final name and point the sidecar at that file and raw root."""
    exported_path = Path(exported)
    final.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != final.resolve():
        os.replace(exported_path, final)
    side_src = Path(str(exported) + ".json")
    side_dst = Path(str(final) + ".json")
    if not side_src.is_file() and not side_dst.is_file():
        return
    source = side_src if side_src.is_file() else side_dst
    meta = json.loads(source.read_text(encoding="utf-8"))
    meta["output_file"] = str(final)
    meta["center_xyz_um"] = [float(value) for value in center_xyz]
    meta["source_array_axis_order"] = "ZYX"
    meta["nifti_axis_order"] = "XYZ"
    meta["array_axis_order"] = "XYZ"
    side_dst.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if side_src.is_file() and side_src.resolve() != side_dst.resolve():
        side_src.unlink()


def _save_stack(volume, center_z: int, z_indices: list[int], path: Path) -> list[int]:
    depth = volume.shape[0]
    chosen = []
    tiles = []
    font = ImageFont.load_default()
    for offset in (-2, -1, 0, 1, 2):
        index = int(center_z) + offset
        if not 0 <= index < depth:
            continue
        gray = Image.fromarray((np.clip(_display(volume[index]), 0, 1) * 255).astype(np.uint8), mode="L")
        tile = Image.new("L", (gray.width, gray.height + 16), 0)
        tile.paste(gray, (0, 16))
        draw = ImageDraw.Draw(tile)
        label = f"z={z_indices[index]}" if index < len(z_indices) else f"k={index}"
        draw.text((4, 2), label, fill=255, font=font)
        tiles.append(tile)
        chosen.append(z_indices[index] if index < len(z_indices) else index)
    if not tiles:
        return []
    gap = 6
    canvas = Image.new("L", (sum(tile.width for tile in tiles) + gap * (len(tiles) - 1), tiles[0].height), 0)
    cursor = 0
    for tile in tiles:
        canvas.paste(tile, (cursor, 0))
        cursor += tile.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return chosen


def _save_locator(full_yx: np.ndarray, origin_xy, spacing_xy, crop_shape_xy, path: Path) -> dict:
    factor = 16
    small = np.asarray(full_yx)[::factor, ::factor].copy()
    shown = (np.clip(_display(small), 0, 1) * 255).astype(np.uint8)
    x0 = int(round(float(origin_xy[0]) / float(spacing_xy[0]) / factor))
    y0 = int(round(float(origin_xy[1]) / float(spacing_xy[1]) / factor))
    x1 = int(round((float(origin_xy[0]) / float(spacing_xy[0]) + crop_shape_xy[0]) / factor))
    y1 = int(round((float(origin_xy[1]) / float(spacing_xy[1]) + crop_shape_xy[1]) / factor))
    x0, x1 = sorted((int(np.clip(x0, 0, shown.shape[1] - 1)), int(np.clip(x1, 0, shown.shape[1] - 1))))
    y0, y1 = sorted((int(np.clip(y0, 0, shown.shape[0] - 1)), int(np.clip(y1, 0, shown.shape[0] - 1))))
    shown[y0:y0 + 2, x0:x1 + 1] = 255
    shown[max(0, y1 - 1):y1 + 1, x0:x1 + 1] = 255
    shown[y0:y1 + 1, x0:x0 + 2] = 255
    shown[y0:y1 + 1, max(0, x1 - 1):x1 + 1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shown, mode="L").save(path)
    return {"downsample": factor, "full_shape_yx": list(full_yx.shape), "box_xyxy": [x0, y0, x1, y1]}


def ortho_aspects(spacing_xyz) -> dict:
    """imshow aspect is vertical spacing / horizontal spacing, in physical units."""
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    if min(sx, sy, sz) <= 0:
        raise ValueError(f"Spacing must be positive: {spacing_xyz}")
    return {"XY": sy / sx, "XZ": sz / sx, "YZ": sz / sy}


def ortho_plane_specs(shape_zyx, local_xyz, spacing_xyz) -> dict:
    """Physical size of each orthogonal plane. shape is Z, Y, X."""
    depth, height, width = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    x = int(np.clip(np.floor(local_xyz[0]), 0, max(width - 1, 0)))
    y = int(np.clip(np.floor(local_xyz[1]), 0, max(height - 1, 0)))
    z = int(np.clip(np.floor(local_xyz[2]), 0, max(depth - 1, 0)))
    aspects = ortho_aspects(spacing_xyz)
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    return {
        "XY": {
            "aspect": aspects["XY"], "marker": (x, y),
            "width_um": width * sx, "height_um": height * sy,
            "horizontal_spacing_um": sx, "slice_index": z,
        },
        "XZ": {
            "aspect": aspects["XZ"], "marker": (x, z),
            "width_um": width * sx, "height_um": depth * sz,
            "horizontal_spacing_um": sx, "slice_index": y,
        },
        "YZ": {
            "aspect": aspects["YZ"], "marker": (y, z),
            "width_um": height * sy, "height_um": depth * sz,
            "horizontal_spacing_um": sy, "slice_index": x,
        },
    }


def build_ortho_figure(volume, local_xyz, spacing):
    """Orthogonal panels with physical aspect, the root marker, and a scale bar."""
    import matplotlib.pyplot as plt
    depth, height, width = volume.shape
    specs = ortho_plane_specs((depth, height, width), local_xyz, spacing)
    x, y = specs["XY"]["marker"]
    z = specs["XZ"]["marker"][1]
    planes = {
        "XY": volume[z],
        "XZ": volume[:, y, :],
        "YZ": volume[:, :, x],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), dpi=110, facecolor="black")
    for ax, name in zip(axes, ("XY", "XZ", "YZ")):
        spec = specs[name]
        plane = planes[name]
        ax.imshow(_display(plane), cmap="gray", vmin=0, vmax=1, origin="upper", aspect=spec["aspect"])
        mx, my = spec["marker"]
        ax.axhline(my, color="cyan", linewidth=0.6, alpha=0.85)
        ax.axvline(mx, color="cyan", linewidth=0.6, alpha=0.85)
        ax.scatter([mx], [my], s=28, marker="s", facecolors="none", edgecolors="yellow", linewidths=0.8)
        scale_um = 100.0 if spec["width_um"] >= 250 else 20.0
        bar = scale_um / float(spec["horizontal_spacing_um"])
        if bar > plane.shape[1] * 0.8:
            scale_um = max(5.0, plane.shape[1] * 0.4 * float(spec["horizontal_spacing_um"]))
            bar = scale_um / float(spec["horizontal_spacing_um"])
        y0 = plane.shape[0] - max(2.0, plane.shape[0] * 0.06)
        ax.plot([2, 2 + bar], [y0, y0], color="white", linewidth=2)
        ax.text(2, max(0, y0 - plane.shape[0] * 0.05), f"{scale_um:.0f} um", color="white", fontsize=7, va="bottom")
        ax.set_title(
            f"{name} root plane\n{spec['width_um']:.0f} x {spec['height_um']:.0f} um"
            f"\naspect {spec['aspect']:.3f}",
            color="white", fontsize=8,
        )
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def _save_ortho(volume, local_xyz, path: Path, spacing) -> None:
    import matplotlib.pyplot as plt
    fig, _axes = build_ortho_figure(volume, local_xyz, spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _needs_wider_cube(local, shape_zyx) -> bool:
    """Expand when the root is close to a face of the single native cube."""
    depth, height, width = shape_zyx
    return bool(
        local[0] < 24 or local[0] > width - 24
        or local[1] < 24 or local[1] > height - 24
        or local[2] < 8 or local[2] > depth - 8
    )


def _inside(local, shape_zyx) -> bool:
    depth, height, width = shape_zyx
    return (0 <= local[0] < width) and (0 <= local[1] < height) and (0 <= local[2] < depth)


def _rel(path: Path, out_root: Path) -> str:
    return path.relative_to(out_root).as_posix()


def production_from_coverage(context_ok: bool, soma_ok: bool, errors,
                             soma_missing=None, context_missing=None,
                             soma_complete=None, context_complete=None,
                             overview_pending: bool = False) -> tuple[str, str]:
    """Map loader metadata onto production and coverage status.

    ``ok`` requires both crops and no missing neighbor. A saved center crop
    with missing neighbors is ``partial_neighbor``. A missing center cube is
    ``center_unavailable``. A missing raw SWC is ``swc_unavailable``.
    """
    neighbor_gap = bool(soma_missing) or bool(context_missing)
    if soma_complete is False or context_complete is False:
        neighbor_gap = True
    text = " ".join(str(item) for item in (errors or []))
    if overview_pending and not context_ok:
        if soma_ok and neighbor_gap:
            return "partial", "partial_neighbor"
        if soma_ok and text:
            return "partial", "partial"
        if soma_ok:
            return "soma_only", "complete"
        if "raw swc" in text.lower() or "swc:" in text.lower():
            return "error", "swc_unavailable"
        return "partial", "center_unavailable"
    if context_ok and soma_ok and neighbor_gap:
        return "partial", "partial_neighbor"
    if context_ok and soma_ok and not text:
        return "ok", "complete"
    if context_ok and soma_ok:
        return "partial", "partial"
    if context_ok and not soma_ok:
        return "partial", "center_unavailable"
    if soma_ok and not context_ok:
        return "partial", "partial"
    if "raw swc" in text.lower() or "swc:" in text.lower():
        return "error", "swc_unavailable"
    return "error", "failed"


def render_neuron(row, rows, swc_sha: str, swc_url: str, swc_source: str,
                  toolkit, out_root: Path, include_context: bool = True) -> dict:
    """Write unmarked, marked, trace, stack and orthogonal evidence for one identity."""
    sample = str(row["SampleID"])
    neuron = canon_neuron_id(row["NeuronID"])
    folder = out_root / sample / neuron.replace(".swc", "")
    folder.mkdir(parents=True, exist_ok=True)
    native = _root_xyz(rows)
    record = {
        "SampleID": sample, "NeuronID": neuron, "UID": f"{sample}|{neuron}",
        "native_soma_x_um": native[0], "native_soma_y_um": native[1], "native_soma_z_um": native[2],
        "native_frame": "raw SWC root XYZ; declared native micrometres; orientation unverified",
        "swc_url": swc_url, "swc_sha256": swc_sha, "swc_source": swc_source,
        "swc_node_count": len(rows),
        "declared_spacing_low_xyz_um": [5.0, 5.0, 3.0],
        "declared_spacing_high_xyz_um": [0.65, 0.65, 3.0],
        "context_requested_um_xyz": list(CONTEXT_UM),
        "panel_version": PANEL_VERSION,
        "overview_source_status": "copied_ch1" if include_context else "overview_pending",
        "errors": [],
    }
    context_ok = False
    if not include_context:
        record["context_coverage_status"] = "overview_pending"
        record["context_pending_reason"] = (
            "Copied 5 um CH1 overview is absent. No other sample context was used. "
            "Portal NMT coordinates were not used."
        )
    else:
        try:
            volume, origin, spacing = toolkit.get_low_res_widefield(
                native, width_um=CONTEXT_UM[0], height_um=CONTEXT_UM[1], depth_um=CONTEXT_UM[2])
            meta = toolkit.last_low_res_metadata or {}
            local = native_to_index(native, origin, spacing)
            if not _inside(local, volume.shape):
                raise ValueError("Native SWC root is outside the low-resolution context crop")
            z = int(np.floor(local[2]))
            plane = volume[z]
            stats = _save_gray(plane, folder / "context_unmarked.png")
            loaded = meta.get("loaded") or []
            plane_ids = context_plane_ids(meta, volume.shape[0], origin, spacing)
            z_index = plane_ids[z]
            title = (
                f"{sample} {neuron} context z-index {z_index}\n"
                f"nominal {spacing} um | FOV {plane.shape[1] * spacing[0]:.0f} x {plane.shape[0] * spacing[1]:.0f} um\n"
                f"native XY, orientation unverified | portal label is not a correction: {display_label(row.get('portal_region_base'))}"
            )
            _draw_marked(plane, folder / "context_marked.png", (local[0], local[1]), spacing[:2], title, scale_um=500)
            context_trace = _save_trace_panel(
                volume, local, spacing, rows, origin, folder / "context_trace.png", title, 500, plane_ids)
            stack_ids = _save_stack(volume, z, plane_ids, folder / "context_stack.png")
            nii = toolkit.export_data(volume, origin, spacing, neuron.replace(".swc", ""),
                                      suffix="Context", soma_coords=native, output_dir=str(folder))
            _place_export(nii, folder / "context.nii.gz", native)
            locator = {}
            sources = meta.get("source_paths") or []
            central_path = ""
            for source in sources:
                if isinstance(source, dict) and source.get("z_index") == z_index:
                    central_path = source.get("path", "")
            if central_path and os.path.isfile(central_path):
                full = toolkit._read_cached_section(central_path)
                locator = _save_locator(full, origin[:2], spacing[:2], (volume.shape[2], volume.shape[1]),
                                        folder / "section_locator.png")
                if sample not in os.path.basename(central_path):
                    raise ValueError(f"Context slice name does not match sample {sample}")
            record.update({
                "context_origin_xyz_um": origin, "context_shape_zyx": list(volume.shape),
                "context_local_xyz": [float(value) for value in local],
                "context_central_z_index": z_index, "context_loaded_z": loaded,
                "context_plane_ids": plane_ids,
                "context_missing_z": meta.get("missing"), "context_complete": meta.get("complete"),
                "context_source_paths": sources, "context_central_path": central_path,
                "context_plane_stats": stats, "context_trace_segments": context_trace["edges_in_slab"],
                "context_trace_slab": context_trace,
                "context_stack_z": stack_ids, "context_locator": locator,
                "context_clipped": meta.get("crop_clipped_to_image_bounds"),
                "soma_inside_context": True,
            })
            context_ok = stats["nonzero_fraction"] > 0 and bool(central_path)
            if stats["nonzero_fraction"] <= 0:
                record["errors"].append("context central plane is empty")
        except Exception as exc:
            record["errors"].append(f"context: {exc}")
            record["context_metadata"] = toolkit.last_low_res_metadata

    soma_ok = False
    try:
        volume, origin, spacing = toolkit.get_high_res_block(native, grid_radius=HIGH_GRID_RADIUS)
        meta = toolkit.last_high_res_metadata or {}
        local = native_to_index(native, origin, spacing)
        if not _inside(local, volume.shape):
            raise ValueError("Native SWC root is outside the high-resolution soma cube")
        # A root on the cube face loses proximal branches. Ask for the 3x3x3
        # neighborhood, keep explicit zero gaps, and fall back to the single cube.
        if _needs_wider_cube(local, volume.shape):
            narrow_meta = meta
            try:
                wider, wider_origin, wider_spacing = toolkit.get_high_res_block(
                    native, grid_radius=2, allow_partial=True)
                wider_local = native_to_index(native, wider_origin, wider_spacing)
                if _inside(wider_local, wider.shape):
                    volume, origin, spacing = wider, wider_origin, wider_spacing
                    local = wider_local
                    meta = toolkit.last_high_res_metadata or {}
                    record["soma_field"] = "3x3x3 cubes because the root is near a cube face"
                else:
                    toolkit.last_high_res_metadata = narrow_meta
                    record["soma_field"] = "single cube; wider field did not contain the root"
            except Exception as exc:
                toolkit.last_high_res_metadata = narrow_meta
                meta = narrow_meta
                record["errors"].append(f"wider soma cube: {exc}")
                record["soma_field"] = "single cube after wider-field failure"
        else:
            record["soma_field"] = "single native cube"
        z = int(np.floor(local[2]))
        plane = volume[z]
        stats = _save_gray(plane, folder / "soma_unmarked.png")
        loaded_n = len(meta.get("loaded") or [])
        missing_n = len(meta.get("missing") or [])
        title = (
            f"{sample} {neuron} soma plane\n"
            f"nominal {spacing[0]:g} x {spacing[1]:g} x {spacing[2]:g} um | "
            f"{record.get('soma_field', 'native cube')}\n"
            f"cubes loaded {loaded_n}, missing {missing_n} | native XY, orientation unverified"
        )
        _draw_marked(plane, folder / "soma_marked.png", (local[0], local[1]), spacing[:2], title, scale_um=50)
        soma_trace = _save_trace_panel(
            volume, local, spacing, rows, origin, folder / "soma_trace.png", title, 50,
            [f"k{index}" for index in range(volume.shape[0])])
        plane_labels = [f"k{index}" for index in range(volume.shape[0])]
        _save_stack(volume, z, plane_labels, folder / "soma_stack.png")
        _save_ortho(volume, local, folder / "soma_ortho.png", spacing)
        nii = toolkit.export_data(volume, origin, spacing, neuron.replace(".swc", ""),
                                  suffix="SomaBlock", soma_coords=native, output_dir=str(folder))
        _place_export(nii, folder / "soma.nii.gz", native)
        urls = meta.get("source_paths") or []
        if any(sample not in str(item) for item in urls):
            raise ValueError("High-resolution source URL does not match the requested sample")
        near_edge = min(local[0], local[1], volume.shape[2] - local[0], volume.shape[1] - local[1]) < 20
        soma_missing = meta.get("missing") or []
        record.update({
            "soma_origin_xyz_um": origin, "soma_shape_zyx": list(volume.shape),
            "soma_local_xyz": [float(value) for value in local],
            "soma_loaded": meta.get("loaded"), "soma_missing": soma_missing,
            "soma_complete": meta.get("complete"),
            "soma_source_urls": urls, "soma_plane_stats": stats,
            "soma_trace_segments": soma_trace["edges_in_slab"],
            "soma_trace_slab": soma_trace, "soma_inside_cube": True,
            "soma_near_block_edge": bool(near_edge),
            "ortho_spacing_xyz_um": [float(value) for value in spacing],
            "ortho_aspect": ortho_aspects(spacing),
        })
        soma_ok = stats["nonzero_fraction"] > 0
        if not soma_ok:
            record["errors"].append("soma central plane is empty")
        elif soma_missing or meta.get("complete") is False:
            record["errors"].append(
                f"high-resolution neighbor cubes missing: {len(soma_missing)}; center crop retained")
    except Exception as exc:
        record["errors"].append(f"soma: {exc}")
        record["soma_metadata"] = toolkit.last_high_res_metadata

    soma_meta = record.get("soma_metadata") if not soma_ok else None
    status, coverage = production_from_coverage(
        context_ok, soma_ok, record["errors"],
        soma_missing=record.get("soma_missing"),
        context_missing=record.get("context_missing_z"),
        soma_complete=record.get("soma_complete"),
        context_complete=record.get("context_complete"),
        overview_pending=not include_context,
    )
    if soma_meta and coverage == "center_unavailable":
        record["soma_missing"] = (soma_meta or {}).get("missing")
    record["production_status"] = status
    record["coverage_status"] = coverage
    (folder / "provenance.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _files_ready(folder: Path, names) -> bool:
    return all((folder / name).is_file() and (folder / name).stat().st_size > 500 for name in names)


def _already_done(folder: Path) -> bool:
    """Skip a finished crop. Missing-SWC rows stay eligible for a bounded retry."""
    provenance = folder / "provenance.json"
    if not provenance.is_file():
        return False
    try:
        record = json.loads(provenance.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    coverage = record.get("coverage_status")
    status = record.get("production_status")
    if record.get("overview_source_status") == "overview_pending" or status == "soma_only":
        if coverage == "swc_unavailable":
            return bool(record.get("swc_attempts_exhausted"))
        if coverage == "center_unavailable":
            return True
        if status in {"soma_only", "partial"} and coverage == "partial_neighbor":
            return _files_ready(folder, SOMA_PACKAGE_FILES)
        if status == "soma_only":
            return _files_ready(folder, SOMA_PACKAGE_FILES)
        return False
    if coverage == "swc_unavailable" or status == "error":
        return False
    if record.get("soma_missing") and status == "ok":
        return False
    if coverage == "partial_neighbor":
        return _files_ready(folder, PANEL_FILES)
    if coverage == "center_unavailable":
        return _files_ready(folder, CONTEXT_RESUME_FILES)
    if status == "ok" and record.get("panel_version") == PANEL_VERSION and coverage in (None, "complete"):
        return _files_ready(folder, PANEL_FILES)
    return False


def render_available(samples: list[str] | None = None, neurons: dict | None = None,
                     out_root: Path = OUT_ROOT, force: bool = False,
                     mode: str = "overview") -> list[dict]:
    """Render copied-overview packages, or own-sample soma packages for overview-pending rows."""
    from Visual_toolkit import Visual_toolkit
    manifest = load_manifest(out_root)
    include_context = mode != "soma"
    status_name = "available" if include_context else "source_pending"
    chosen = manifest[manifest["image_source_status"] == status_name].copy()
    if samples:
        chosen = chosen[chosen["SampleID"].isin(samples)]
    if neurons:
        mask = chosen.apply(lambda row: row["NeuronID"] in neurons.get(str(row["SampleID"]), set()), axis=1)
        chosen = chosen[mask]
    order = (
        chosen.groupby("SampleID")["sample_insula_count"].max().sort_values(ascending=False).index.tolist())
    results = []
    log_path = out_root / "manifest" / "render_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for sample in order:
        subset = chosen[chosen["SampleID"] == sample].sort_values(["category_rank", "NeuronID"])
        print(f"[render] {sample}: {len(subset)} candidates", flush=True)
        toolkit = Visual_toolkit(sample, cache_dir=str(out_root / "cache"), low_res_slice_cache_limit=3)
        try:
            prepared = []
            for row in subset.to_dict(orient="records"):
                neuron = canon_neuron_id(row["NeuronID"])
                folder = out_root / sample / neuron.replace(".swc", "")
                if not force and _already_done(folder):
                    record = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
                    results.append(record)
                    print(f"  skip {neuron}", flush=True)
                    continue
                try:
                    swc_rows, digest, url, source = fetch_raw_swc(
                        sample, neuron, str(row["project_id"]), out_root)
                    prepared.append((_root_xyz(swc_rows)[2], row, digest, url, source))
                    del swc_rows
                except Exception as exc:
                    _status, coverage = production_from_coverage(False, False, [f"swc: {exc}"])
                    record = {
                        "SampleID": sample, "NeuronID": neuron, "UID": f"{sample}|{neuron}",
                        "production_status": "error", "coverage_status": coverage,
                        "overview_source_status": "overview_pending" if str(row.get("image_source_status")) == "source_pending" else "copied_ch1",
                        "errors": [f"swc: {exc}"],
                        "nmt_fallback": False,
                        "swc_attempts_exhausted": True,
                    }
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    results.append(record)
                    print(f"  swc fail {neuron}: {exc}", flush=True)
            prepared.sort(key=lambda item: item[0])
            for _z, row, digest, url, source in prepared:
                neuron = canon_neuron_id(row["NeuronID"])
                try:
                    swc_path = out_root / "swc_raw" / sample / neuron
                    swc_rows = parse_swc(swc_path.read_text(encoding="utf-8", errors="replace"), source=str(swc_path))
                    record = render_neuron(
                        row, swc_rows, digest, url, source, toolkit, out_root,
                        include_context=include_context)
                    del swc_rows
                except Exception as exc:
                    record = {
                        "SampleID": sample, "NeuronID": neuron, "UID": f"{sample}|{neuron}",
                        "production_status": "error",
                        "errors": [f"render: {exc}", traceback.format_exc(limit=2)],
                    }
                results.append(record)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "uid": record["UID"], "status": record["production_status"],
                        "errors": record.get("errors"),
                    }, default=str) + "\n")
                print(f"  {record['production_status']} {neuron}", flush=True)
        finally:
            toolkit.close()
    return results


def _blank_human() -> dict:
    return {name: "" for name in HUMAN_FIELDS}


def _cell_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def load_human_annotations(path: Path) -> dict:
    """Read reviewer columns. Conflicting duplicate UIDs in one file abort."""
    if not path.is_file():
        return {}
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str)
    else:
        frame = pd.read_excel(path, dtype=str)
    if "UID" not in frame.columns:
        return {}
    grouped = {}
    order = []
    for row in frame.to_dict(orient="records"):
        uid = _cell_text(row.get("UID"))
        if not uid:
            continue
        if uid not in grouped:
            order.append(uid)
            grouped[uid] = []
        grouped[uid].append({name: _cell_text(row.get(name)) for name in HUMAN_FIELDS})
    conflicts = []
    found = {}
    for uid in order:
        merged = {}
        for name in HUMAN_FIELDS:
            values = []
            for item in grouped[uid]:
                if item[name] and item[name] not in values:
                    values.append(item[name])
            if len(values) > 1:
                conflicts.append(f"{uid} {name}: {' vs '.join(repr(value) for value in values)}")
            merged[name] = values[0] if values else ""
        found[uid] = merged
    if conflicts:
        raise RuntimeError(
            "Duplicate reviewer rows conflict within one table; nothing was rewritten:\n" + "\n".join(conflicts))
    return found


def merge_human_annotations(csv_map: dict, xlsx_map: dict) -> dict:
    """Keep a nonblank answer from either file. Disagreeing nonblank values abort."""
    conflicts = []
    merged = {}
    for uid in set(csv_map) | set(xlsx_map):
        left = csv_map.get(uid, {})
        right = xlsx_map.get(uid, {})
        item = {}
        for name in HUMAN_FIELDS:
            csv_value = left.get(name, "")
            xlsx_value = right.get(name, "")
            if csv_value and xlsx_value and csv_value != xlsx_value:
                conflicts.append(f"{uid} {name}: csv {csv_value!r} vs xlsx {xlsx_value!r}")
            item[name] = csv_value or xlsx_value
        merged[uid] = item
    if conflicts:
        raise RuntimeError(
            "Human annotation conflict; review tables were not rewritten:\n" + "\n".join(conflicts))
    return merged


def table_source_errors(errors: list, out_root: Path) -> str:
    """Keep short diagnostics inline; retain oversized lists in a hashed sidecar."""
    encoded = json.dumps(errors)
    if len(encoded) <= 32767:
        return encoded
    payload = (encoded + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = out_root / "tables" / "details" / f"source_errors_{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() != payload:
        raise RuntimeError(f"Diagnostic sidecar hash mismatch: {path}")
    if not path.is_file():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return json.dumps({"detail_file": _rel(path, out_root), "sha256": digest,
                       "items": len(errors)})


def write_correction_table(out_root: Path = OUT_ROOT) -> pd.DataFrame:
    table_dir = out_root / "tables"
    csv_path = table_dir / "correction_table.csv"
    xlsx_path = table_dir / "correction_table.xlsx"
    archive_path = table_dir / "orphaned_annotations.csv"
    human = merge_human_annotations(load_human_annotations(csv_path), load_human_annotations(xlsx_path))
    if archive_path.is_file():
        human = merge_human_annotations(human, load_human_annotations(archive_path))
    manifest = load_manifest(out_root)
    manifest_uids = set(manifest["UID"].astype(str))
    orphans = []
    for uid, fields in human.items():
        if uid not in manifest_uids and any(fields.values()):
            item = {"UID": uid, "archive_reason": "UID absent from the current identity manifest"}
            item.update(fields)
            orphans.append(item)
    table_dir.mkdir(parents=True, exist_ok=True)
    orphan_frame = pd.DataFrame(orphans, columns=["UID", *HUMAN_FIELDS, "archive_reason"])
    orphan_frame.to_csv(archive_path, index=False)
    (table_dir / "annotation_archive.json").write_text(json.dumps({
        "archive": str(archive_path),
        "orphaned_uids": [item["UID"] for item in orphans],
        "count": len(orphans),
    }, indent=2), encoding="utf-8")
    records = []
    for row in manifest.to_dict(orient="records"):
        sample = str(row["SampleID"])
        neuron = canon_neuron_id(row["NeuronID"])
        folder = out_root / sample / neuron.replace(".swc", "")
        provenance = {}
        path = folder / "provenance.json"
        if path.is_file():
            provenance = json.loads(path.read_text(encoding="utf-8"))
        status = provenance.get("production_status")
        if row["image_source_status"] == "source_pending" and status not in {"soma_only", "partial", "error"}:
            status = status or "source_pending"
        elif not status:
            status = "not_started"
        item = dict(row)
        item.update(human.get(f"{sample}|{neuron}", _blank_human()))
        item.update({
            "production_status": status,
            "coverage_status": provenance.get("coverage_status", ""),
            "overview_source_status": provenance.get("overview_source_status") or row.get("overview_source_status", ""),
            "context_coverage_status": provenance.get("context_coverage_status", ""),
            "context_pending_reason": provenance.get("context_pending_reason", ""),
            "context_source_errors": table_source_errors(
                (provenance.get("derived_context") or {}).get("source_errors") or [], out_root),
            "context_coverage_fraction": (provenance.get("derived_context") or {}).get("coverage_fraction", ""),
            "context_requested_field_xyz_um": json.dumps(provenance.get("context_requested_um_xyz") or []),
            "staging_region_source": row.get("staging_region_source", ""),
            "error": " | ".join(str(err) for err in provenance.get("errors") or []),
            "native_soma_x_um": provenance.get("native_soma_x_um", ""),
            "native_soma_y_um": provenance.get("native_soma_y_um", ""),
            "native_soma_z_um": provenance.get("native_soma_z_um", ""),
            "native_frame": provenance.get("native_frame", ""),
            "swc_url": provenance.get("swc_url", ""),
            "swc_sha256": provenance.get("swc_sha256", ""),
            "swc_node_count": provenance.get("swc_node_count", ""),
            "context_central_slice": provenance.get("context_central_path", ""),
            "context_loaded_z": json.dumps(provenance.get("context_loaded_z")) if provenance.get("context_loaded_z") is not None else "",
            "context_missing_z": json.dumps(provenance.get("context_missing_z")) if provenance.get("context_missing_z") is not None else "",
            "soma_source_urls": json.dumps(provenance.get("soma_source_urls")) if provenance.get("soma_source_urls") is not None else "",
            "soma_missing": json.dumps(provenance.get("soma_missing")) if provenance.get("soma_missing") is not None else "",
            "context_nonzero_fraction": (provenance.get("context_plane_stats") or {}).get("nonzero_fraction", ""),
            "soma_nonzero_fraction": (provenance.get("soma_plane_stats") or {}).get("nonzero_fraction", ""),
            "soma_inside_context": provenance.get("soma_inside_context", ""),
            "soma_inside_cube": provenance.get("soma_inside_cube", ""),
            "context_trace_segments": provenance.get("context_trace_segments", ""),
            "soma_trace_segments": provenance.get("soma_trace_segments", ""),
            "declared_spacing_low_xyz_um": ",".join(str(value) for value in (
                (provenance.get("derived_context") or {}).get("derived_spacing_xyz_um") or [5, 5, 3])),
            "declared_spacing_high_xyz_um": "0.65,0.65,3",
        })
        for name in (*PANEL_FILES, *OPTIONAL_PANEL_FILES):
            file_path = folder / name
            item[name] = _rel(file_path, out_root) if file_path.is_file() else ""
        records.append(item)
    frame = pd.DataFrame(records).fillna("")
    for item in records:
        for field, value in item.items():
            if isinstance(value, str) and len(value) > 32767:
                raise RuntimeError(f"Excel cell limit exceeded: {item['UID']} {field}; tables were not rewritten")
    table_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False, sheet_name="Review")
    write_component_queue(frame, out_root)
    return frame


def write_component_queue(frame: pd.DataFrame, out_root: Path = OUT_ROOT) -> pd.DataFrame:
    """List the source components still absent for each identity. Does not invent images."""
    records = []
    for row in frame.to_dict(orient="records"):
        missing = []
        overview = str(row.get("overview_source_status") or "")
        context_state = str(row.get("context_coverage_status") or "")
        coverage = str(row.get("coverage_status") or "")
        status = str(row.get("production_status") or "")
        if status in {"", "source_pending", "not_started"}:
            missing.append("soma_package")
        elif coverage == "center_unavailable":
            missing.append("soma_center_cube")
        elif coverage == "swc_unavailable":
            missing.append("raw_swc")
        else:
            for name in SOMA_PACKAGE_FILES:
                if not str(row.get(name) or ""):
                    missing.append(name)
            if coverage == "partial_neighbor":
                missing.append("soma_neighbor_cubes")
        if overview == "overview_pending" and context_state not in DERIVED_CONTEXT_STATES:
            missing.extend(["copied_5um_overview", "regional_context"])
        elif context_state == "derived_partial":
            missing.append("derived_context_missing_tiles")
        elif overview == "copied_ch1" and coverage != "swc_unavailable":
            for name in CONTEXT_PACKAGE_FILES:
                if not str(row.get(name) or ""):
                    missing.append(name)
        if context_state in DERIVED_CONTEXT_STATES:
            for name in CONTEXT_PACKAGE_FILES:
                if not str(row.get(name) or ""):
                    missing.append(name)
        deduped = list(dict.fromkeys(missing))
        if not deduped:
            continue
        records.append({
            "SampleID": row.get("SampleID", ""),
            "NeuronID": row.get("NeuronID", ""),
            "UID": row.get("UID", ""),
            "review_category": row.get("review_category", ""),
            "overview_source_status": overview,
            "production_status": status,
            "coverage_status": coverage,
            "context_coverage_status": context_state,
            "missing_components": "|".join(deduped),
        })
    queue = pd.DataFrame(records, columns=[
        "SampleID", "NeuronID", "UID", "review_category", "overview_source_status",
        "production_status", "coverage_status", "context_coverage_status", "missing_components",
    ])
    manifest_dir = out_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    queue.to_csv(manifest_dir / "component_queue.csv", index=False)
    return queue


def write_gallery(out_root: Path = OUT_ROOT) -> None:
    frame = write_correction_table(out_root)
    gallery = out_root / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    pages = []
    for sample, subset in frame.groupby("SampleID", sort=False):
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(str(sample))} native review</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee}",
            "img{max-width:420px;margin:4px;background:#000} .card{border-top:1px solid #444;padding:12px}</style></head><body>",
            f"<h1>{html.escape(str(sample))}</h1>",
            "<p>Nominal spacing is a repository declaration. Atlas text is provenance, not a correction. "
            "Human fields stay empty until a reviewer fills them; existing answers are kept. "
            "partial_neighbor keeps the center crop. center_unavailable has overview context only. "
            "swc_unavailable has no native crop. overview_pending has no copied 5 um series; "
            "a soma package from that sample's own raw SWC and cubes is not regional context.</p>",
        ]
        status = subset["image_source_status"].iloc[0] if len(subset) else ""
        if status == "source_pending":
            parts.append(
                f"<p>Overview source pending for {len(subset)} identities. "
                "Soma packages use this sample's own raw SWC and high-resolution cubes only.</p>")
        for row in subset.to_dict(orient="records"):
            if row.get("image_source_status") == "source_pending" and row.get("production_status") in {"", "source_pending", "not_started"}:
                parts.append(
                    "<div class='card'><h2>{} {}</h2><p>{} | {} | overview pending | soma not started</p>"
                    "<p>portal {} | step1 {} | staging source {}</p><p>reasons: {}</p></div>".format(
                        html.escape(str(row["SampleID"])), html.escape(str(row["NeuronID"])),
                        html.escape(str(row["review_category"])), html.escape(str(row["UID"])),
                        html.escape(str(row.get("portal_region") or "")),
                        html.escape(str(row.get("step1_soma_region") or "")),
                        html.escape(str(row.get("staging_region_source") or "")),
                        html.escape(str(row.get("selection_reasons") or ""))))
                continue
            links = []
            for name in ("context_unmarked.png", "context_marked.png", "context_trace.png",
                         "context_coverage.png", "context_ortho.png", "section_locator.png", "context_stack.png", "soma_unmarked.png",
                         "soma_marked.png", "soma_trace.png", "soma_ortho.png", "soma_stack.png"):
                rel = row.get(name) or ""
                if rel:
                    links.append(f"<figure><img src='../{html.escape(rel)}' alt='{html.escape(name)}'><figcaption>{html.escape(name)}</figcaption></figure>")
            human_bits = [
                f"{name}: {row.get(name)}" for name in HUMAN_FIELDS if _cell_text(row.get(name))
            ]
            parts.append(
                "<div class='card'><h2>{} {}</h2><p>{} | {} | status {} | soma {} | context {}</p><p>portal {} | step1 {} | refined {} | staging source {}</p><p>reasons: {}</p><p>{}</p>{}</div>".format(
                    html.escape(str(row["SampleID"])), html.escape(str(row["NeuronID"])),
                    html.escape(str(row["review_category"])), html.escape(str(row["UID"])),
                    html.escape(str(row["production_status"])),
                    html.escape(str(row.get("coverage_status") or "")),
                    html.escape(str(row.get("context_coverage_status") or row.get("overview_source_status") or "")
                                + (f"; acquired {float(row['context_coverage_fraction']):.1%} of sampled field"
                                   if _cell_text(row.get("context_coverage_fraction")) else "")),
                    html.escape(str(row["portal_region"])), html.escape(str(row["step1_soma_region"])),
                    html.escape(str(row["staging_refined_label"])),
                    html.escape(str(row.get("staging_region_source") or "")),
                    html.escape(str(row["selection_reasons"])),
                    html.escape("; ".join(human_bits)),
                    "".join(links)))
        page = gallery / f"{sample}.html"
        page.write_text("".join(parts) + "</body></html>", encoding="utf-8")
        pages.append((sample, len(subset), status))
    index = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Native review gallery</title></head><body>",
             "<h1>fMOST native review 2026-10-02</h1><ul>"]
    for sample, count, status in pages:
        index.append(f"<li><a href='{sample}.html'>{sample}</a> — {count} identities ({status})</li>")
    index.append("</ul></body></html>")
    (gallery / "index.html").write_text("".join(index), encoding="utf-8")
    print(f"[gallery] {gallery / 'index.html'}")


def reconcile_coverage_records(out_root: Path = OUT_ROOT) -> dict:
    """Relabel saved crops from their metadata. Does not download or redraw images."""
    summary = {"complete": [], "partial_neighbor": [], "center_unavailable": [], "swc_unavailable": [], "other": []}
    for sample in AVAILABLE_SAMPLES:
        root = out_root / sample
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/provenance.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            errors = [str(item) for item in (record.get("errors") or [])]
            soma_missing = record.get("soma_missing") or []
            context_missing = record.get("context_missing_z") or []
            folder = path.parent
            context_ok = bool(record.get("soma_inside_context")) and (folder / "context.nii.gz").is_file()
            soma_ok = bool(record.get("soma_inside_cube")) and (folder / "soma.nii.gz").is_file()
            soma_complete = False if soma_missing else record.get("soma_complete", True)
            status, coverage = production_from_coverage(
                context_ok, soma_ok, errors,
                soma_missing=soma_missing, context_missing=context_missing,
                soma_complete=soma_complete, context_complete=record.get("context_complete", True),
            )
            if coverage == "partial_neighbor":
                note = f"high-resolution neighbor cubes missing: {len(soma_missing)}; center crop retained"
                if note not in errors:
                    errors.append(note)
                record["soma_complete"] = False
            record["errors"] = errors
            record["production_status"] = status
            record["coverage_status"] = coverage
            path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            summary.setdefault(coverage, []).append(record.get("UID"))
    print("[coverage] " + ", ".join(f"{key}={len(value)}" for key, value in summary.items()))
    return summary


def refresh_ortho_panels(out_root: Path = OUT_ROOT) -> int:
    """Redraw soma_ortho.png from existing NIfTIs using physical axis aspect."""
    import nibabel as nib
    count = 0
    for sample in AVAILABLE_SAMPLES:
        root = out_root / sample
        if not root.is_dir():
            continue
        for nii_path in sorted(root.glob("*/soma.nii.gz")):
            folder = nii_path.parent
            prov_path = folder / "provenance.json"
            record = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
            image = nib.load(str(nii_path))
            data_xyz = np.asanyarray(image.dataobj)
            volume = np.transpose(data_xyz, (2, 1, 0))
            spacing = [float(value) for value in np.diag(image.affine)[:3]]
            local = record.get("soma_local_xyz")
            if not local:
                local = native_to_index(
                    [record["native_soma_x_um"], record["native_soma_y_um"], record["native_soma_z_um"]],
                    record["soma_origin_xyz_um"], spacing)
            _save_ortho(volume, local, folder / "soma_ortho.png", spacing)
            record["ortho_spacing_xyz_um"] = spacing
            record["ortho_aspect"] = ortho_aspects(spacing)
            record["ortho_physical_um_xyz"] = [
                float(volume.shape[2] * spacing[0]),
                float(volume.shape[1] * spacing[1]),
                float(volume.shape[0] * spacing[2]),
            ]
            record["soma_shape_zyx"] = [int(value) for value in volume.shape]
            if prov_path.is_file():
                prov_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            count += 1
            print(
                f"[ortho] {record.get('UID', folder.name)} "
                f"{volume.shape[2]}x{volume.shape[1]}x{volume.shape[0]} "
                f"XZ {record['ortho_physical_um_xyz'][0]:.0f}x{record['ortho_physical_um_xyz'][2]:.0f} um",
                flush=True)
            del data_xyz, volume
    print(f"[ortho] refreshed {count}")
    return count


def verify_outputs(out_root: Path = OUT_ROOT) -> dict:
    """Read persisted panels and hashes. Does not rewrite the review tables."""
    import nibabel as nib
    problems = []
    csv_path = out_root / "tables" / "correction_table.csv"
    xlsx_path = out_root / "tables" / "correction_table.xlsx"
    try:
        merge_human_annotations(load_human_annotations(csv_path), load_human_annotations(xlsx_path))
    except RuntimeError as exc:
        problems.append(str(exc).splitlines()[0])
    if not csv_path.is_file():
        problems.append("correction table missing")
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(csv_path, dtype=str).fillna("")
    baseline_path = out_root / "manifest" / "canonical_hashes.json"
    if not baseline_path.is_file():
        problems.append("canonical baseline missing")
        hash_ok = False
    else:
        saved = _canonical_map(json.loads(baseline_path.read_text(encoding="utf-8")))
        hash_ok = saved == canonical_hashes()
        if not hash_ok:
            problems.append("canonical cohort or Henry table hash changed")
    if frame.empty or "image_source_status" not in frame.columns:
        pending = frame
        available = frame
    else:
        pending = frame[frame["image_source_status"] == "source_pending"]
        available = frame[frame["image_source_status"] == "available"]
    image_cols = list(PANEL_FILES)
    if len(pending) and "overview_source_status" not in pending.columns:
        if "production_status" in pending.columns and pending["production_status"].ne("source_pending").any():
            problems.append("pending row is missing source_pending status")
        if all(name in pending.columns for name in image_cols):
            pending_images = pending[image_cols].astype(str).apply(lambda col: col.str.len().gt(0)).any(axis=1)
            if pending_images.any():
                problems.append("source-pending row has an image path")
    elif len(pending):
        for row in pending.to_dict(orient="records"):
            context_state = str(row.get("context_coverage_status") or "")
            overview = str(row.get("overview_source_status") or "overview_pending")
            for name in CONTEXT_PACKAGE_FILES:
                if row.get(name) and context_state not in DERIVED_CONTEXT_STATES and overview == "overview_pending":
                    problems.append(f"{row['UID']} overview-pending row has context file {name}")
            soma_state = str(row.get("coverage_status") or "")
            if soma_state == "complete" or row.get("production_status") == "soma_only":
                for name in SOMA_PACKAGE_FILES:
                    if not row.get(name):
                        problems.append(f"{row['UID']} soma package missing {name}")
                        break
    for sample in PENDING_SAMPLES:
        folder = out_root / sample
        if not folder.exists():
            continue
        for png in folder.rglob("*.png"):
            if not (png.name.startswith("context") or png.name == "section_locator.png"):
                continue
            state = ""
            prov = png.parent / "provenance.json"
            if prov.is_file():
                state = json.loads(prov.read_text(encoding="utf-8")).get("context_coverage_status", "")
            if state not in DERIVED_CONTEXT_STATES:
                problems.append(f"pending sample {sample} has overview image {png.name}")
    if "production_status" not in available.columns:
        ok = available.iloc[0:0]
    else:
        ok = available[available["production_status"] == "ok"]
    for row in ok.to_dict(orient="records"):
        required = SOMA_PACKAGE_FILES if row.get("production_status") == "soma_only" else PANEL_FILES
        for name in required:
            if not row.get(name):
                problems.append(f"{row['UID']} ok row missing {name}")
                break
    # Stratified read-back: one ok row per available sample and category, plus every error.
    inspected = []
    ok_groups = ok.groupby(["SampleID", "review_category"], dropna=False)
    picks = []
    for _, subset in ok_groups:
        picks.append(subset.iloc[0])
    errors = available[available["production_status"].isin(["error", "partial"])]
    for _, row in errors.iterrows():
        picks.append(row)
    for row in picks:
        uid = row["UID"]
        for nii_name, shape_key in (("context.nii.gz", "context"), ("soma.nii.gz", "soma")):
            rel = row.get(nii_name) or ""
            if not rel:
                if row["production_status"] == "ok":
                    problems.append(f"{uid} missing {nii_name}")
                continue
            path = out_root / rel
            image = nib.load(str(path))
            data = np.asanyarray(image.dataobj)
            spacing = np.diag(image.affine)[:3]
            if not np.allclose(spacing, [5, 5, 3] if "context" in nii_name else [0.65, 0.65, 3]):
                problems.append(f"{uid} {nii_name} spacing {spacing.tolist()}")
            if str(image.header.get_xyzt_units()[0]) != "micron":
                problems.append(f"{uid} {nii_name} units are not micron")
            if data.ndim != 3 or int(np.max(data)) == 0:
                problems.append(f"{uid} {nii_name} is empty")
            inspected.append({"uid": uid, "file": nii_name, "shape_xyz": list(data.shape), "max": int(np.max(data))})
        unmarked = row.get("context_unmarked.png") or ""
        if unmarked:
            with Image.open(out_root / unmarked) as picture:
                if picture.size[0] < 10 or picture.mode != "L":
                    problems.append(f"{uid} context unmarked is not a grayscale panel")
        for nii_name in ("context.nii.gz", "soma.nii.gz"):
            rel = row.get(nii_name) or ""
            if not rel:
                continue
            sidecar = Path(str(out_root / rel) + ".json")
            if not sidecar.is_file():
                if row["production_status"] == "ok":
                    problems.append(f"{uid} missing sidecar {sidecar.name}")
                continue
            try:
                json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                problems.append(f"{uid} sidecar {sidecar.name} is not valid JSON")
    if len(available):
        for row in available.to_dict(orient="records"):
            if row.get("production_status") not in {"ok", "partial"}:
                continue
            uid = row["UID"]
            swc_path = out_root / "swc_raw" / str(row["SampleID"]) / str(row["NeuronID"])
            if not swc_path.is_file():
                problems.append(f"{uid} raw SWC cache missing")
                continue
            try:
                root = _root_xyz(parse_swc(
                    swc_path.read_text(encoding="utf-8", errors="replace"), source=str(swc_path)))
            except (OSError, ValueError) as exc:
                problems.append(f"{uid} raw SWC unreadable: {exc}")
                continue
            for axis, key in zip(root, ("native_soma_x_um", "native_soma_y_um", "native_soma_z_um")):
                text = str(row.get(key) or "")
                if not text or abs(float(text) - axis) > 1e-3:
                    problems.append(f"{uid} {key} does not match the raw SWC root")
                    break
            urls = str(row.get("soma_source_urls") or "")
            if row.get("coverage_status") in {"complete", "partial_neighbor"} and str(row["SampleID"]) not in urls:
                problems.append(f"{uid} soma source is not its own sample")
            central = str(row.get("context_central_slice") or "")
            if central and str(row["SampleID"]) not in central:
                problems.append(f"{uid} context slice is not its own sample")
    if len(pending) and "production_status" in pending.columns:
        rendered_pending = pending[pending["production_status"].isin(["soma_only", "partial", "error"])]
        for row in rendered_pending.to_dict(orient="records"):
            if row.get("production_status") == "error" and row.get("coverage_status") == "swc_unavailable":
                continue
            if row.get("production_status") not in {"soma_only", "partial"}:
                continue
            uid = row["UID"]
            swc_path = out_root / "swc_raw" / str(row["SampleID"]) / str(row["NeuronID"])
            if not swc_path.is_file():
                problems.append(f"{uid} raw SWC cache missing")
                continue
            try:
                root = _root_xyz(parse_swc(
                    swc_path.read_text(encoding="utf-8", errors="replace"), source=str(swc_path)))
            except (OSError, ValueError) as exc:
                problems.append(f"{uid} raw SWC unreadable: {exc}")
                continue
            for axis, key in zip(root, ("native_soma_x_um", "native_soma_y_um", "native_soma_z_um")):
                text = str(row.get(key) or "")
                if not text or abs(float(text) - axis) > 1e-3:
                    problems.append(f"{uid} {key} does not match the raw SWC root")
                    break
            urls = str(row.get("soma_source_urls") or "")
            if row.get("coverage_status") in {"complete", "partial_neighbor"} and str(row["SampleID"]) not in urls:
                problems.append(f"{uid} soma source is not its own sample")
    if len(frame) and "soma.nii.gz" in frame.columns:
        for row in frame.to_dict(orient="records"):
            for nii_name in ("soma.nii.gz", "context.nii.gz"):
                rel = str(row.get(nii_name) or "")
                if not rel:
                    continue
                sidecar = Path(str(out_root / rel) + ".json")
                if not sidecar.is_file():
                    problems.append(f"{row['UID']} missing sidecar {nii_name}.json")
                    continue
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    problems.append(f"{row['UID']} sidecar {nii_name}.json is not valid JSON")
                    continue
                if meta.get("nifti_axis_order") != "XYZ" or meta.get("source_array_axis_order") != "ZYX":
                    problems.append(f"{row['UID']} {nii_name} axis order is not source ZYX / NIfTI XYZ")
                if Path(str(meta.get("output_file") or "")).name != nii_name:
                    problems.append(f"{row['UID']} {nii_name} sidecar output_file is stale")
                native = [row.get("native_soma_x_um"), row.get("native_soma_y_um"), row.get("native_soma_z_um")]
                center = meta.get("center_xyz_um")
                if all(str(value) not in {"", "None", "nan"} for value in native) and isinstance(center, list):
                    if any(abs(float(left) - float(right)) > 1e-3 for left, right in zip(native, center)):
                        problems.append(f"{row['UID']} {nii_name} center is not the raw SWC root")
                acquisition = meta.get("acquisition") if isinstance(meta.get("acquisition"), dict) else {}
                if acquisition.get("complete") is False and meta.get("complete") is True:
                    problems.append(f"{row['UID']} {nii_name} complete contradicts the acquisition")
                if nii_name == "context.nii.gz" and row.get("context_coverage_status") in DERIVED_CONTEXT_STATES:
                    origin = np.asarray(meta.get("origin_xyz_um") or [], dtype=float)
                    spacing = np.asarray(meta.get("spacing_xyz_um") or [], dtype=float)
                    first = np.asarray(acquisition.get("source_first_index_xyz") or [], dtype=float)
                    stride = np.asarray(acquisition.get("source_stride_xyz") or [], dtype=float)
                    if any(values.shape != (3,) for values in (origin, spacing, first, stride)):
                        problems.append(f"{row['UID']} derived context is missing source-grid geometry")
                    elif (not np.isfinite(np.concatenate([origin, spacing, first, stride])).all()
                          or not np.allclose(first, np.rint(first)) or not np.allclose(stride, np.rint(stride))
                          or (stride < 1).any() or (first < 0).any()
                          or not np.allclose(origin, first * [.65, .65, 3])
                          or not np.allclose(spacing, stride * [.65, .65, 3])):
                        problems.append(f"{row['UID']} derived context is off the source sampling grid")
                    else:
                        image = nib.load(str(out_root / rel))
                        if (not np.allclose(image.affine[:3, 3], origin)
                                or not np.allclose(np.diag(image.affine)[:3], spacing)
                                or list(image.shape) != meta.get("shape_xyz")
                                or image.header.get_xyzt_units()[0] != "micron"):
                            problems.append(f"{row['UID']} derived NIfTI contradicts its source-grid sidecar")
                    if not acquisition.get("root_pixel_covered") or acquisition.get("sample_id") != str(row["SampleID"]):
                        problems.append(f"{row['UID']} derived context lacks its own-sample root coverage")
    # Identity completeness: every manifest UID is in the correction table once.
    if len(available) and int((available["production_status"] == "not_started").sum()):
        problems.append("available identities were not rendered")
    if len(frame) and "production_status" in frame.columns:
        unfinished = frame[frame["production_status"].isin(["not_started"])]
        if len(unfinished):
            problems.append(f"{len(unfinished)} identities are not started")
    if len(available) and int((available["production_status"] == "ok").sum()) == 0:
        problems.append("no successful available panels")
    if len(frame) and frame["UID"].duplicated().any():
        problems.append("duplicate UID in correction table")
    manifest = load_manifest(out_root) if (out_root / "manifest" / "identity_manifest.csv").is_file() else frame
    if len(frame) and set(manifest["UID"].astype(str)) != set(frame["UID"].astype(str)):
        problems.append("correction table does not match the identity manifest")
    coverage_counts = {}
    if len(available) and "coverage_status" in available.columns:
        coverage_counts = available["coverage_status"].value_counts().to_dict()
        bad_ok = available[(available["production_status"] == "ok") & (available["coverage_status"] != "complete")]
        if len(bad_ok):
            problems.append(f"{len(bad_ok)} ok rows are not coverage complete")
        for _, row in available.iterrows():
            missing_text = str(row.get("soma_missing") or "")
            missing_n = 0
            if missing_text not in {"", "[]", "null"}:
                try:
                    missing_n = len(json.loads(missing_text))
                except json.JSONDecodeError:
                    missing_n = -1
            if row["production_status"] == "ok" and missing_n != 0:
                problems.append(f"{row['UID']} ok row still lists missing neighbor cubes")
            if row.get("coverage_status") == "partial_neighbor" and missing_n <= 0:
                problems.append(f"{row['UID']} partial_neighbor has no missing cubes")
            if row.get("coverage_status") == "center_unavailable" and row["production_status"] != "partial":
                problems.append(f"{row['UID']} center gap is not partial")
            if row.get("coverage_status") == "swc_unavailable" and row["production_status"] != "error":
                problems.append(f"{row['UID']} missing SWC is not an error")
        present = set(available["UID"].astype(str))
        for uid in NEIGHBOR_GAP_UIDS:
            if uid not in present:
                continue
            row = available[available["UID"] == uid].iloc[0]
            if row["production_status"] != "partial" or row.get("coverage_status") != "partial_neighbor":
                problems.append(f"{uid} is {row['production_status']}/{row.get('coverage_status')}")
            sample_id, neuron_id = uid.split("|", 1)
            prov_path = out_root / sample_id / neuron_id.replace(".swc", "") / "provenance.json"
            if prov_path.is_file():
                saved_record = json.loads(prov_path.read_text(encoding="utf-8"))
                if saved_record.get("coverage_status") != "partial_neighbor":
                    problems.append(f"{uid} provenance coverage is {saved_record.get('coverage_status')}")
                if not saved_record.get("soma_missing"):
                    problems.append(f"{uid} provenance dropped the missing neighbor list")
    png_decoded = 0
    for sample in NEW_SAMPLES:
        folder = out_root / sample
        if not folder.is_dir():
            continue
        for png in folder.rglob("*.png"):
            try:
                with Image.open(png) as picture:
                    picture.load()
                png_decoded += 1
            except Exception as exc:
                problems.append(f"png decode failed {png.name}: {exc}")
    if len(available):
        for sample, subset in available.groupby("SampleID"):
            page_path = out_root / "gallery" / f"{sample}.html"
            if not page_path.is_file():
                problems.append(f"gallery page missing for {sample}")
                continue
            page = page_path.read_text(encoding="utf-8")
            for row in subset.to_dict(orient="records"):
                token = f"{row['SampleID']} {row['NeuronID']}"
                if token not in page:
                    problems.append(f"gallery missing {token}")
                for name in PANEL_FILES:
                    rel = row.get(name) or ""
                    if rel and not (out_root / rel).is_file():
                        problems.append(f"{row['UID']} missing file {rel}")
                    if rel.endswith(".png") and rel and f"../{rel}" not in page and row["production_status"] != "error":
                        if (out_root / rel).is_file() and f"../{rel}" not in page:
                            problems.append(f"gallery link missing {rel}")
    if len(pending) and "production_status" in pending.columns:
        for sample, subset in pending.groupby("SampleID"):
            rendered = subset[subset["production_status"].isin(["soma_only", "partial", "error"])]
            if rendered.empty:
                continue
            page_path = out_root / "gallery" / f"{sample}.html"
            if not page_path.is_file():
                problems.append(f"gallery page missing for {sample}")
                continue
            page = page_path.read_text(encoding="utf-8")
            for row in rendered.to_dict(orient="records"):
                token = f"{row['SampleID']} {row['NeuronID']}"
                if token not in page:
                    problems.append(f"gallery missing {token}")
                for name in SOMA_PACKAGE_FILES:
                    rel = row.get(name) or ""
                    if rel and not (out_root / rel).is_file():
                        problems.append(f"{row['UID']} missing file {rel}")
                    if rel.endswith(".png") and rel and f"../{rel}" not in page and row["production_status"] != "error":
                        problems.append(f"gallery link missing {rel}")
    index_path = out_root / "gallery" / "index.html"
    if index_path.is_file() and len(frame) and "SampleID" in frame.columns:
        index_text = index_path.read_text(encoding="utf-8")
        for sample in frame["SampleID"].astype(str).unique():
            if f"{sample}.html" not in index_text:
                problems.append(f"gallery index missing {sample}")
    ortho_prov = out_root / "252527" / "048" / "provenance.json"
    if ortho_prov.is_file():
        ortho_record = json.loads(ortho_prov.read_text(encoding="utf-8"))
        aspect = (ortho_record.get("ortho_aspect") or {}).get("XZ")
        if aspect is None or abs(float(aspect) - (3.0 / 0.65)) > 1e-6:
            problems.append(f"252527|048 XZ aspect is {aspect}")
        shape = ortho_record.get("soma_shape_zyx")
        if shape == [270, 1080, 1080]:
            spec = ortho_plane_specs(shape, ortho_record["soma_local_xyz"], ortho_record["ortho_spacing_xyz_um"])
            if abs(spec["XZ"]["width_um"] - 702) > 0.2 or abs(spec["XZ"]["height_um"] - 810) > 0.2:
                problems.append(
                    f"252527|048 XZ physical size {spec['XZ']['width_um']:.1f} x {spec['XZ']['height_um']:.1f} um")
        slab = ortho_record.get("soma_trace_slab") or {}
        if int(slab.get("sections") or 0) != TRACE_SLAB_SECTIONS or int(slab.get("edges_outside_slab") or 0) <= 0:
            problems.append(
                f"252527|048 soma trace slab sections {slab.get('sections')} "
                f"outside {slab.get('edges_outside_slab')}")
    queue_path = out_root / "manifest" / "component_queue.csv"
    if queue_path.is_file() and len(pending) and "overview_source_status" in pending.columns:
        queued = set(pd.read_csv(queue_path, dtype=str).fillna("")["UID"].astype(str))
        for row in pending.to_dict(orient="records"):
            state = str(row.get("context_coverage_status") or "")
            overview = str(row.get("overview_source_status") or "")
            if overview == "overview_pending" and state not in DERIVED_CONTEXT_STATES and row["UID"] not in queued:
                problems.append(f"{row['UID']} missing from the component queue")
    if len(available):
        summary_groups = (
            available.groupby(["SampleID", "review_category", "production_status"]).size()
            .reset_index(name="n").to_dict(orient="records"))
    else:
        summary_groups = []
    summary = {
        "identities": int(len(frame)),
        "available": int(len(available)),
        "pending": int(len(pending)),
        "ok": int((available["production_status"] == "ok").sum()) if len(available) else 0,
        "partial": int((available["production_status"] == "partial").sum()) if len(available) else 0,
        "error": int((available["production_status"] == "error").sum()) if len(available) else 0,
        "not_started": int((available["production_status"] == "not_started").sum()) if len(available) else 0,
        "coverage": {str(key): int(value) for key, value in coverage_counts.items()},
        "png_decoded": png_decoded,
        "canonical_hashes_unchanged": hash_ok,
        "pending_status": (
            {str(key): int(value) for key, value in pending["production_status"].value_counts().to_dict().items()}
            if len(pending) and "production_status" in pending.columns else {}),
        "pending_coverage": (
            {str(key): int(value) for key, value in pending["coverage_status"].value_counts().to_dict().items()}
            if len(pending) and "coverage_status" in pending.columns else {}),
        "inspected": inspected,
        "problems": problems,
        "by_sample_category_status": summary_groups,
    }
    (out_root / "manifest" / "verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        key: summary[key] for key in (
            "identities", "available", "pending", "ok", "partial", "error",
            "not_started", "coverage", "png_decoded", "canonical_hashes_unchanged")
    }, indent=2))
    if problems:
        print(f"[verify] {len(problems)} problems; first: {problems[:8]}")
        raise SystemExit(1)
    print("[verify] passed")
    return summary


def repair_crop_sidecars(out_root: Path = OUT_ROOT) -> int:
    """Correct export JSON in place. NIfTI bytes are not rewritten."""
    repaired = 0
    for sample in NEW_SAMPLES:
        root = out_root / sample
        if not root.is_dir():
            continue
        for prov_path in sorted(root.glob("*/provenance.json")):
            record = json.loads(prov_path.read_text(encoding="utf-8"))
            center = [
                record.get("native_soma_x_um"), record.get("native_soma_y_um"), record.get("native_soma_z_um")]
            if any(value is None or value == "" for value in center):
                continue
            for nii_name in ("context.nii.gz", "soma.nii.gz"):
                nii_path = prov_path.parent / nii_name
                side_path = Path(str(nii_path) + ".json")
                if not nii_path.is_file() or not side_path.is_file():
                    continue
                before = hashlib.sha256(nii_path.read_bytes()).hexdigest()
                meta = json.loads(side_path.read_text(encoding="utf-8"))
                acquisition = meta.get("acquisition") if isinstance(meta.get("acquisition"), dict) else {}
                meta["center_xyz_um"] = [float(value) for value in center]
                meta["source_array_axis_order"] = "ZYX"
                meta["nifti_axis_order"] = "XYZ"
                meta["array_axis_order"] = "XYZ"
                shape = meta.get("shape_zyx") or []
                if len(shape) == 3:
                    meta["shape_xyz"] = [int(shape[2]), int(shape[1]), int(shape[0])]
                if acquisition:
                    meta["complete"] = acquisition.get("complete")
                    meta["missing"] = acquisition.get("missing")
                    meta["loaded"] = acquisition.get("loaded")
                    meta["requested"] = acquisition.get("requested")
                    meta["source_paths"] = acquisition.get("source_paths")
                meta["output_file"] = str(nii_path)
                side_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
                after = hashlib.sha256(nii_path.read_bytes()).hexdigest()
                if before != after:
                    raise RuntimeError(f"NIfTI bytes changed while repairing {nii_path}")
                repaired += 1
    print(f"[sidecar] repaired {repaired}")
    return repaired


def context_plane_ids(acquisition, plane_count, origin, spacing):
    """Map every array slot, including missing sections, to a source plane."""
    requested = acquisition.get("source_plane_ids") or acquisition.get("requested")
    if requested and len(requested) == plane_count and all(np.isscalar(v) for v in requested):
        return list(requested)
    # Copied and derived contexts retain the nominal global 3 um Z lattice.
    first = float(origin[2]) / 3.0
    step = float(spacing[2]) / 3.0
    if not np.isclose(first, round(first)) or not np.isclose(step, round(step)):
        raise ValueError("Context Z geometry does not map to the declared source plane lattice")
    return [int(round(first + index * step)) for index in range(plane_count)]


def refresh_trace_previews(out_root: Path = OUT_ROOT, samples=None) -> int:
    """Redraw trace panels from persisted crops and raw SWCs. Other panels stay put."""
    import nibabel as nib
    refreshed = 0
    for sample in (samples or AVAILABLE_SAMPLES):
        root = out_root / sample
        if not root.is_dir():
            continue
        for prov_path in sorted(root.glob("*/provenance.json")):
            record = json.loads(prov_path.read_text(encoding="utf-8"))
            neuron = record.get("NeuronID") or (prov_path.parent.name + ".swc")
            swc_path = out_root / "swc_raw" / sample / str(neuron)
            if not swc_path.is_file():
                continue
            rows = parse_swc(swc_path.read_text(encoding="utf-8", errors="replace"), source=str(swc_path))
            native = [record["native_soma_x_um"], record["native_soma_y_um"], record["native_soma_z_um"]]
            changed = False
            for kind, nii_name, png_name, spacing_key in (
                ("context", "context.nii.gz", "context_trace.png", "declared_spacing_low_xyz_um"),
                ("soma", "soma.nii.gz", "soma_trace.png", "declared_spacing_high_xyz_um"),
            ):
                nii_path = prov_path.parent / nii_name
                if not nii_path.is_file():
                    continue
                image = nib.load(str(nii_path))
                volume = np.transpose(np.asanyarray(image.dataobj), (2, 1, 0))
                spacing = [float(value) for value in np.diag(image.affine)[:3]]
                origin = record.get(f"{kind}_origin_xyz_um")
                local = record.get(f"{kind}_local_xyz")
                if origin is None or local is None:
                    origin = [float(value) for value in image.affine[:3, 3]]
                    local = native_to_index(native, origin, spacing)
                if kind == "context":
                    side_path = Path(str(nii_path) + ".json")
                    side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.is_file() else {}
                    plane_ids = context_plane_ids(side.get("acquisition") or {}, volume.shape[0], origin, spacing)
                    record["context_plane_ids"] = plane_ids
                else:
                    plane_ids = [f"k{index}" for index in range(volume.shape[0])]
                title = f"{sample} {neuron} {kind} trace\nnominal spacing {spacing} um"
                scale = 500 if kind == "context" else 50
                slab = _save_trace_panel(
                    volume, local, spacing, rows, origin, prov_path.parent / png_name,
                    title, scale, plane_ids)
                record[f"{kind}_trace_slab"] = slab
                record[f"{kind}_trace_segments"] = slab["edges_in_slab"]
                refreshed += 1
                changed = True
                del volume
            if changed:
                prov_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
                print(f"[trace] {record.get('UID', neuron)}", flush=True)
    print(f"[trace] refreshed {refreshed}")
    return refreshed


def highres_context_cubes(native_xyz, field_um: float = 4000.0, depth_um: float = 30.0,
                          spacing=(0.65, 0.65, 3.0), block=(360, 360, 90)) -> list[tuple[int, int, int]]:
    """Cube indices covering a thin native field. This does not download anything."""
    halves = (field_um / 2.0, field_um / 2.0, depth_um / 2.0)
    ranges = []
    for axis, half in enumerate(halves):
        start = (float(native_xyz[axis]) - half) / float(spacing[axis]) / float(block[axis])
        stop = (float(native_xyz[axis]) + half) / float(spacing[axis]) / float(block[axis])
        ranges.append(range(int(np.floor(start)), int(np.floor(stop - 1e-9)) + 1))
    cubes = [(x, y, z) for z in ranges[2] for y in ranges[1] for x in ranges[0]]
    return cubes


# Share the tested source-grid and resource-bound implementation.
from fmost_derived_context import derive_highres_context  # noqa: E402


def derived_context_title(sample: str, neuron: str, report: dict, shape_zyx, spacing) -> str:
    """Distinguish requested tiles from acquisition outcomes in review panels."""
    extent = [round(shape_zyx[2-axis] * spacing[axis], 1) for axis in range(3)]
    return (
        f"{sample} {neuron} derived context\n"
        f"requested field {report['requested_field_um_xyz']} um; sampled voxel extent {extent} um\n"
        f"spacing {spacing} um | source coverage {100 * report['coverage_fraction']:.1f}%\n"
        f"requested cubes {report['cube_count']} | acquired {len(report.get('loaded') or [])} | "
        f"missing {len(report.get('missing') or [])} | invalid {len(report.get('invalid') or [])} | "
        f"unattempted {len(report.get('unattempted') or [])}\n"
        "own-sample high-resolution tiles, orientation unverified"
    )


def pilot_derived_context(sample: str, neuron: str, out_root: Path = OUT_ROOT,
                          max_cubes: int = 400, field_um: float = 4000.0,
                          depth_um: float = 30.0, factor: int = 8,
                          max_seconds: float = 900, cube_download=None,
                          max_source_bytes: int = 12 * 1024**3) -> dict:
    """One own-sample thin context from high-resolution cubes. Does not borrow another sample."""
    from Visual_toolkit import Visual_toolkit, HTTP_HOST, HTTP_PATH
    import nibabel as nib
    neuron = canon_neuron_id(neuron)
    manifest = load_manifest(out_root)
    match = manifest[(manifest["SampleID"] == str(sample)) & (manifest["NeuronID"] == neuron)]
    if match.empty:
        raise RuntimeError(f"{sample}|{neuron} is not in the identity manifest")
    folder = out_root / str(sample) / neuron.replace(".swc", "")
    prov_path = folder / "provenance.json"
    existing = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    # Reject before any source request. Copied sections have their own acquisition.
    if str(sample) in AVAILABLE_SAMPLES or (
            (folder / "context.nii.gz").exists() and not existing.get("derived_context")):
        raise ValueError("Derived pilot cannot replace an existing copied or unclassified context")
    project_id = str(match.iloc[0]["project_id"])
    rows, digest, url, source = fetch_raw_swc(sample, neuron, project_id, out_root)
    native = _root_xyz(rows)
    folder = out_root / str(sample) / neuron.replace(".swc", "")
    prov_path = folder / "provenance.json"
    record = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {
        "SampleID": str(sample), "NeuronID": neuron, "UID": f"{sample}|{neuron}",
        "overview_source_status": "overview_pending",
    }
    if (folder / "soma.nii.gz").is_file() and record.get("swc_sha256") != digest:
        raise RuntimeError("Existing soma package has different or unknown raw SWC lineage; rebuild the package together")
    toolkit = Visual_toolkit(str(sample), cache_dir=str(out_root / "cache"), low_res_slice_cache_limit=0)
    try:
        canvas, report = derive_highres_context(
            native, (lambda *index: cube_download(toolkit, *index)) if cube_download else toolkit._download_http_block,
            max_cubes=max_cubes, max_source_bytes=max_source_bytes,
            field_um=field_um, depth_um=depth_um, factor=factor, max_seconds=max_seconds)
    finally:
        toolkit.close()
    report.update({
        "SampleID": str(sample), "NeuronID": neuron, "UID": f"{sample}|{neuron}",
        "native_soma_xyz_um": native, "swc_sha256": digest, "swc_url": url, "swc_source": source,
        "nmt_used": False, "borrowed_sample": False,
    })
    sources = []
    for x, y, z in report.get("loaded") or []:
        path = Path(toolkit.cache_http_dir) / str(z) / f"{x}_{y}_{z}.tif"
        sources.append({
            "cube_xyz": [x, y, z], "path": str(path),
            "url": f"{HTTP_HOST}/{HTTP_PATH}/{sample}/cube/{z}/{x}_{y}_{z}.tif",
            "sha256": sha256_file(path) if path.is_file() else None,
            "file_bytes": path.stat().st_size if path.is_file() else None,
        })
    report["source_paths"] = sources
    report["source_errors"] = [{"cube_xyz": list(index), "reason": reason}
                               for index, reason in toolkit._block_errors.items()]
    report_path = out_root / "manifest" / f"derived_context_pilot_{sample}_{neuron.replace('.swc', '')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if canvas is None:
        # A failed replacement must not invalidate or relabel existing artifacts.
        attempt_path = (report_path.with_name(report_path.name.replace("_pilot_", "_attempt_"))
                        if (folder / "context.nii.gz").exists() else report_path)
        attempt_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if not (folder / "context.nii.gz").exists() and prov_path.is_file():
            record["context_coverage_status"] = report["status"]
            record["derived_context"] = report
            record["context_pending_reason"] = str(report.get("reason") or "") + " | " + " | ".join(
                str(error.get("reason")) for error in report.get("source_errors") or [])
            prov_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"[context] not run {sample}|{neuron}: {report.get('reason')}")
        return report
    folder.mkdir(parents=True, exist_ok=True)
    spacing = report["derived_spacing_xyz_um"]
    origin = report["origin_xyz_um"]
    affine = native_affine(origin, spacing)
    image = nib.Nifti1Image(np.transpose(canvas, (2, 1, 0)), affine)
    image.header.set_xyzt_units("micron")
    image.header["descrip"] = b"Derived high-res thin context; orientation unverified"
    nii_path = folder / "context.nii.gz"
    nib.save(image, str(nii_path))
    sidecar = {
        "center_xyz_um": [float(value) for value in native],
        "origin_xyz_um": [float(value) for value in origin],
        "source_array_axis_order": "ZYX",
        "nifti_axis_order": "XYZ",
        "array_axis_order": "XYZ",
        "shape_zyx": list(canvas.shape),
        "shape_xyz": [int(canvas.shape[2]), int(canvas.shape[1]), int(canvas.shape[0])],
        "complete": report.get("status") == "derived_highres",
        "missing": report.get("missing"),
        "output_file": str(nii_path),
        "spacing_xyz_um": spacing,
        "derived_from": "own-sample high-resolution cubes",
        "acquisition": {
            "sample_id": str(sample), "coordinate_units": "um", "array_axis_order": "ZYX",
            "origin_xyz_um": origin, "spacing_xyz_um": spacing, "shape_zyx": list(canvas.shape),
            "complete": report.get("status") == "derived_highres",
            "missing": report.get("missing"),
            "invalid": report.get("invalid"), "unattempted": report.get("unattempted"),
            "requested": report["requested"], "loaded": report.get("loaded"),
            "source_plane_ids": report["source_plane_ids"],
            "sampling": report["sampling"], "source_stride_xyz": report["source_stride_xyz"],
            "source_first_index_xyz": report["source_first_index_xyz"], "source_paths": sources,
            "root_pixel_covered": report["root_pixel_covered"],
            "loaded_count": len(report.get("loaded") or []),
        },
    }
    Path(str(nii_path) + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    local = native_to_index(native, origin, spacing)
    z = int(np.floor(local[2]))
    plane = canvas[z]
    stats = _save_gray(plane, folder / "context_unmarked.png")
    title = derived_context_title(sample, neuron, report, canvas.shape, spacing)
    if _inside(local, canvas.shape):
        _draw_marked(plane, folder / "context_marked.png", (local[0], local[1]), spacing[:2], title, scale_um=500)
    trace = _save_trace_panel(canvas, local, spacing, rows, origin,
                              folder / "context_trace.png", title, 500, report["source_plane_ids"])
    stack_ids = _save_stack(canvas, z, report["source_plane_ids"], folder / "context_stack.png")
    _save_ortho(canvas, local, folder / "context_ortho.png", spacing)
    write_derived_coverage(folder, report, native)
    record["context_coverage_status"] = report["status"]
    record["context_origin_xyz_um"] = origin
    record["context_shape_zyx"] = list(canvas.shape)
    record.update({
        "native_soma_x_um": float(native[0]), "native_soma_y_um": float(native[1]),
        "native_soma_z_um": float(native[2]), "context_local_xyz": [float(value) for value in local],
        "context_plane_ids": report["source_plane_ids"], "context_central_z_index": report["source_plane_ids"][z],
        "context_plane_stats": stats, "context_stack_z": stack_ids,
        "context_trace_slab": trace, "context_trace_segments": trace["edges_in_slab"],
        "soma_inside_context": bool(_inside(local, canvas.shape)),
        "context_complete": report["complete"],
        "context_requested_um_xyz": report["requested_field_um_xyz"],
        "section_locator_status": "unavailable: derived local field is not a full-section image",
        "context_pending_reason": "partial source coverage" if not report["complete"] else "",
    })
    record["derived_context"] = {
        key: value for key, value in report.items() if key not in {"loaded"}
    }
    record["derived_context"]["loaded_count"] = len(report.get("loaded") or [])
    record["overview_source_status"] = record.get("overview_source_status") or "overview_pending"
    prov_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[context] {report['status']} {sample}|{neuron} cubes {report['cube_count']}", flush=True)
    return report


def write_derived_coverage(folder: Path, report: dict, native_xyz) -> None:
    """Show acquisition gaps separately from acquired zero-intensity pixels."""
    import nibabel as nib
    from fmost_context_resources import coverage_mask
    mask = coverage_mask(report)
    spacing, origin = report["derived_spacing_xyz_um"], report["origin_xyz_um"]
    local = native_to_index(native_xyz, origin, spacing)
    z = int(np.floor(local[2]))
    image = nib.Nifti1Image(mask.transpose(2, 1, 0), native_affine(origin, spacing))
    image.header.set_xyzt_units("micron")
    image.header["descrip"] = b"Source coverage: 1 acquired, 0 unavailable; not intensity"
    path = folder / "context_coverage.nii.gz"
    nib.save(image, str(path))
    Path(str(path) + ".json").write_text(json.dumps({
        "output_file": str(path), "source_intensity_file": str(folder / "context.nii.gz"),
        "source_array_axis_order": "ZYX", "nifti_axis_order": "XYZ",
        "spacing_xyz_um": spacing, "origin_xyz_um": origin,
        "values": {"1": "source acquired", "0": "source unavailable; not anatomical absence"},
        "loaded_cube_xyz": report["loaded"], "source_first_index_xyz": report["source_first_index_xyz"],
        "source_stride_xyz": report["source_stride_xyz"],
    }, indent=2), encoding="utf-8")
    picture = Image.fromarray(mask[z] * 255).convert("RGB")
    captioned = Image.new("RGB", (picture.width, picture.height + 60))
    captioned.paste(picture, (0, 60))
    draw = ImageDraw.Draw(captioned)
    draw.text((8, 5), "SOURCE COVERAGE: white = acquired; black = unavailable", fill="white")
    draw.text((8, 22), "Intensity and anatomical absence cannot be inferred from this mask.", fill="white")
    x, y = float(local[0]), float(local[1]) + 60
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="cyan")
    captioned.save(folder / "context_coverage.png")


def _smoke_targets(manifest: pd.DataFrame) -> dict:
    hints = {}
    probe = PORTAL_DIR / "native_source_probe.json"
    if probe.is_file():
        for item in json.loads(probe.read_text(encoding="utf-8")):
            if item.get("sample") in AVAILABLE_SAMPLES:
                hints.setdefault(item["sample"], canon_neuron_id(item.get("neuron")))
    targets = {}
    for sample in AVAILABLE_SAMPLES:
        subset = manifest[manifest["SampleID"] == sample]
        hint = hints.get(sample)
        if hint and (subset["NeuronID"] == hint).any():
            targets[sample] = {hint}
            continue
        insula = subset[subset["review_category"] == "INS"]
        pick = (insula if len(insula) else subset).iloc[0]["NeuronID"]
        targets[sample] = {canon_neuron_id(pick)}
    return targets


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Native fMOST review manifest and panels")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--unresolved-report", type=Path)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--force", action="store_true")
    render = sub.add_parser("render")
    render.add_argument("--sample", action="append")
    render.add_argument("--force", action="store_true")
    soma = sub.add_parser("render-soma")
    soma.add_argument("--sample", action="append")
    soma.add_argument("--force", action="store_true")
    sub.add_parser("repair-sidecars")
    sub.add_parser("refresh-traces")
    pilot = sub.add_parser("pilot-context")
    pilot.add_argument("--sample", required=True)
    pilot.add_argument("--neuron", required=True)
    pilot.add_argument("--max-cubes", type=int, default=400)
    sub.add_parser("gallery")
    sub.add_parser("reconcile-coverage")
    sub.add_parser("refresh-ortho")
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        build_manifest(unresolved_report=args.unresolved_report)
    elif args.command == "smoke":
        manifest = load_manifest()
        render_available(neurons=_smoke_targets(manifest), force=args.force)
        write_gallery()
    elif args.command == "render":
        render_available(samples=args.sample, force=args.force)
        write_gallery()
    elif args.command == "render-soma":
        render_available(samples=args.sample, force=args.force, mode="soma")
        write_gallery()
    elif args.command == "repair-sidecars":
        repair_crop_sidecars()
    elif args.command == "refresh-traces":
        refresh_trace_previews()
    elif args.command == "pilot-context":
        pilot_derived_context(args.sample, args.neuron, max_cubes=args.max_cubes)
    elif args.command == "gallery":
        write_gallery()
    elif args.command == "reconcile-coverage":
        reconcile_coverage_records()
    elif args.command == "refresh-ortho":
        refresh_ortho_panels()
    elif args.command == "verify":
        verify_outputs()


if __name__ == "__main__":
    main()
