"""
utils.py - Shared helpers: parsing, debug snapshots, file I/O.
"""

import ast
import os
from pathlib import Path
from numbers import Real

import numpy as np
import pandas as pd

from region_labels import is_explicit_unknown_label, normalize_region_label


def terminal_target_status(value) -> str:
    """Reporting status of one target, without accepting/rejecting atlas anatomy.

    ``known`` means a nonempty literal label without an explicit sentinel. It
    does not verify atlas membership or biological terminal identity. Exact
    control tokens avoid rejecting anatomical names containing ``Unknown``.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return "absent"
    if not isinstance(value, str):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "absent"
        return "invalid"
    token = normalize_region_label(value)
    # Subcortical prefixes are stripped only to recognize control tokens;
    # literal anatomical labels (including SL_Pi versus CL_Pi) are unchanged.
    if token.startswith(("SL_", "SR_")):
        token = token[3:]
    if token in {"", "NONE", "NULL", "NAN"}:
        return "absent"
    if token == "OUT_OF_BOUNDS":
        return "outside"
    if token in {"_UNMAPPED", "UNMAPPED"}:
        return "unmapped"
    if is_explicit_unknown_label(token):
        return "unknown"
    return "known"


def is_known_terminal_target(value) -> bool:
    """True for a usable literal target label under the reporting contract."""
    return terminal_target_status(value) == "known"


def terminal_target_region_counts(region_lists, include_unresolved=False):
    """Count parsed entries without explode's phantom entry for empty lists.

    Retain literal string labels. Absent/invalid values receive display-only
    placeholders when all entries are requested; they never become anatomy.
    """
    entries = []
    for regions in region_lists:
        for value in regions:
            status = terminal_target_status(value)
            if status == "known" or include_unresolved:
                label = value if isinstance(value, str) and value.strip() else f"<{status} target>"
                entries.append(label)
    return pd.Series(entries, dtype=object).value_counts()


def parse_terminal_regions(x) -> list:
    """Parse Terminal_Regions from various formats to unique-order list."""
    if isinstance(x, (list, tuple)):
        return _dedupe(x)
    if isinstance(x, str):
        try:
            if x.startswith("[") and x.endswith("]"):
                return _dedupe(ast.literal_eval(x))
            if "," in x:
                parts = [p.strip().strip("'\"") for p in x.split(",")]
                return _dedupe(parts)
            return [x.strip().strip("'\"")]
        except Exception:
            return [x]
    return []


def _dedupe(seq) -> list:
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def save_debug_snapshot(
    voxel_coords,
    neuron_name: str,
    template_img,
    point_type: str,
    folder: str = "../resource/debug_outliers",
):
    if template_img is None:
        return
    os.makedirs(folder, exist_ok=True)

    import nibabel.affines
    from nilearn import plotting

    world_coords = nibabel.affines.apply_affine(template_img.affine, voxel_coords)
    try:
        display = plotting.plot_anat(
            template_img,
            cut_coords=world_coords,
            draw_cross=True,
            title=f"Outlier {point_type}: {neuron_name} -> {voxel_coords}",
        )
        fname = (
            f"{neuron_name}_{point_type}_"
            f"{int(voxel_coords[0])}_{int(voxel_coords[1])}_{int(voxel_coords[2])}.png"
        )
        full_path = os.path.join(folder, fname)
        display.savefig(full_path)
        display.close()
        print(f"     [DEBUG] Snapshot saved: {full_path}")
    except Exception as e:
        print(f"     [DEBUG] Failed: {e}")


def parse_projection_lengths(value) -> dict:
    """Read a literal length mapping; missing/invalid values are not zeros."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Projection lengths must be a literal dictionary") from exc
    if not isinstance(value, dict):
        raise ValueError("Missing or invalid projection-length dictionary")
    for region, length in value.items():
        if not isinstance(region, str) or not region.strip():
            raise ValueError("Projection targets must be nonempty literal labels")
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, Real) or not np.isfinite(length) or length < 0:
            raise ValueError(f"Projection length must be finite and nonnegative: {region}")
    return value.copy()


def normalize_region_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Decode saved cells before numeric consumers can silently discard them."""
    df = df.copy()
    for column in df.columns:
        if column in {"Region_projection_length", "Ipsilateral_Projection_Length",
                      "Contralateral_Projection_Length", "Unknown_Laterality_Projection_Length"} or column.startswith("Region_Projection_Length_"):
            df[column] = df[column].apply(parse_projection_lengths)
        elif column == "Terminal_Regions" or column.startswith("Proj_"):
            df[column] = df[column].apply(parse_terminal_regions)
    return df


def projection_column_names(regions, atlas_regions=()) -> dict:
    """Remove hemisphere while preserving colliding cortical/subcortical names.

    For example, cortical parainsula and subcortical pineal gland cannot both
    become ``Pi``. Use all declared atlas labels when available so naming is
    stable even when only one homonym occurs in the selected cohort.
    """
    regions = list(regions)
    domains = {}
    for region in regions + list(atlas_regions):
        if isinstance(region, str) and region.startswith(("CL_", "CR_", "SL_", "SR_")):
            domains.setdefault(region[3:], set()).add(region[0])
    names = {}
    for region in regions:
        if region.startswith(("CL_", "CR_", "SL_", "SR_")):
            base = region[3:]
            names[region] = f"{region[0]}_{base}" if len(domains[base]) > 1 else base
        else:
            names[region] = region
    reverse = {}
    for region, column in names.items():
        # Opposite hemispheres of one anatomical region intentionally share a
        # column. Generated names must not collide with a literal input name.
        identity = (region[0], region[3:]) if region.startswith(("CL_", "CR_", "SL_", "SR_")) else (None, region)
        if column in reverse and reverse[column] != identity:
            raise ValueError(f"Ambiguous projection column name: {column}")
        reverse[column] = identity
    return names


def _validate_neuron_identity(frame, allow_repeated_records=False):
    keys = (["SampleID", "NeuronID"] if "SampleID" in frame else
            ["NeuronUID"] if "NeuronUID" in frame else ["NeuronID"])
    if any(key not in frame for key in keys) or frame[keys].isna().any().any():
        raise ValueError("Missing neuron identity")
    if "NeuronID" not in frame or not frame.NeuronID.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("NeuronID must be present literal text")
    if any(frame[key].astype(str).str.strip().eq("").any() for key in keys) or (not allow_repeated_records and frame.duplicated(keys).any()):
        raise ValueError("Missing or duplicate exact neuron identity")
    if "NeuronUID" in frame:
        if not frame.NeuronUID.map(lambda value: isinstance(value, str) and "::" in value and bool(value.split("::", 1)[0].strip())).all():
            raise ValueError("NeuronUID must encode SampleID::NeuronID")
        if not frame.NeuronUID.str.split("::", n=1).str[1].eq(frame.NeuronID).all():
            raise ValueError("NeuronUID disagrees with NeuronID")
        if "SampleID" in frame:
            tokens = frame.SampleID.map(lambda value: str(int(value)) if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and float(value).is_integer() else str(value).strip())
            if not frame.NeuronUID.eq(tokens + "::" + frame.NeuronID).all():
                raise ValueError("NeuronUID disagrees with SampleID/NeuronID")
        if not allow_repeated_records and frame.NeuronUID.duplicated().any():
            raise ValueError("Duplicate NeuronUID")
    return keys


def neuron_metadata(frame):
    """Preserve exact subject identity when bare neuron IDs are not unique."""
    _validate_neuron_identity(frame)
    columns = ["NeuronID", "Neuron_Type"]
    if "SampleID" in frame and (frame.SampleID.nunique() > 1 or "NeuronUID" in frame):
        columns.insert(0, "SampleID")
    if "NeuronUID" in frame:
        columns.insert(0, "NeuronUID")
    result = frame.reindex(columns=columns).reset_index(drop=True)
    if "Neuron_Type" not in frame:
        result["Neuron_Type"] = ""
    return result


def _join_keys(summary, sheet, name):
    _validate_neuron_identity(summary)
    _validate_neuron_identity(sheet, allow_repeated_records=True)
    if "SampleID" in summary and "SampleID" in sheet:
        keys = ["SampleID", "NeuronID"]
    elif "NeuronUID" in summary and "NeuronUID" in sheet:
        keys = ["NeuronUID"]
    else:
        keys = ["NeuronID"]
    if any(key not in sheet or key not in summary for key in keys):
        raise ValueError(f"{name} lacks exact neuron identity")
    if summary[keys].isna().any().any() or sheet[keys].isna().any().any() or summary.duplicated(keys).any():
        raise ValueError(f"{name} has missing or ambiguous neuron identities")
    return keys


def _aligned_sheet(summary, sheet, name):
    keys = _join_keys(summary, sheet, name)
    if sheet.duplicated(keys).any():
        raise ValueError(f"{name} has missing or ambiguous neuron identities")
    left = summary[keys].apply(tuple, axis=1).tolist()
    right = sheet[keys].apply(tuple, axis=1).tolist()
    if set(left) != set(right):
        raise ValueError(f"{name} neuron membership differs from Summary")
    positions = {key: index for index, key in enumerate(right)}
    aligned = sheet.iloc[[positions[key] for key in left]].reset_index(drop=True)
    if "Neuron_Type" in summary and "Neuron_Type" in aligned and not summary.Neuron_Type.reset_index(drop=True).fillna("").equals(aligned.Neuron_Type.fillna("")):
        raise ValueError(f"{name} neuron types differ from Summary")
    return aligned


def _long_sheet_records(summary, sheet, name, record_columns):
    """Join repeated records without conflating matching IDs across animals."""
    if sheet.empty:
        return [[] for _ in range(len(summary))]
    keys = _join_keys(summary, sheet, name)
    if not set(record_columns) <= set(sheet):
        raise ValueError(f"{name} lacks required record columns")
    allowed = set(summary[keys].apply(tuple, axis=1))
    grouped = {}
    for _, row in sheet.iterrows():
        key = tuple(row[column] for column in keys)
        if key not in allowed:
            raise ValueError(f"{name} contains a neuron absent from Summary")
        grouped.setdefault(key, []).append({column: row[column] for column in record_columns})
    return [grouped.get(key, []) for key in summary[keys].apply(tuple, axis=1)]


def load_processed_df(path) -> pd.DataFrame:
    """Load literal-cell tables or reconstruct a lossless multi-sheet export.

    Absolute projection labels are required: hemisphere-stripped split sheets
    cannot reconstruct anatomical identity or unknown-side lengths.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return normalize_region_dataframe(pd.read_csv(path))
    if path.suffix.lower() != ".xlsx":
        raise ValueError("File must be .xlsx or .csv")
    sheets = pd.read_excel(path, sheet_name=None)
    summary = sheets.get("Summary", next(iter(sheets.values()))).copy()
    _validate_neuron_identity(summary)
    if not any(column in summary for column in ("Region_projection_length", "Region_Projection_Length_finest")) and "Projection_Length_all" not in sheets and any(name.startswith("Projection_Length") for name in sheets):
        raise ValueError("Split-only workbook cannot recover absolute projection lengths; use its source analysis table")
    if "Region_projection_length" not in summary and "Projection_Length_all" in sheets:
        lengths = _aligned_sheet(summary, sheets["Projection_Length_all"], "Projection_Length_all")
        metadata = {"SampleID", "NeuronID", "NeuronUID", "Neuron_Type", "Length_Unit"}
        targets = [column for column in lengths if column not in metadata]
        summary["Region_projection_length"] = [parse_projection_lengths({target: row[target] for target in targets if row[target] != 0})
                                                for _, row in lengths.iterrows()]
    if "Terminal_Regions" not in summary and "Terminal_Sites" in sheets:
        records = _long_sheet_records(summary, sheets["Terminal_Sites"], "Terminal_Sites", ["Terminal_Region"])
        summary["Terminal_Regions"] = [[record["Terminal_Region"] for record in row] for row in records]
        if "Terminal_Count" in summary and not summary.Terminal_Count.eq(summary.Terminal_Regions.apply(len)).all():
            raise ValueError("Terminal_Sites target counts differ from Summary")
    if "Outlier_Details" not in summary and "Outliers" in sheets:
        records = _long_sheet_records(summary, sheets["Outliers"], "Outliers", ["Outlier_Type", "Region", "Voxel_X", "Voxel_Y", "Voxel_Z"])
        summary["Outlier_Details"] = [[{"type": record["Outlier_Type"], "region": record["Region"],
                                        "coords": tuple(record[column] for column in ("Voxel_X", "Voxel_Y", "Voxel_Z"))}
                                       for record in row] for row in records]
        if "Outlier_Count" in summary and not summary.Outlier_Count.eq(summary.Outlier_Details.apply(len)).all():
            raise ValueError("Outliers counts differ from Summary")
    if "Soma_Hierarchy" in sheets and not sheets["Soma_Hierarchy"].empty:
        hierarchy = _aligned_sheet(summary, sheets["Soma_Hierarchy"], "Soma_Hierarchy")
        for column in hierarchy:
            if column.startswith("L") and column[1:].isdigit():
                summary["Soma_Level_" + column[1:]] = hierarchy[column].values
    return normalize_region_dataframe(summary)
