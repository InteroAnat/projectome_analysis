"""Testing ground for ION pyswcloader against the Liu 2026 NeuroM bridge.

Version: 0.1.0 (local/unreleased)
Last updated: 2026-10-18
Change history: inline below
Status: evaluation only. This does not replace NeuroM or the VEN bridge.

The operator reads SWC files and writes a JSON receipt. It does not edit
source reconstructions, annotation volumes, or cohort tables.

Change history
--------------
0.1.0 — 2026-10-18 (local/unreleased)
    Added a dry-run and single-case comparison of pyswcloader 1.1.1 with the
    existing NeuroM dendritic features used for Liu et al. 2026. The package
    default Allen mouse box is reported separately from cable length.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path


# Package defaults copied from pyswcloader.swc.swc_preprocess. They describe
# the Allen mouse CCF in micrometres (13.2 x 8.0 x 11.4 mm), not a macaque atlas.
ALLEN_DIMENSION_UM = (13200.0, 8000.0, 11400.0)
ALLEN_Z_MIDLINE_UM = ALLEN_DIMENSION_UM[2] / 2.0
PINNED_VERSION = "1.1.1"
LIU_R_LOCAL_UM = 800.0
R_LOCAL_MAX_UM = 5000.0
MODES = ("inspect", "compare", "preprocess-risk", "distance")

SETTINGS = {
    "package": "pyswcloader",
    "pinned_version": PINNED_VERSION,
    "reference_tool": "NeuroM",
    "reference_study": "Liu et al. 2026 Nature Cell Biology, DOI 10.1038/s41556-026-02009-4",
    "reference_bridge": "external/ven_validation/scripts/morph_features.py",
    "r_local_um": LIU_R_LOCAL_UM,
    "allen_dimension_um": list(ALLEN_DIMENSION_UM),
    "allen_flip_z_default": True,
    "modes": list(MODES),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    """Repo-relative posix path when the file lives in this checkout."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(resolved)


def ground_root() -> Path:
    return repo_root() / "work" / "pyswcloader_ground"


def vendor_dir() -> Path:
    return ground_root() / "vendor"


def receipts_dir() -> Path:
    return ground_root() / "receipts"


def ensure_import_path() -> None:
    """Prefer the pinned wheel unpacked under the testing ground."""
    vendor = str(vendor_dir())
    if vendor_dir().is_dir() and vendor not in sys.path:
        sys.path.insert(0, vendor)


def installed_version() -> str:
    ensure_import_path()
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pyswcloader")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "pyswcloader is not installed. From the repo root run: "
            "python -m pip install --target work/pyswcloader_ground/vendor "
            "--no-deps pyswcloader==1.1.1 treelib rdp"
        ) from exc


def load_swc_module():
    ensure_import_path()
    found = installed_version()
    if found != PINNED_VERSION:
        raise RuntimeError(
            f"pyswcloader {found} is loaded; this ground is pinned to {PINNED_VERSION}"
        )
    from pyswcloader import swc

    return swc


def load_projection_module():
    load_swc_module()
    from pyswcloader import projection

    return projection


def neurom_status() -> dict:
    """Availability of the Liu-bridge backend. Import errors stay in the receipt."""
    try:
        import neurom as nm
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    scripts = repo_root() / "external" / "ven_validation" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        from morph_features import extract_features_neurom
    except Exception as exc:
        return {
            "available": False,
            "version": getattr(nm, "__version__", None),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": getattr(nm, "__version__", None),
        "extractor": "extract_features_neurom",
        "callable": extract_features_neurom,
    }


def parse_rows(path: Path) -> list[tuple]:
    """Use this repo's structural SWC parser so cable totals follow node ids."""
    scripts = repo_root() / "main_scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from swc_validation import parse_swc

    return parse_swc(path.read_text(encoding="utf-8", errors="replace"), source=str(path))


def cable_summary(rows: list[tuple]) -> dict:
    """Length by child compartment, plus the pyswcloader non-dendrite rule.

    pyswcloader.swc.total_length adds an edge only when both endpoints have a
    type outside {3, 4}. Types 3 and 4 are basal and apical dendrite in the
    SWC specification, which is the compartment Liu et al. measured.
    """
    by_id = {row[0]: row for row in rows}
    summary = {
        "n_nodes": len(rows),
        "axon": 0.0,
        "basal": 0.0,
        "apical": 0.0,
        "other": 0.0,
        "pyswc_nondendrite_rule": 0.0,
    }
    for row in rows:
        parent_id = row[6]
        if parent_id == -1:
            continue
        parent = by_id[parent_id]
        distance = math.dist(row[2:5], parent[2:5])
        summary[{2: "axon", 3: "basal", 4: "apical"}.get(row[1], "other")] += distance
        if row[1] not in (3, 4) and parent[1] not in (3, 4):
            summary["pyswc_nondendrite_rule"] += distance
    return summary


def soma_xyz(rows: list[tuple]) -> list[float]:
    roots = [row for row in rows if row[6] == -1]
    if len(roots) != 1:
        raise ValueError(f"expected one SWC root, found {len(roots)}")
    return [float(value) for value in roots[0][2:5]]


def allen_risk(swc, path: Path, flip: bool, dimension: tuple[float, float, float]) -> dict:
    """Run the package preprocess in memory and describe the coordinate change."""
    raw = swc.read_swc(str(path))
    first = [float(raw.loc[1, axis]) for axis in ("x", "y", "z")]
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages):
        processed = swc.swc_preprocess(
            str(path),
            flip=flip,
            dimension=list(dimension),
            check_validity=True,
            save=False,
        )
    after = [float(processed.loc[1, axis]) for axis in ("x", "y", "z")]
    clipped = []
    for axis, limit, before_value, after_value in zip("xyz", dimension, first, after):
        if before_value > limit or abs(after_value - before_value) > 1e-6:
            if axis != "z" or abs(after_value - (dimension[2] - before_value)) > 1e-4:
                clipped.append(axis)
    flipped = abs(after[2] - (dimension[2] - first[2])) <= 1e-4 and abs(after[2] - first[2]) > 1e-4
    return {
        "first_row_xyz_um": first,
        "preprocessed_first_row_xyz_um": after,
        "z_flipped": bool(flipped),
        "axes_changed": clipped,
        "negative_coordinate": any(value < 0 for value in after),
        "dimension_um": list(dimension),
        "flip_requested": bool(flip),
        "package_messages": [line for line in messages.getvalue().splitlines() if line.strip()],
    }


def neurom_features(path: Path, r_local: float) -> dict:
    status = neurom_status()
    if not status["available"]:
        return {"available": False, "error": status.get("error")}
    features = status["callable"](path, r_local=r_local, source="pyswcloader-ground")
    keep = (
        "backend",
        "r_local_um",
        "complete_morph_labels",
        "n_dendrite_neurites",
        "total_dendrite_length",
        "basal_length",
        "apical_length",
        "basal_max_path",
        "apical_max_path",
        "max_radial_distance_dend",
        "n_bifurcations_dend",
        "basal_gt_500",
    )
    out = {key: features[key] for key in keep}
    out["available"] = True
    out["version"] = status["version"]
    return out


def compare_case(
    path: Path,
    r_local: float = LIU_R_LOCAL_UM,
    flip: bool = True,
    include_neurom: bool = True,
) -> dict:
    """Compare one SWC on the Liu dendritic question and the pyswcloader length rule."""
    path = Path(path)
    swc = load_swc_module()
    rows = parse_rows(path)
    cables = cable_summary(rows)
    package_length = float(swc.total_length(str(path)))
    neurom = neurom_features(path, r_local) if include_neurom else {"available": False, "skipped": True}
    risk = allen_risk(swc, path, flip=flip, dimension=ALLEN_DIMENSION_UM)
    dendrite = cables["basal"] + cables["apical"]
    return {
        "path": display_path(path),
        "soma_xyz_um": soma_xyz(rows),
        "cables_um": cables,
        "pyswcloader_total_length_um": package_length,
        "pyswcloader_matches_nondendrite_rule": math.isclose(
            package_length, cables["pyswc_nondendrite_rule"], rel_tol=1e-6, abs_tol=1e-4
        ),
        "neurom": neurom,
        "allen_preprocess": risk,
        "questions": {
            "pyswcloader_total_length": (
                "Sum of edges whose child and parent types are both outside "
                "SWC types 3 and 4. This keeps axon and soma links."
            ),
            "neurom_liu_bridge": (
                f"Dendritic cable and path inside a {r_local:g} µm soma ball "
                "after the axon is removed, using NeuroM on a MorphIO crop."
            ),
            "same_measurement": False,
            "reason": (
                f"pyswcloader total_length is {package_length:.3f} µm. "
                f"Full dendritic cable is {dendrite:.3f} µm."
            ),
        },
    }


def synthetic_bipolar(path: Path) -> Path:
    """Liu-like bipolar dendrites plus a long axon, soma at the origin.

    Basal type 3 runs 600 µm, apical type 4 runs 400 µm, and axon type 2 runs
    5000 µm. The 800 µm crop used by the VEN bridge keeps the dendrites and
    drops the far axon.
    """
    text = "\n".join(
        [
            "1 1 0 0 0 1 -1",
            "2 3 0 0 0 0.5 1",
            "3 3 600 0 0 0.5 2",
            "4 4 0 0 0 0.5 1",
            "5 4 -400 0 0 0.5 4",
            "6 2 0 0 0 0.5 1",
            "7 2 0 5000 0 0.5 6",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def exemplar_paths() -> list[Path]:
    """The two Liu-bridge exemplars: one fMOST SWC and one slice-local VEN-L SWC."""
    fmost = repo_root() / "processed_neurons" / "251637" / "007.swc"
    ven_dir = repo_root() / "external" / "liu2026" / "swc" / "VENL"
    preferred = ven_dir / "unSM1114.swc"
    ven = [preferred] if preferred.is_file() else sorted(ven_dir.glob("*.swc"))
    missing = []
    if not fmost.is_file():
        missing.append(str(fmost))
    if not ven:
        missing.append(str(ven_dir / "*.swc"))
    if missing:
        raise FileNotFoundError("exemplar SWC missing: " + "; ".join(missing))
    return [fmost, ven[0]]


def resolve_output(path: str | None, mode: str) -> Path:
    root = receipts_dir().resolve()
    target = root / f"{mode}.json" if path is None else Path(path).resolve()
    if target != root and root not in target.parents:
        raise SystemExit(
            f"output must stay inside {root}; got {target}"
        )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ION pyswcloader 1.1.1 with the NeuroM features used for "
            "the Liu 2026 VEN bridge. Reads SWC files and writes one JSON receipt."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run and write nothing.")
    parser.add_argument("--mode", choices=MODES, default="compare")
    parser.add_argument("--case", type=Path, help="One SWC file.")
    parser.add_argument("--other", type=Path, help="Second SWC, required for --mode distance.")
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in bipolar neuron.")
    parser.add_argument(
        "--exemplars",
        action="store_true",
        help="Compare 251637/007.swc and the first Liu VENL SWC only.",
    )
    parser.add_argument("--r-local", type=float, default=LIU_R_LOCAL_UM, help="NeuroM soma-ball radius in µm.")
    parser.add_argument("--flip", dest="flip", action="store_true", default=True)
    parser.add_argument("--no-flip", dest="flip", action="store_false")
    parser.add_argument("--output", help="Receipt path inside work/pyswcloader_ground/receipts.")
    return parser


def validate(args: argparse.Namespace) -> None:
    if not math.isfinite(args.r_local) or args.r_local <= 0 or args.r_local > R_LOCAL_MAX_UM:
        raise SystemExit(
            f"r-local must be greater than 0 and at most {R_LOCAL_MAX_UM:g} µm; got {args.r_local}"
        )
    selected = sum(bool(item) for item in (args.case, args.synthetic, args.exemplars))
    if args.mode == "distance":
        if args.case is None or args.other is None:
            raise SystemExit("distance mode requires --case and --other")
    elif selected != 1:
        raise SystemExit("provide exactly one of --case, --synthetic, or --exemplars")
    if args.case is not None and not args.case.is_file():
        raise SystemExit(f"case not found: {args.case}")
    if args.other is not None and not args.other.is_file():
        raise SystemExit(f"other not found: {args.other}")


def resolved_cases(args: argparse.Namespace, output: Path) -> list[Path]:
    if args.synthetic:
        return [output.with_name("_synthetic_bipolar.swc")]
    if args.exemplars:
        return exemplar_paths()
    return [args.case]


def run(args: argparse.Namespace) -> dict:
    validate(args)
    output = resolve_output(args.output, args.mode)
    status = neurom_status()
    if args.dry_run and args.synthetic:
        cases = ["<synthetic bipolar neuron>"]
    elif args.mode == "distance":
        cases = [display_path(args.case), display_path(args.other)]
    else:
        cases = [display_path(path) for path in resolved_cases(args, output)]
    receipt = {
        "tool": "tools/pyswcloader_ground.py",
        "version": "0.1.0",
        "settings": {
            **SETTINGS,
            "mode": args.mode,
            "r_local_um": args.r_local,
            "flip_z": args.flip,
            "dry_run": bool(args.dry_run),
        },
        "neurom_available": bool(status["available"]),
        "neurom_version": status.get("version"),
        "output": display_path(output),
        "cases": cases,
    }
    if args.dry_run:
        receipt["writes_files"] = False
        return receipt

    receipt["pyswcloader_version"] = installed_version()
    if args.mode == "distance":
        swc = load_swc_module()
        from pyswcloader import distance

        receipt["morphology_distance"] = float(
            distance.morphology_distance(str(args.case), str(args.other))
        )
        receipt["cases"] = [display_path(args.case), display_path(args.other)]
    else:
        cases = resolved_cases(args, output)
        if args.synthetic:
            synthetic_bipolar(cases[0])
        results = []
        for case in cases:
            if args.mode == "preprocess-risk":
                item = {
                    "path": display_path(case),
                    "allen_preprocess": allen_risk(
                        load_swc_module(), case, flip=args.flip, dimension=ALLEN_DIMENSION_UM
                    ),
                }
            else:
                item = compare_case(
                    case,
                    r_local=args.r_local,
                    flip=args.flip,
                    include_neurom=args.mode == "compare",
                )
            results.append(item)
        receipt["results"] = results
    receipt["writes_files"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, default=_json_default), encoding="utf-8")
    return receipt


def _json_default(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"cannot encode {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        receipt = run(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: receipt[key] for key in ("settings", "cases", "output", "writes_files")}, indent=2))
    if not args_writes(receipt):
        print("dry-run: no receipt written")
    else:
        print(f"wrote {receipt['output']}")
    return 0


def args_writes(receipt: dict) -> bool:
    return bool(receipt.get("writes_files"))


if __name__ == "__main__":
    raise SystemExit(main())
