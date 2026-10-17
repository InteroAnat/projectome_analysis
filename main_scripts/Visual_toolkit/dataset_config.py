"""Resolve dataset links for the VisualToolkit from visual_toolkit_config.yaml.

Version: 1.9.1 (released 2867eef; package version)
Last updated: 2026-10-17
Change history: CHANGELOG.md (this folder)

One operator entry point for the per-sample dataset links (DS00 processed volume,
5 um resample, raw-SWC cache). Modules read through this instead of hard-coding
paths, so a new sample or a moved share is a configuration change.

Operator check:
    cd main_scripts
    python -m Visual_toolkit.dataset_config --check
    python -m Visual_toolkit.dataset_config --check --sample 252527 --strict

Override the file with the PROJECTOME_DATASETS environment variable. When the file
is absent, ``load()`` returns an empty mapping and callers keep their built-in
defaults, so older checkouts still run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "visual_toolkit_config.yaml"
ENV_VAR = "PROJECTOME_DATASETS"
SCHEMA_VERSION = 1
DATASET_KINDS = ("ds00_processed", "resample_5um", "swc_cache")


def config_path() -> Path:
    """Configured YAML path (environment override wins)."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    legacy = DEFAULT_CONFIG.with_name("datasets.yaml")
    return legacy if legacy.exists() else DEFAULT_CONFIG


def _resolve(value, base: Path | None = None):
    """Expand ~ and environment variables; resolve relative paths against the repo."""
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    return candidate if candidate.is_absolute() else (base or REPO_ROOT) / candidate


def _join(root, value):
    """Resolve ``value`` under ``root`` unless it is already absolute or null."""
    if value is None:
        return None
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate) if root else _resolve(candidate)


def load(path: str | Path | None = None) -> dict:
    """Load and validate the dataset links.

    Returns ``{"path", "defaults", "samples", "visual_toolkit"}`` with every path
    resolved. An absent file yields empty sections rather than an error.
    """
    source = Path(path).expanduser() if path else config_path()
    if not source.exists():
        return {"path": source, "defaults": {}, "samples": {}, "visual_toolkit": {}}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency
        raise RuntimeError("PyYAML is required to read the dataset config") from exc
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level must be a mapping")
    unknown = set(raw) - {"version", "defaults", "samples", "visual_toolkit"}
    if unknown:
        raise ValueError(f"{source}: unknown top-level keys {sorted(unknown)}")
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{source}: unsupported version {version!r}; expected {SCHEMA_VERSION}")

    raw_defaults = raw.get("defaults") or {}
    defaults = {
        "ds00_root": _resolve(raw_defaults.get("ds00_root")),
        "resample_5um_root": _resolve(raw_defaults.get("resample_5um_root")),
        "swc_cache_root": _resolve(raw_defaults.get("swc_cache_root")),
    }
    samples = {}
    for sample_id, entry in (raw.get("samples") or {}).items():
        sample_id = str(sample_id)
        if not sample_id:
            raise ValueError(f"{source}: empty sample id")
        entry = entry or {}
        unknown_fields = set(entry) - {"monkey", "ds00_processed", "ds00_sha256", "resample_5um", "swc_cache"}
        if unknown_fields:
            print(f"[dataset_config] {sample_id}: ignoring unknown fields {sorted(unknown_fields)}",
                  file=sys.stderr)
        samples[sample_id] = {
            "monkey": None if entry.get("monkey") is None else str(entry["monkey"]),
            "ds00_processed": _resolve(entry.get("ds00_processed")),
            "ds00_sha256": entry.get("ds00_sha256"),
            "resample_5um": _join(defaults["resample_5um_root"], entry.get("resample_5um")),
            "swc_cache": _join(defaults["swc_cache_root"], entry.get("swc_cache")),
        }
    raw_ui = raw.get("visual_toolkit") or {}
    visual_toolkit = {
        "output_root": _resolve(raw_ui.get("output_root")),
        "auto_reload": bool(raw_ui.get("auto_reload", True)),
    }
    return {"path": source, "defaults": defaults, "samples": samples,
            "visual_toolkit": visual_toolkit}


def sample(sample_id: str, config: dict | None = None) -> dict | None:
    """Resolved links for one sample, or ``None`` when it is not configured."""
    config = config if config is not None else load()
    entry = config["samples"].get(str(sample_id))
    return dict(entry) if entry else None


def sample_ids(config: dict | None = None) -> list[str]:
    config = config if config is not None else load()
    return sorted(config["samples"])


def check(sample_id: str | None = None, config: dict | None = None) -> list[dict]:
    """Report each configured dataset and whether it currently exists."""
    config = config if config is not None else load()
    rows = []
    ids = [sample_id] if sample_id else sample_ids(config)
    for sid in ids:
        entry = config["samples"].get(str(sid))
        if entry is None:
            rows.append({"sample": str(sid), "dataset": "-", "path": None, "exists": None,
                         "note": "not configured"})
            continue
        for kind in DATASET_KINDS:
            path = entry.get(kind)
            exists = None
            if path is not None:
                try:
                    exists = Path(path).exists()
                except OSError:
                    # A share can deny stat() from this session; the path may still be
                    # valid. Report it as unverified instead of failing the whole check.
                    exists = "denied"
            rows.append({"sample": str(sid), "dataset": kind,
                         "path": None if path is None else str(path),
                         "exists": exists})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check dataset links (main_scripts/Visual_toolkit/visual_toolkit_config.yaml).")
    parser.add_argument("--check", action="store_true", help="print resolved paths and whether they exist")
    parser.add_argument("--sample", help="restrict to one sample id")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any configured path is missing")
    args = parser.parse_args(argv)

    if not args.check:
        parser.print_help()
        return 2
    config = load()
    if not config["samples"]:
        print(f"no dataset config found at {config['path']}")
        return 2
    rows = check(args.sample, config)
    if args.json:
        print(json.dumps({"config": str(config["path"]), "rows": rows}, indent=2))
    else:
        print(f"config: {config['path']}\n")
        print(f"{'sample':>8}  {'dataset':<16}  {'exists':<7}  path")
        for row in rows:
            state = row["exists"]
            exists = "-" if state is None else ("denied" if state == "denied" else ("yes" if state else "NO"))
            print(f"{row['sample']:>8}  {row['dataset']:<16}  {exists:<7}  {row['path']}")
        missing = [row for row in rows if row["exists"] is False]
        print(f"\nmissing: {len(missing)} of {len(rows)} configured datasets")
    if args.strict and any(row["exists"] is False for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
