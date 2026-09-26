"""Run step1 region analysis for a single fMOST sample ID (CLI)."""
from __future__ import annotations

import sys

from pathlib import Path

# Reuse Phase 2 implementation
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "step1_multi", SCRIPT_DIR / "02_run_step1_multi.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run_step1_one_sample.py <fMOST_ID>")
        return 2
    sid = sys.argv[1].strip()
    import nibabel as nib
    import pandas as pd

    for f in (
        _mod.ATLAS_PATH,
        _mod.TABLE_PATH,
        _mod.TEMPLATE,
        _mod.CTX_HIER_CSV,
        _mod.SUBCTX_HIER_CSV,
    ):
        if not Path(f).exists():
            print(f"[ERROR] Missing: {f}")
            return 1

    print(f"[step1] Loading atlas once for {sid}")
    atlas_data = nib.load(_mod.ATLAS_PATH).get_fdata()
    atlas_table = pd.read_csv(_mod.TABLE_PATH, delimiter="\t")
    template = nib.load(_mod.TEMPLATE)
    out = _mod.run_one(sid, atlas_data, atlas_table, template)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
