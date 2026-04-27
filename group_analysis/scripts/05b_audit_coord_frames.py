"""
Phase 5b: Coordinate-frame audit.

Question: are SWCs returned by IONData.getNeuronByID (the "normalized" /
non-raw variant) in a shared NMT v2.1 macaque coordinate frame? This is a
hard requirement for cross-monkey FNT to be meaningful.

Method: for one example neuron from each sample, compare:
  - SWC root soma coordinate (parsed from getNeuronByID text)
  - Summary sheet Soma_Phys_X/Y/Z from that monkey's step1 output

If all 4 monkeys' SWC root coords match their Summary's Soma_Phys_X/Y/Z, we
know the SWC is in the same physical (NMT-aligned) frame -> Path A is OK.

Output:
  group_analysis/fnt/coord_frame_audit.csv
"""
from __future__ import annotations

import os
import sys
import glob
import pandas as pd

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
sys.path.insert(0, NEUROVIS)

import IONData  # noqa: E402

GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")
OUT_DIR = os.path.join(GROUP_DIR, "fnt")
os.makedirs(OUT_DIR, exist_ok=True)

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]


def find_results_xlsx(sid):
    pattern = os.path.join(STEP1_DIR, f"{sid}_*_region_analysis", "tables",
                           f"{sid}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def parse_swc_root(swc_text: str):
    """Return (x, y, z, n_lines) of the first non-comment node in SWC text."""
    n = 0
    root = None
    for line in swc_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
        if root is None:
            parts = s.split()
            if len(parts) >= 5:
                try:
                    root = (float(parts[2]), float(parts[3]), float(parts[4]))
                except ValueError:
                    pass
    return root, n


def main():
    iondata = IONData.IONData()
    rows = []

    # 251637 (already untouched but check from existing cached SWCs if present)
    samples_to_check = ["251637"] + NEW_SAMPLES

    for sid in samples_to_check:
        # Pick one neuron — first IAL/Ial neuron from the sample if possible
        try:
            neurons = iondata.getNeuronListBySampleID(sid)
        except Exception as e:
            rows.append(dict(sample_id=sid, status=f"err_list: {e}"))
            continue
        if not neurons:
            rows.append(dict(sample_id=sid, status="empty_neuron_list"))
            continue
        target = neurons[0]
        nid = target["name"]
        try:
            swc_text = iondata.getNeuronByID(sid, nid)
        except Exception as e:
            rows.append(dict(sample_id=sid, neuron=nid,
                             status=f"err_swc: {e}"))
            continue
        if not swc_text:
            rows.append(dict(sample_id=sid, neuron=nid,
                             status="empty_swc"))
            continue
        root, n_lines = parse_swc_root(swc_text)
        if root is None:
            rows.append(dict(sample_id=sid, neuron=nid,
                             status="no_root", n_lines=n_lines))
            continue

        # Find this neuron's row in step1 Summary (251637 has its own special table)
        if sid == "251637":
            xlsx = os.path.join(PROJECT_ROOT, "neuron_tables_new",
                                "251637_INS_HE_inferred.xlsx")
        else:
            xlsx = find_results_xlsx(sid)
        ref_row = None
        if xlsx and os.path.exists(xlsx):
            try:
                summ = pd.read_excel(xlsx, sheet_name="Summary")
                m = summ[summ["NeuronID"] == nid]
                if len(m):
                    ref_row = m.iloc[0]
            except Exception as e:
                pass

        ref_phys = (ref_row["Soma_Phys_X"], ref_row["Soma_Phys_Y"],
                    ref_row["Soma_Phys_Z"]) if ref_row is not None else (None, None, None)
        ref_nii = (ref_row["Soma_NII_X"], ref_row["Soma_NII_Y"],
                   ref_row["Soma_NII_Z"]) if ref_row is not None else (None, None, None)

        # Compute distance between SWC root and ref Phys
        if ref_phys[0] is not None:
            dxyz = tuple(abs(a - b) for a, b in zip(root, ref_phys))
        else:
            dxyz = (None, None, None)

        rows.append(dict(
            sample_id=sid, neuron=nid, n_lines=n_lines,
            swc_root_x=round(root[0], 4), swc_root_y=round(root[1], 4), swc_root_z=round(root[2], 4),
            ref_phys_x=ref_phys[0], ref_phys_y=ref_phys[1], ref_phys_z=ref_phys[2],
            ref_nii_x=ref_nii[0], ref_nii_y=ref_nii[1], ref_nii_z=ref_nii[2],
            dx=dxyz[0], dy=dxyz[1], dz=dxyz[2],
        ))
        print(f"[{sid}] {nid}: SWC root=({root[0]:.2f}, {root[1]:.2f}, {root[2]:.2f}); "
              f"ref Phys=({ref_phys[0]}, {ref_phys[1]}, {ref_phys[2]}); "
              f"NII=({ref_nii[0]}, {ref_nii[1]}, {ref_nii[2]}); "
              f"diff=({dxyz[0]}, {dxyz[1]}, {dxyz[2]})")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "coord_frame_audit.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
