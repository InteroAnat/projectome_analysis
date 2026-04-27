"""
Phase 5c: Cross-monkey FNT distance matrix on NMT-aligned SWCs.

Pipeline (Path A, confirmed in Phase 5b):
  1. Fetch NMT-aligned SWCs via IONData.getNeuronByID for all 306 keepers
     (251637 + new-monkey insula keepers). Write to:
         group_analysis/fnt/nmt_swcs/{sample_id}/{neuron_id}.swc
  2. Mirror right-hemifield nodes (x > 32000) to left (x' = 64000 - x):
         group_analysis/fnt/mirrored_swcs/{NeuronUID}.swc
     where NeuronUID = "{sample_id}_{neuron_basename}" so neuron names are
     globally unique (e.g. 251637_001, 252385_063).
  3. fnt-from-swc {file}.swc -> {file}.fnt
  4. fnt-decimate -d 5000 -a 5000 {file}.fnt -> {file}.decimate.fnt
  5. Update .decimate.fnt last line to "0 Neuron {NeuronUID}".
  6. fnt-join *.decimate.fnt -> multi_monkey_INS_joined.fnt
  7. fnt-dist multi_monkey_INS_joined.fnt -o multi_monkey_INS_dist.txt

Outputs:
  group_analysis/fnt/multi_monkey_INS_joined.fnt
  group_analysis/fnt/multi_monkey_INS_dist.txt
  group_analysis/fnt/run_log.csv
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
import glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
sys.path.insert(0, NEUROVIS)

import IONData  # noqa: E402

GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX = os.path.join(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
FNT_DIR = os.path.join(GROUP_DIR, "fnt")
NMT_SWC_DIR = os.path.join(FNT_DIR, "nmt_swcs")
MIRROR_DIR = os.path.join(FNT_DIR, "mirrored_swcs")
WORK_DIR = os.path.join(FNT_DIR, "fnt_work")
for d in (FNT_DIR, NMT_SWC_DIR, MIRROR_DIR, WORK_DIR):
    os.makedirs(d, exist_ok=True)

NMT_MIDLINE_X = 32000.0  # NMT v2.1 physical-microns midline
DECIMATE_D = 5000
DECIMATE_A = 5000


def run_cmd(cmd: str, verbose: bool = False) -> tuple[bool, str, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.returncode == 0, p.stdout, p.stderr)


def fetch_nmt_swc(iondata, sample_id: str, neuron_id: str) -> str | None:
    """Fetch NMT-aligned SWC text via IONData.getNeuronByID and save to
    group_analysis/fnt/nmt_swcs/{sample_id}/{neuron_id}.
    Returns the saved file path, or None on failure."""
    out_dir = os.path.join(NMT_SWC_DIR, sample_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, neuron_id)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        text = iondata.getNeuronByID(sample_id, neuron_id)
    except Exception as e:
        print(f"  [err] fetch {sample_id}/{neuron_id}: {e}")
        return None
    if not text:
        print(f"  [err] empty SWC for {sample_id}/{neuron_id}")
        return None
    # Strip Windows line endings, ensure trailing newline
    text = text.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out_path


def mirror_swc(in_path: str, out_path: str,
               midline: float = NMT_MIDLINE_X) -> tuple[bool, int, int]:
    """Read SWC, fold any node with x > midline to left hemifield via
    x' = 2*midline - x. Write to out_path. Returns (ok, n_flipped, n_kept)."""
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out_lines = []
    flipped = kept = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            out_lines.append(line)
            continue
        parts = s.split()
        if len(parts) < 5:
            out_lines.append(line)
            continue
        try:
            x = float(parts[2])
        except ValueError:
            out_lines.append(line)
            continue
        if x > midline:
            parts[2] = f"{2 * midline - x:.6f}"
            flipped += 1
        else:
            kept += 1
        out_lines.append(" ".join(parts) + "\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(out_lines)
    return True, flipped, kept


def update_fnt_name(fnt_file: str, name: str) -> bool:
    try:
        with open(fnt_file, "rb") as f:
            content = f.read()
        if not content:
            return False
        text = content.decode("utf-8", errors="replace").replace("\r\n", "\n")
        ls = text.splitlines()
        if not ls:
            return False
        new_last = f"0 Neuron {name}"
        if ls[-1].strip() != new_last.strip():
            ls[-1] = new_last
        with open(fnt_file, "w", encoding="utf-8", newline="\n") as f:
            for li in ls:
                f.write(li + "\n")
        return True
    except Exception as e:
        print(f"  [err] update_fnt_name {fnt_file}: {e}")
        return False


def main() -> int:
    # Load combined table -> list of (SampleID, NeuronID) keepers
    combined = pd.read_excel(COMBINED_XLSX, sheet_name="Summary")
    print(f"[5c] {len(combined)} neurons in combined table")
    keepers = combined[["SampleID", "NeuronID"]].drop_duplicates().copy()
    keepers["NeuronUID"] = (keepers["SampleID"].astype(str) + "_"
                              + keepers["NeuronID"].astype(str).str.replace(
                                    ".swc", "", regex=False))
    print(f"  unique neurons: {len(keepers)}")

    iondata = IONData.IONData()
    log_rows = []

    success = 0
    for i, row in enumerate(keepers.itertuples(), 1):
        sid = str(row.SampleID)
        nid = str(row.NeuronID)
        uid = row.NeuronUID
        print(f"[{i}/{len(keepers)}] {sid}/{nid}  ->  UID={uid}")

        # Step 1: fetch NMT-aligned SWC
        nmt_path = fetch_nmt_swc(iondata, sid, nid)
        if not nmt_path:
            log_rows.append(dict(NeuronUID=uid, SampleID=sid, NeuronID=nid,
                                  status="fetch_failed"))
            continue

        # Step 2: mirror right -> left
        mirror_path = os.path.join(MIRROR_DIR, f"{uid}.swc")
        if not (os.path.exists(mirror_path) and os.path.getsize(mirror_path) > 0):
            ok, flipped, kept = mirror_swc(nmt_path, mirror_path,
                                            midline=NMT_MIDLINE_X)
            if not ok:
                log_rows.append(dict(NeuronUID=uid, status="mirror_failed"))
                continue
        else:
            flipped = kept = -1  # cached

        # Step 3: fnt-from-swc
        fnt_path = os.path.join(WORK_DIR, f"{uid}.fnt")
        if not (os.path.exists(fnt_path) and os.path.getsize(fnt_path) > 0):
            cmd = f'fnt-from-swc "{mirror_path}" "{fnt_path}"'
            ok, _, err = run_cmd(cmd)
            if not ok or not os.path.exists(fnt_path):
                print(f"  [err] fnt-from-swc: {err[:200]}")
                log_rows.append(dict(NeuronUID=uid, status="fnt_from_swc_failed"))
                continue

        # Step 4: fnt-decimate
        dec_path = os.path.join(WORK_DIR, f"{uid}.decimate.fnt")
        if not (os.path.exists(dec_path) and os.path.getsize(dec_path) > 0):
            cmd = (f'fnt-decimate -d {DECIMATE_D} -a {DECIMATE_A} '
                   f'"{fnt_path}" "{dec_path}"')
            ok, _, err = run_cmd(cmd)
            if not ok or not os.path.exists(dec_path):
                print(f"  [err] fnt-decimate: {err[:200]}")
                log_rows.append(dict(NeuronUID=uid, status="decimate_failed"))
                continue

        # Step 5: update name
        if not update_fnt_name(dec_path, uid):
            log_rows.append(dict(NeuronUID=uid, status="rename_failed"))
            continue

        success += 1
        log_rows.append(dict(NeuronUID=uid, SampleID=sid, NeuronID=nid,
                              flipped_nodes=flipped, kept_nodes=kept,
                              status="ok"))

    log_df = pd.DataFrame(log_rows)
    log_csv = os.path.join(FNT_DIR, "run_log.csv")
    log_df.to_csv(log_csv, index=False)
    print(f"\n[saved] {log_csv}")
    print(f"  success: {success}/{len(keepers)}")
    print(log_df["status"].value_counts().to_string())

    if success == 0:
        print("[fatal] no successful FNT files; aborting join/dist")
        return 1

    # Step 6: fnt-join (use bash to expand wildcard - command line too long
    # otherwise on Windows for ~300 files)
    joined = os.path.join(FNT_DIR, "multi_monkey_INS_joined.fnt")
    if os.path.exists(joined):
        os.remove(joined)
    pattern = WORK_DIR.replace("\\", "/") + "/*.decimate.fnt"
    join_cmd = f'bash -c \'fnt-join.exe {pattern} -o {joined.replace(chr(92), "/")}\''
    print(f"\n[join] {join_cmd}")
    ok, out, err = run_cmd(join_cmd, verbose=True)
    print(out[:400] if out else "")
    if err:
        print("stderr:", err[:400])
    if not ok or not os.path.exists(joined):
        print("[fatal] join failed")
        return 2
    print(f"[saved] {joined}  size={os.path.getsize(joined):,} bytes")

    # Step 7: fnt-dist
    dist_path = os.path.join(FNT_DIR, "multi_monkey_INS_dist.txt")
    if os.path.exists(dist_path):
        os.remove(dist_path)
    dist_cmd = f'fnt-dist.exe "{joined}" -o "{dist_path}"'
    print(f"\n[dist] {dist_cmd}")
    ok, out, err = run_cmd(dist_cmd, verbose=True)
    print(out[:400] if out else "")
    if err:
        print("stderr:", err[:400])
    if not ok or not os.path.exists(dist_path):
        print("[fatal] dist failed")
        return 3
    print(f"[saved] {dist_path}  size={os.path.getsize(dist_path):,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
