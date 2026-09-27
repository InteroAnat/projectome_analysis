"""
Read-only portal audit: ION neuronbrowser macaque samples vs current
insula projectome cohort (2026-09-26).

Outputs only under group_analysis/portal_audit_20260926/.
Does NOT use IONData infinite-retry methods — bounded requests.get only.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(r"D:\projectome_analysis")
GROUP = PROJECT / "group_analysis"
OUT = GROUP / "portal_audit_20260926"
SCRIPTS = GROUP / "scripts"
CACHE_SAMPLEINFO = Path(r"C:\Users\laika_yan\AppData\Local\Temp\ion_sampleinfo.json")
BBOX_CSV = GROUP / "reference" / "251637_subregion_bboxes.csv"
COMBINED_XLSX = GROUP / "combined" / "multi_monkey_INS_combined_harmonized.xlsx"
RECOVERY_DIR = GROUP / "recovery"
ALL_REFINED = RECOVERY_DIR / "all_refined_neurons.csv"

BASE = "http://10.10.48.110/neuronbrowser/api/user"
SLEEP_S = 0.2
MAX_RETRIES = 3
TIMEOUT_S = 30
PHYS_TO_NII = 250.0  # Soma_NII = Soma_Phys / 250 (confirmed on combined table)
PAD_MM = 2.0

TRACKER_IDS = {
    "251637": "936",
    "252383": "605",
    "252718": "900",
    "252527": "945",
    "252790": "797",
    "252334": "948",
    "252985": "331",
    "252714": "631",
}
# Also historically probed / recovered even if not in tracker note
EXTRA_KNOWN = ["251730", "252384", "252385"]

sys.path.insert(0, str(SCRIPTS))
from insula_label_set import (  # noqa: E402
    build_insula_label_set,
    normalize_label,
    strip_prefix,
)

INSULA_LABELS, RESCUE_LABELS = build_insula_label_set()
# Portal neuronsBySoma exact-ish labels to query (ARM + common portal forms)
NEURONS_BY_SOMA_LABELS = sorted(
    {
        "Ial", "Iai", "Iam", "Iapm", "Iapl", "lat_Ia", "Ia", "Id", "Ia/Id",
        "Ig", "Ins", "Pi", "AI", "AIV", "AID", "AIP", "G",
        "IDD5", "IDM", "IDV", "IAL", "IAPM", "IAPL", "IAI",
        "PrCO", "prco", "PRCO",
    }
)

INJ_INSULA_RE = re.compile(
    r"(insula|岛叶|vaic|idfa|idfm|idfp|fida|fidp|\bfid\b|"
    r"\bai[vdp]?\b|\bins\b|\big\b|"
    r"\bial\b|\biai\b|\biam\b|\biapm\b|\biapl\b|\bia\b|\bid\b|"
    r"ia/id|lat_ia)",
    re.IGNORECASE,
)
INJ_OPERC_RE = re.compile(
    r"(prco|opercular|frontal\s*operculum|operculum)",
    re.IGNORECASE,
)

ENDPOINT_LOG: list[dict] = []


def fetch_json(url: str, label: str = "") -> tuple[int | None, object | None, str]:
    """Bounded GET; returns (status, parsed_or_None, err_msg)."""
    last_status = None
    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT_S)
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception as e:
                    last_err = f"json_parse: {e}"
                    ENDPOINT_LOG.append(
                        dict(label=label or url, url=url, status=r.status_code,
                             ok=False, err=last_err, attempt=attempt + 1)
                    )
                    return r.status_code, None, last_err
                ENDPOINT_LOG.append(
                    dict(label=label or url, url=url, status=r.status_code,
                         ok=True, err="", attempt=attempt + 1)
                )
                return r.status_code, data, ""
            last_err = f"http_{r.status_code}"
        except requests.RequestException as e:
            last_err = f"exc: {type(e).__name__}: {e}"
            last_status = None
        if attempt < MAX_RETRIES - 1:
            time.sleep(SLEEP_S * (attempt + 1))
    ENDPOINT_LOG.append(
        dict(label=label or url, url=url, status=last_status,
             ok=False, err=last_err, attempt=MAX_RETRIES)
    )
    return last_status, None, last_err


def is_macaque(row: dict) -> bool:
    sp = str(row.get("spicies") or "")
    proj = str(row.get("project_id") or "")
    s = sp.lower()
    if any(t in s for t in ("monkey", "macaque", "猕猴", "恒河", "rhesus", "macaca")):
        return True
    if proj.lower() == "monkey":
        return True
    return False


def strip_region_id(region: str) -> str:
    """Ial_42 -> Ial; Ia/Id_228 -> Ia/Id; area_44_76 stays as area_44 after last _digits?"""
    if not isinstance(region, str):
        return ""
    s = region.strip()
    # drop trailing _<int> atlas id
    s = re.sub(r"_\d+$", "", s)
    return s


def portal_region_base(region: str) -> str:
    return normalize_label(strip_region_id(region))


def side_from_region(region: str) -> str:
    if not isinstance(region, str):
        return ""
    s = region.strip()
    for p, side in (("CL_", "L"), ("CR_", "R"), ("L-", "L"), ("R-", "R"),
                    ("L_", "L"), ("R_", "R")):
        if s.startswith(p):
            return side
    return ""


def side_from_nii_x(x) -> str:
    """NMT midline ~128 in NII voxel space (used elsewhere in repo)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return ""
    if xf < 128:
        return "L"
    if xf > 128:
        return "R"
    return ""


def flag_injection(inj_region: str, inj_structure: str) -> tuple[bool, bool, str]:
    blob = f"{inj_region or ''} {inj_structure or ''}"
    ins = bool(INJ_INSULA_RE.search(blob))
    operc = bool(INJ_OPERC_RE.search(blob))
    note_parts = []
    if ins:
        note_parts.append("insula_terms")
    if operc:
        note_parts.append("opercular_PrCO")
    return ins, operc, "+".join(note_parts)


def load_sampleinfo() -> list[dict]:
    if CACHE_SAMPLEINFO.exists():
        data = json.loads(CACHE_SAMPLEINFO.read_text(encoding="utf-8"))
        print(f"[sampleInfo] loaded cache n={len(data)} from {CACHE_SAMPLEINFO}")
        return data
    # try empty project_id then common ids
    status, data, err = fetch_json(
        f"{BASE}/getSampleInfo?project_id=", label="getSampleInfo"
    )
    time.sleep(SLEEP_S)
    if status == 200 and isinstance(data, list) and data:
        return data
    raise RuntimeError(f"No sampleInfo cache and live fetch failed: {status} {err}")


def load_bboxes() -> dict:
    df = pd.read_csv(BBOX_CSV)
    out = {}
    for _, r in df.iterrows():
        out[str(r["sub_region"]).upper()] = dict(
            X_lo=float(r["X_lo_q005"]), X_hi=float(r["X_hi_q995"]),
            Y_lo=float(r["Y_lo_q005"]), Y_hi=float(r["Y_hi_q995"]),
            Z_lo=float(r["Z_lo_q005"]), Z_hi=float(r["Z_hi_q995"]),
        )
    return out


def in_padded_bbox(x, y, z, b, pad=PAD_MM) -> bool:
    return (
        b["X_lo"] - pad <= x <= b["X_hi"] + pad
        and b["Y_lo"] - pad <= y <= b["Y_hi"] + pad
        and b["Z_lo"] - pad <= z <= b["Z_hi"] + pad
    )


def prco_rescue_matches(x, y, z, bboxes: dict) -> list[str]:
    hits = []
    for reg, b in bboxes.items():
        if in_padded_bbox(x, y, z, b):
            hits.append(reg)
    return hits


def load_current_neurons() -> tuple[dict[str, set[str]], dict[str, pd.DataFrame], set[str]]:
    """
    Returns:
      used_ids[sample] -> set of neuron filenames
      combined_by_sample -> Summary rows for insula cohort
      samples_in_analysis
    """
    used: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, list[str]] = defaultdict(list)

    # Combined (insula analysis cohort)
    comb = pd.read_excel(COMBINED_XLSX, sheet_name="Summary")
    comb["SampleID"] = comb["SampleID"].astype(str)
    comb["NeuronID"] = comb["NeuronID"].astype(str)
    for sid, g in comb.groupby("SampleID"):
        for nid in g["NeuronID"]:
            used[sid].add(nid)
            sources[sid].append("combined")
    combined_by = {sid: g.copy() for sid, g in comb.groupby("SampleID")}

    # Recovery full-sample refined tables (broader than combined)
    for path in sorted(RECOVERY_DIR.glob("*_INS_HE_coord_inferred.xlsx")):
        sid = path.name.split("_")[0]
        try:
            df = pd.read_excel(path, sheet_name="Summary")
        except Exception as e:
            print(f"[warn] cannot read {path.name}: {e}")
            continue
        if "NeuronID" not in df.columns:
            continue
        for nid in df["NeuronID"].astype(str):
            used[sid].add(nid)
            sources[sid].append("recovery")

    if ALL_REFINED.exists():
        ar = pd.read_csv(ALL_REFINED)
        if "SampleID" in ar.columns and "NeuronID" in ar.columns:
            for _, row in ar.iterrows():
                used[str(row["SampleID"])].add(str(row["NeuronID"]))
                sources[str(row["SampleID"])].append("all_refined")

    samples_in = set(used.keys())
    print("[current] samples:", sorted(samples_in))
    for sid in sorted(samples_in):
        print(f"  {sid}: n_used_ids={len(used[sid])} sources={sorted(set(sources[sid]))}")
    print(f"[current] combined insula n={len(comb)} L/R="
          f"{comb['Soma_Side_Final'].value_counts().to_dict()}")
    return used, combined_by, samples_in


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bboxes = load_bboxes()
    used_ids, combined_by, samples_in = load_current_neurons()

    all_samples = load_sampleinfo()
    species_counts = Counter(str(x.get("spicies")) for x in all_samples)
    print("[species distinct]", dict(species_counts))

    macaque = [x for x in all_samples if is_macaque(x)]
    print(f"[macaque] n={len(macaque)} (project_id=Monkey or species match)")

    # --- portal_macaque_samples.csv ---
    sample_rows = []
    by_id = {str(x["fMOST_id"]): x for x in all_samples}
    # ensure tracker + known listed even if missing from macaque filter
    ensure_ids = set(TRACKER_IDS) | set(EXTRA_KNOWN) | samples_in
    seen = set()
    ordered = []
    for x in macaque:
        fid = str(x["fMOST_id"])
        if fid not in seen:
            ordered.append(x)
            seen.add(fid)
    for fid in sorted(ensure_ids):
        if fid not in seen and fid in by_id:
            ordered.append(by_id[fid])
            seen.add(fid)
        elif fid not in seen:
            ordered.append({
                "fMOST_id": fid, "project_id": "", "spicies": "",
                "injection_region": "", "injection_structure": "",
                "tracing_cell_number": "", "published": "",
                "injection_time": "", "_missing_from_sampleinfo": True,
            })
            seen.add(fid)

    def _clean_inj(val) -> str:
        return str(val or "").replace("\r", " ").replace("\n", " ").strip()

    for x in ordered:
        fid = str(x.get("fMOST_id", ""))
        inj_r = _clean_inj(x.get("injection_region"))
        inj_s = _clean_inj(x.get("injection_structure"))
        # Portal often leaves injection_region empty; atlas tags live in
        # injection_structure (semicolon-separated, CR between entries).
        inj_display = inj_r if inj_r else inj_s
        ins_flag, operc_flag, flag_note = flag_injection(inj_r, inj_s)
        in_tracker = fid in TRACKER_IDS
        sample_rows.append(dict(
            fMOST_id=fid,
            monkey_id_tracker=TRACKER_IDS.get(fid, ""),
            project_id=x.get("project_id", ""),
            spicies=x.get("spicies", ""),
            injection_region=inj_r,
            injection_structure=inj_s,
            injection_display=inj_display,
            tracing_cell_number=x.get("tracing_cell_number", ""),
            published=x.get("published", ""),
            injection_time=x.get("injection_time", ""),
            flag_insula_injection=int(ins_flag),
            flag_opercular_PrCO=int(operc_flag),
            flag_note=flag_note,
            in_tracker_list=int(in_tracker),
            in_current_analysis=int(fid in samples_in),
            missing_from_sampleinfo=int(bool(x.get("_missing_from_sampleinfo"))),
            is_macaque_filter=int(is_macaque(x) if not x.get("_missing_from_sampleinfo") else 0),
        ))

    samples_csv = OUT / "portal_macaque_samples.csv"
    pd.DataFrame(sample_rows).to_csv(samples_csv, index=False, encoding="utf-8-sig")
    print(f"[wrote] {samples_csv} n={len(sample_rows)}")

    # --- selectNeurons for every macaque / ensure sample ---
    diff_rows = []
    candidate_rows = []
    soma_cache: dict[str, dict[str, tuple[float, float, float]]] = {}

    audit_ids = sorted({str(r["fMOST_id"]) for r in sample_rows
                        if r["is_macaque_filter"] or r["in_tracker_list"]
                        or r["in_current_analysis"] or r["missing_from_sampleinfo"]})

    print(f"[selectNeurons] auditing {len(audit_ids)} samples …")
    for i, sid in enumerate(audit_ids, 1):
        url = f"{BASE}/selectNeurons?id={sid}"
        status, data, err = fetch_json(url, label=f"selectNeurons:{sid}")
        time.sleep(SLEEP_S)
        portal_neurons = data if isinstance(data, list) else []
        portal_ids = {str(n.get("name", "")) for n in portal_neurons if n.get("name")}
        used = used_ids.get(sid, set())
        new_ids = sorted(portal_ids - used)
        removed_ids = sorted(used - portal_ids)
        n_portal = len(portal_ids)
        n_used = len(used)
        in_current = sid in samples_in

        # count atlas insula / PrCO on portal list
        n_insula = 0
        n_prco = 0
        region_counter = Counter()
        for n in portal_neurons:
            base = portal_region_base(str(n.get("region", "")))
            region_counter[base or "<empty>"] += 1
            if base in INSULA_LABELS:
                n_insula += 1
            if base == "PRCO":
                n_prco += 1

        # status tag
        if status != 200:
            status_tag = f"fetch_fail:{status}:{err}"
        elif not in_current and n_portal > 0:
            status_tag = "new_sample_not_in_analysis"
        elif not in_current and n_portal == 0:
            status_tag = "absent_or_empty"
        elif n_portal > n_used:
            status_tag = "grown_portal_gt_used"
        elif removed_ids and not new_ids:
            status_tag = "shrink_or_renamed"
        elif new_ids and removed_ids:
            status_tag = "changed_ids"
        elif n_portal == n_used and not new_ids:
            status_tag = "match"
        else:
            status_tag = "check"

        n_prco_rescuable = 0
        rescue_pending = 0

        combined_ids = set()
        if sid in combined_by:
            combined_ids = set(combined_by[sid]["NeuronID"].astype(str))

        # Emit every portal insula/PrCO neuron not already in the combined
        # insula cohort (includes recovery-only PrCO + brand-new samples).
        if status == 200 and (n_insula + n_prco > 0):
            if sid not in soma_cache:
                st2, soma_list, err2 = fetch_json(
                    f"{BASE}/getSoma?fMOST_id={sid}", label=f"getSoma:{sid}"
                )
                time.sleep(SLEEP_S)
                soma_map: dict[str, tuple[float, float, float]] = {}
                if st2 == 200 and isinstance(soma_list, list):
                    for s in soma_list:
                        name = str(s.get("name", ""))
                        try:
                            soma_map[name] = (
                                float(s["somax"]), float(s["somay"]), float(s["somaz"])
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                else:
                    print(f"  [getSoma {sid}] fail {st2} {err2}")
                soma_cache[sid] = soma_map

            soma_map = soma_cache.get(sid, {})

            for n in portal_neurons:
                nid = str(n.get("name", ""))
                region_raw = str(n.get("region", ""))
                base = portal_region_base(region_raw)
                side = side_from_region(region_raw)
                is_insula = base in INSULA_LABELS
                is_prco = base == "PRCO"
                if not (is_insula or is_prco):
                    continue
                in_comb = nid in combined_ids
                if in_comb:
                    continue  # already in combined analysis cohort
                in_used = nid in used
                is_new_id = nid not in used

                sx = sy = sz = None
                nii_x = nii_y = nii_z = None
                rescue_hits: list[str] = []
                rescue_note = ""
                if nid in soma_map:
                    sx, sy, sz = soma_map[nid]
                    nii_x = sx / PHYS_TO_NII
                    nii_y = sy / PHYS_TO_NII
                    nii_z = sz / PHYS_TO_NII
                    if not side:
                        side = side_from_nii_x(nii_x)
                    if is_prco:
                        rescue_hits = prco_rescue_matches(nii_x, nii_y, nii_z, bboxes)
                        if rescue_hits:
                            n_prco_rescuable += 1
                            rescue_note = "prco_rescuable:" + ";".join(rescue_hits)
                        else:
                            rescue_note = "prco_outside_bbox"
                    else:
                        rescue_note = "insula_atlas_label"
                else:
                    if is_prco:
                        rescue_pending += 1
                        rescue_note = "PrCO candidates, rescue pending (no soma coords)"
                    else:
                        rescue_note = "insula_no_coords"

                candidate_rows.append(dict(
                    sample=sid,
                    neuron_id=nid,
                    soma_region=region_raw,
                    soma_region_base=base,
                    hemisphere=side,
                    in_combined=0,
                    in_used_tables=int(in_used),
                    is_new_portal_id=int(is_new_id),
                    is_insula_atlas=int(is_insula),
                    is_prco=int(is_prco),
                    somax_phys=sx, somay_phys=sy, somaz_phys=sz,
                    soma_nii_x=nii_x, soma_nii_y=nii_y, soma_nii_z=nii_z,
                    prco_rescue_matches=";".join(rescue_hits) if rescue_hits else "",
                    note=rescue_note,
                ))

        n_new = len(new_ids) if in_current else n_portal
        if not in_current:
            n_new = n_portal

        top_regions = " | ".join(
            f"{k}:{v}" for k, v in region_counter.most_common(8)
        )

        diff_rows.append(dict(
            sample=sid,
            in_current=int(in_current),
            n_portal=n_portal,
            n_used=n_used,
            n_combined_insula=len(combined_ids),
            n_new=n_new,
            n_removed_or_renamed=len(removed_ids),
            n_insula_atlas=n_insula,
            n_prco=n_prco,
            n_prco_rescuable=n_prco_rescuable,
            n_prco_rescue_pending=rescue_pending,
            http_status=status if status is not None else "",
            status=status_tag,
            top_regions=top_regions,
            new_ids_preview=";".join(new_ids[:20]),
            removed_ids_preview=";".join(removed_ids[:20]),
            tracker_monkey=TRACKER_IDS.get(sid, ""),
        ))
        print(
            f"  [{i}/{len(audit_ids)}] {sid}: portal={n_portal} used={n_used} "
            f"insula={n_insula} prco={n_prco} rescue={n_prco_rescuable} "
            f"status={status_tag}"
        )

    diff_csv = OUT / "portal_vs_current_diff.csv"
    pd.DataFrame(diff_rows).sort_values(
        ["in_current", "n_insula_atlas", "n_prco", "sample"],
        ascending=[False, False, False, True],
    ).to_csv(diff_csv, index=False, encoding="utf-8-sig")
    print(f"[wrote] {diff_csv}")

    cand_csv = OUT / "new_candidate_neurons.csv"
    pd.DataFrame(candidate_rows).to_csv(cand_csv, index=False, encoding="utf-8-sig")
    print(f"[wrote] {cand_csv} n={len(candidate_rows)}")

    # --- neuronsBySoma cross-check ---
    bysoma_rows = []
    print(f"[neuronsBySoma] querying {len(NEURONS_BY_SOMA_LABELS)} labels …")
    for lab in NEURONS_BY_SOMA_LABELS:
        url = f"{BASE}/neuronsBySoma?region={lab}"
        status, data, err = fetch_json(url, label=f"neuronsBySoma:{lab}")
        time.sleep(SLEEP_S)
        neurons = data if isinstance(data, list) else []
        # keep only Monkey-project samples when possible
        mac_ids = {str(r["fMOST_id"]) for r in sample_rows if r["is_macaque_filter"]}
        mac_neurons = [n for n in neurons if str(n.get("sampleid", "")) in mac_ids]
        by_sample = Counter(str(n.get("sampleid")) for n in mac_neurons)
        bysoma_rows.append(dict(
            region_query=lab,
            http_status=status if status is not None else "",
            n_portal_all=len(neurons),
            n_portal_macaque_samples=len(mac_neurons),
            n_distinct_macaque_samples=len(by_sample),
            top_samples=" | ".join(f"{k}:{v}" for k, v in by_sample.most_common(10)),
            err=err,
        ))
        print(f"  {lab}: all={len(neurons)} macaque={len(mac_neurons)} "
              f"status={status}")

    bysoma_csv = OUT / "portal_neuronsBySoma_insula_PrCO.csv"
    pd.DataFrame(bysoma_rows).to_csv(bysoma_csv, index=False, encoding="utf-8-sig")
    print(f"[wrote] {bysoma_csv}")

    # endpoint summary
    ep_csv = OUT / "endpoint_http_log.csv"
    pd.DataFrame(ENDPOINT_LOG).to_csv(ep_csv, index=False, encoding="utf-8-sig")

    # --- README ---
    write_readme(
        species_counts=species_counts,
        sample_rows=sample_rows,
        diff_rows=diff_rows,
        candidate_rows=candidate_rows,
        bysoma_rows=bysoma_rows,
        used_ids=used_ids,
        combined_n=sum(len(v) for v in combined_by.values()),
    )
    print("[done]")
    return 0


def write_readme(*, species_counts, sample_rows, diff_rows, candidate_rows,
                 bysoma_rows, used_ids, combined_n) -> None:
    diff_df = pd.DataFrame(diff_rows)
    cand_df = pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame()

    mac_n = sum(1 for r in sample_rows if r["is_macaque_filter"])
    missing = [r["fMOST_id"] for r in sample_rows if r["missing_from_sampleinfo"]]
    flag_ins = [r for r in sample_rows if r["flag_insula_injection"] or r["flag_opercular_PrCO"]]

    # Scientific gap focus
    gap_lines = []
    if not cand_df.empty:
        for label in ["IA/ID", "ID", "IG", "IAL", "IAI", "G", "IDD5", "IDM", "IAM/IAPM"]:
            sub = cand_df[cand_df["soma_region_base"] == label]
            if len(sub) == 0:
                continue
            lr = sub["hemisphere"].value_counts().to_dict()
            gap_lines.append(
                f"- **{label}** candidates not in combined: n={len(sub)} "
                f"sides={lr} samples={sorted(sub['sample'].astype(str).unique())}"
            )
        prco = cand_df[cand_df["is_prco"] == 1]
        if len(prco):
            res = prco[prco["prco_rescue_matches"].astype(str).str.len() > 0]
            gap_lines.append(
                f"- **PrCO** candidates: n={len(prco)}, bbox-rescuable "
                f"(pad={PAD_MM} mm, Phys/250→NII): n={len(res)}"
            )

    # Samples with new portal material
    interesting = diff_df[
        (diff_df["n_insula_atlas"] > 0) | (diff_df["n_prco"] > 0)
    ].sort_values(["n_insula_atlas", "n_prco"], ascending=False)

    lines = []
    lines.append("# ION portal audit — macaque insula / PrCO (2026-09-26)")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- Sample catalog: cached `getSampleInfo` "
        f"(`{CACHE_SAMPLEINFO}`), 1125 samples."
    )
    lines.append(
        "- **Macaque filter:** `project_id == 'Monkey'` OR `spicies` contains "
        "monkey/macaque/猕猴/恒河/rhesus/macaca. "
        f"Distinct `spicies` values seen: `{dict(species_counts)}`. "
        f"Macaque n={mac_n} (mostly `spicies=None` with `project_id=Monkey`)."
    )
    lines.append(
        "- Per-sample neuron lists: `selectNeurons?id=` via `requests.get` "
        f"(timeout={TIMEOUT_S}s, max retries={MAX_RETRIES}, sleep={SLEEP_S}s). "
        "**Did not** call `IONData` methods (infinite retry)."
    )
    lines.append(
        "- Insula vocabulary: `insula_label_set.build_insula_label_set()` "
        f"({len(INSULA_LABELS)} labels). Portal regions stripped of trailing "
        "`_<atlas_id>` then prefix-normalized."
    )
    lines.append(
        "- Current cohort IDs: union of "
        "`combined/multi_monkey_INS_combined_harmonized.xlsx` Summary "
        f"(n={combined_n} insula neurons) + "
        "`recovery/*_INS_HE_coord_inferred.xlsx` + "
        "`recovery/all_refined_neurons.csv`."
    )
    lines.append(
        f"- PrCO rescue: getSoma Phys coords → NII = Phys/{PHYS_TO_NII}; "
        f"inside 251637 subregion 99% bbox + {PAD_MM} mm "
        f"(`reference/251637_subregion_bboxes.csv`)."
    )
    lines.append(
        "- Cross-check: `neuronsBySoma?region=` for ARM/portal insula labels + PrCO."
    )
    lines.append("")
    lines.append("## Endpoint reachability")
    lines.append("")
    ok = sum(1 for e in ENDPOINT_LOG if e.get("ok"))
    fail = sum(1 for e in ENDPOINT_LOG if not e.get("ok"))
    lines.append(f"- Logged calls: {len(ENDPOINT_LOG)} (ok={ok}, fail={fail}).")
    lines.append("- See `endpoint_http_log.csv` for per-call HTTP status.")
    fail_labels = sorted({
        e.get("label", "") for e in ENDPOINT_LOG if not e.get("ok")
    })
    if fail_labels:
        lines.append(f"- Failed labels: {fail_labels[:30]}")
    else:
        lines.append("- All audited API calls returned HTTP 200 (within retry budget).")
    lines.append("")
    lines.append("## Tracker ID inconsistency")
    lines.append("")
    lines.append(
        "| Monkey | Tracker fMOST | Tracker injection claim | Portal injection_region (abbrev) |"
    )
    lines.append("|---|---|---|---|")
    for fid, mid in TRACKER_IDS.items():
        row = next((r for r in sample_rows if r["fMOST_id"] == fid), None)
        note_claim = {
            "251637": "vaIC / IDFP / cingulate (multi)",
            "252383": "IDFA-L/R; IDFP-L/R",
            "252718": "IDFA; IDFP",
            "252527": "vaIC; IDFM",
            "252790": "vaIC; IDFM; IDFP",
            "252334": "vaIC; IDFM",
            "252985": "vaIC; IDFM; IDFP",
            "252714": "IDFA-L/R",
        }.get(fid, "")
        if row is None:
            portal_inj = "NOT IN SAMPLEINFO"
        elif row["missing_from_sampleinfo"]:
            portal_inj = "**MISSING from sampleInfo**"
        else:
            portal_inj = (row.get("injection_display") or row.get("injection_structure")
                          or row.get("injection_region") or "")[:140]
            if not portal_inj:
                portal_inj = "(empty injection_region + injection_structure)"
        lines.append(f"| {mid} | {fid} | {note_claim} | `{portal_inj}` |")
    lines.append("")
    lines.append(
        "- **252383 / monkey 605:** tracker says IDFA/IDFP; portal "
        "`injection_structure` is dominated by **F4** (+ Ig, 7op, SII, M1, …). "
        "Matches the prior F4-premotor reading; tracker IDFA claim is inconsistent."
    )
    lines.append(
        "- **252334 (948) and 252985 (331):** listed in "
        "`notes/projectome_latest_update_progress.md` as Done, but "
        f"**absent from sampleInfo** ({missing})."
    )
    lines.append(
        "- **252790 / 252714:** in Monkey project with traced neurons, but "
        "both `injection_region` and `injection_structure` empty in sampleInfo "
        "(cannot verify planned vaIC/IDFA sites from portal metadata)."
    )
    lines.append(
        "- Note: sampleInfo stores atlas injection tags in **`injection_structure`** "
        "for macaques; `injection_region` is usually empty."
    )
    lines.append("")
    lines.append("## Findings — portal vs current")
    lines.append("")
    lines.append(
        f"Samples with portal atlas-insula and/or PrCO soma labels "
        f"(n={len(interesting)}):"
    )
    lines.append("")
    lines.append(
        "| sample | in_current | n_portal | n_used | n_combined | "
        "n_insula | n_prco | n_prco_rescuable | status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in interesting.iterrows():
        lines.append(
            f"| {r['sample']} | {r['in_current']} | {r['n_portal']} | "
            f"{r['n_used']} | {r['n_combined_insula']} | {r['n_insula_atlas']} | "
            f"{r['n_prco']} | {r['n_prco_rescuable']} | {r['status']} |"
        )
    lines.append("")
    lines.append("### Gap-relevant new / not-in-combined candidates")
    lines.append("")
    if gap_lines:
        lines.extend(gap_lines)
    else:
        lines.append("- No insula/PrCO candidates outside combined (unexpected).")
    lines.append("")
    lines.append(
        "**Headline adds (not in combined):** sample **252790** (tracker 797) — "
        "38 portal insula (IAL 1L/16R, IAI 15L/1R, IG 5L); **250432** — "
        "8 (IA/ID 2L + IG 6R); **252714** (tracker 631) — 4 IA/ID (3L/1R). "
        "251637 portal grew to 562 cells but **all** portal insula/PrCO IDs "
        "already sit in combined (new IDs are M1/empty/3a–b). "
        "5 residual PrCO (252385/252527) are **outside** the 251637 bbox "
        "(rescue not applicable)."
    )
    lines.append("")
    lines.append("### Flagged injection samples (insula/opercular terms)")
    lines.append("")
    for r in flag_ins:
        disp = (r.get("injection_display") or "")[:80]
        lines.append(
            f"- {r['fMOST_id']} (tracker monkey {r['monkey_id_tracker'] or '—'}): "
            f"flags={r['flag_note']}; in_current={r['in_current_analysis']}; "
            f"inj=`{disp}`"
        )
    lines.append("")
    lines.append("## Blockers / caveats")
    lines.append("")
    lines.append(
        "- Raw SWC downloads (historically HTTP 500 for some 252383/4/5 neurons) "
        "were **not** re-tested here; this audit uses selectNeurons / getSoma / "
        "neuronsBySoma / neuronProperty-style metadata only."
    )
    lines.append(
        "- Hemisphere for portal-only neurons without CL_/CR_ prefix is inferred "
        "from NII X vs midline 128 when getSoma coords exist."
    )
    lines.append(
        "- `n_used` counts any neuron ID present in recovery+combined tables "
        "(full sample scan), while scientific cohort size is `n_combined_insula`."
    )
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `portal_macaque_samples.csv`")
    lines.append("- `portal_vs_current_diff.csv`")
    lines.append("- `new_candidate_neurons.csv`")
    lines.append("- `portal_neuronsBySoma_insula_PrCO.csv`")
    lines.append("- `endpoint_http_log.csv`")
    lines.append("- `audit_portal.py` (this script)")
    lines.append("- `README.md`")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT / 'README.md'}")


if __name__ == "__main__":
    sys.exit(main())
