"""Merge terminal region evidence into living notes, preserving dated history."""
import json
from pathlib import Path
from datetime import datetime
import rerun_actual_region_tables as audit

ROOT, OUT = audit.ROOT, audit.OUT

def main():
    report = json.loads((OUT / "final_actual_region_readback.json").read_text(encoding="utf-8"))
    cases = json.loads((OUT / "final_boundary_and_manual_cases.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed" and report["unique_neurons"] == 1912
    assert len(report["subtables"]) == 11 and not cases["direct_atlas_soma_changes"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    section = f"""### Completed region software and visual review ({stamp})

**Agent: Codex | Date: {stamp} (Asia/Shanghai)**

The public step1 path completed isolated real-data exports for **1,483 cached atlas-space sources** and **127 recovered own-sample atlas-space sources**. Recovery retained the original missing-source inventory and verified each sample's endpoint against a known cached graph. The existing regeneration path produced a fresh **562-neuron 251637 master and eleven subtables**, preserving all historical template memberships. Independent saved-output readback passes **1,912 distinct neurons / 2,172 export rows** with **zero failures**, zero changed bound SWCs and zero changed protected inputs. The overlap of 260 neurons between the cached and full master batches is counted once in the distinct-neuron total.

The checks cover full-precision roots, atlas soma/endpoint lookup, official hemisphere-mask semantics, label conflicts, declared neuron-type rules, parent-child lengths, parent-voxel projection allocation, hierarchy labels, ipsi/contra/unknown lengths, finest projection/log-strength matrices and cross-sheet types. Every rebuilt subtable sheet matches the corresponding fresh-master subset. All **seven baseline rounding-boundary rows** (five new-monkey rows and two older-table instances) were reconciled against source roots, with **zero direct atlas soma-label changes**. Initial checker findings came from vector NumPy rounding versus the existing scalar Python per-edge rounding; production length policy was retained and the corrected independent checker passed. Original findings remain in the dated batch reports and final report.

The official hemisphere reference is wired into the existing editable step1 script. Manual soma labels and their sources are retained alongside automatic lookup. All twelve April manual type-correction cases were reconciled: ten recompute as ITi; **251637/112 and /114** retain manual **L-IDM / L-IDD5**, conflict with their atlas-space R roots, and resolve to **Unknown / Unclassified** in the new derivatives. Their ipsi/contra classification is withheld and unknown-side lengths remain explicit. Original accepted workbooks are unchanged; these conflicts still require annotation/native/transform reconciliation.

Actual visual inspection covered eleven atlas example/conflict figures and **all seventeen native locator pages / 299 available full-section locators** across the four copied samples. Atlas-L crops appear on viewer-right and atlas-R on viewer-left in these native locators, without obvious relative-side contradiction. **252384/003** has a marked crop in/adjacent to a blank rectangular section area and needs local tissue review. This is locator-level inspection, not high-resolution inspection of every soma/trace or absolute-orientation acceptance.

**Rendering distinction:** the GUI source is unchanged, but its shared toolkit has substantive geometry, spacing/sign, clipping, missing-source and provenance changes. The public soma/wide-field styles remain present; a saved actual 252385/050 pair was rendered through those public methods and visually inspected. The October batch uses custom panels, so its style is not identical to the public toolkit/GUI. Atlas and locator sheets are supplementary checking figures. Existing bulk images were not overwritten during this review. See [visual methods and style](../notes/region_analysis_review_20261004/visual_methods_and_style.md).

**Sixteen region tests pass**, including real XLSX export readback. Seven pre-review production snapshots and the scoped rollback patch are saved; reverse-apply dry-run passes, preserving inherited population/regenerator edits. New reports/modules are separately identifiable. No commit, push, cohort promotion or accepted K was made.

Evidence: [region review](../notes/region_analysis_review_20261004/README.md), `final_actual_region_readback.json` (SHA-256 `{audit.digest(OUT / 'final_actual_region_readback.json')}`), `final_boundary_and_manual_cases.json`, `native_contact_sheet_visual_readback.json`, `historical_notes_readback.json`, and `region_unittest_20261004.log`. Native calibration/orientation, registration accuracy, fine region boundaries, membership acceptance and biological cluster count remain scientific follow-up.

"""
    note = ROOT / "docs/project_note.md"
    text = note.read_text(encoding="utf-8")
    text = text.replace("## Status snapshot (2026-10-04; region and visual laterality validation active)",
                        "## Status snapshot (2026-10-04; software/output review complete; scientific acceptance pending)", 1)
    text = text.replace("**Agent: Codex | Date: 2026-10-04 07:01 (Asia/Shanghai)**", f"**Agent: Codex | Date: {stamp} (Asia/Shanghai)**", 1)
    text = text.replace("Region-analysis and laterality validation is an additional requirement and remains in progress.",
                        "Region software readback now passes 1,912 distinct neurons and eleven rebuilt subtables; two manual-label conflicts remain explicit.", 1)
    text = text.replace("| **Next** | Complete real-data region/laterality reruns, integrate the official hemisphere reference, reconcile refined-label conflicts and audit all neuron-table exports before reporting completion.",
                        "| **Next** | Review the two retained manual-label conflicts and the flagged native section crop; preserve original annotations until native/transform evidence resolves them.", 1)
    text = text.replace("The additional region/laterality requirement and the full completion audit remain active.",
                        "The additional region software/export audit has passed; anatomical acceptance remains separate.", 1)
    marker = "### Region and visual laterality audit (2026-10-04 07:01)"
    assert marker in text
    text = text.replace(marker, section + marker, 1)
    note.write_text(text, encoding="utf-8")
    readme = OUT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("Five of the new-monkey rows are on a rounding boundary and still need comparison with the original full-precision SWC roots.",
                        "All five new-monkey rounding-boundary rows have now been reconciled with full-precision SWC roots; direct soma labels are unchanged.", 1)
    text = text.replace("Its full real-data execution remains to be verified.",
                        "Its full 562-neuron real-data execution and every sheet of eleven rebuilt subtables pass independent saved-output readback.", 1)
    start, end = text.index("## Remaining work"), text.index("## Reversibility")
    completed = """## Terminal real-data verification

`final_actual_region_readback.json` passes 1,912 distinct neurons / 2,172 exported rows: 1,483 cached-source rows, 127 recovered-source rows, and a 562-neuron historical master (260 overlap). Every sheet of all eleven rebuilt subtables matches its fresh-master membership subset. Bound source SWCs and protected workbooks/manifest/cohort inputs are unchanged. The original vector-rounding checker findings remain preserved; the corrected scalar-rounding independent checker passes without changing production length policy.

`final_boundary_and_manual_cases.json` reconciles seven baseline rounding-boundary instances and all twelve April correction cases, with no direct atlas soma-label changes. Ten correction cases recompute as ITi. The two retained L-label / R-mask conflicts resolve to Unknown/Unclassified in isolated derivatives; accepted annotations remain unchanged.

`native_contact_sheet_visual_readback.json` records actual inspection of all seventeen pages containing 299 available native locators. Relative side matches the atlas annotations at locator scale; 252384/003's crop is in/adjacent to a blank rectangular source area and needs tissue review. The other five samples lack full-section locators in the copied source set. No absolute native orientation or fine cytoarchitectural acceptance is inferred.

See `visual_methods_and_style.md` for the public toolkit versus custom bulk and supplementary QC distinction. GUI source is unchanged; toolkit backend changes are substantive. The public-style actual 252385/050 soma/wide-field pair was visually inspected, with hashed input/output provenance. Existing bulk packages were not overwritten.

Software/output review is complete. Scientific follow-up is reconciliation of the two manual-label conflicts, tissue review of the flagged crop, native orientation/registration evidence, accepted membership and biological cluster count. These are not software-test conclusions.

"""
    text = text[:start] + completed + text[end:]
    readme.write_text(text, encoding="utf-8")
    completion = ROOT / "notes/bulk_visual_review_20261002/completion_audit_20261003.md"
    text = completion.read_text(encoding="utf-8")
    current = f"""# Current completion reconciliation, {stamp} Asia/Shanghai

**Agent: Codex. Software/output work complete; scientific acceptance pending.** The dated October 3 checkpoint below retains its historical scope and active-process descriptions. Cursor is no longer running this task.

- Repository/notes review, fMOST research (29 primary/official references and eight Zotero indexed-PDF method checks), clustering improvements and six exploratory runs are complete on `codex/fmost-methods-review-20261002`. No accepted cluster count or cohort promotion.
- All-nine bulk scope is 832 identities: 827 usable contexts and 810 usable soma packages; explicit own-source/missing-center failures remain reviewable. Final integrity passes 85,105 bound files / 379,262,869,363 bytes; all source/integrity workers have exited. See `final_component_integrity_20261004.json`.
- Region public-path reruns and independent readback pass 1,912 distinct neurons / 2,172 exported rows, including all 127 recovered atlas-space sources and the 562-neuron master. Every sheet of eleven rebuilt subtables matches the fresh-master subset. Protected source inputs remain unchanged. Region regression suite: sixteen tests pass.
- All 299 available full-section locators were visually checked on seventeen pages, alongside eleven atlas panels and actual public-style toolkit reference plots. Flag 252384/003's blank source area for tissue review. Two 251637 manual-label / atlas-side conflicts remain explicit, with resolved side Unknown in isolated derivatives.
- GUI source is unchanged; shared toolkit backend corrections are substantive. October bulk custom panels and supplementary black-and-white QC sheets must not be described as identical public toolkit/GUI styling. Original soma and wider-field trace review purpose remains; existing bulk packages were preserved. See `../region_analysis_review_20261004/visual_methods_and_style.md`.
- Seven-file scoped region rollback snapshots/patch are saved and reverse-apply checked. No commit, push, accepted-workbook replacement or publication occurred. Native calibration/orientation, transform accuracy, cytoarchitecture, membership and K remain scientific review items.

Current detailed evidence: `../region_analysis_review_20261004/README.md`, `final_actual_region_readback.json`, `final_boundary_and_manual_cases.json`, and `../../docs/project_note.md`.

---

"""
    completion.write_text(current + text, encoding="utf-8")
    print(json.dumps({"status": "merged_terminal_evidence", "updated": [str(p.relative_to(ROOT)) for p in (note, readme, completion)]}))

if __name__ == "__main__":
    main()
