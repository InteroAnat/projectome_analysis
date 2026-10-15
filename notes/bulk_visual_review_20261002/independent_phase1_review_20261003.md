# Independent phase-1 review

**Author: Codex independent subagent | Date: 2026-10-03 (Asia/Shanghai)**

This review used source reads and synthetic disposable temporary directories.
It did not modify production code, crop arrays, manifests, reviewer tables or
canonical workbooks. Cursor/Grok was editing concurrently, so these findings
are provisional snapshot results rather than acceptance of a completed phase.
Pending phase-2 trace-depth and export-sidecar issues are recorded separately.

## Snapshot provenance

- Initial renderer SHA256:
  `a2d93b63efde382250ab0b244c28a3e8a27d1af2dc96ce4098dc8c379884cec3`.
- Both annotation-loss reproductions and the baseline test ran against:
  `33ae1d4e657d0bf6702b12b2cd0ec2cfe9aa22d2c8730b1ea4f6c8365c85d1fe`.
  This source hash was unchanged immediately before and after those tests.
- A later verifier conflict/read-only test ran against:
  `e0882d75c44c517e9f461965cdabd6c8dfe6bd330090c7098a773af348e7df2b`.
  This source hash was also unchanged during its test. The implementation
  changed between inspection snapshots; do not treat earlier in-progress
  verifier definitions as final failures.
- Nine pure-function assertions for historical labels, missing-neighbor
  coverage and anisotropic dimensions passed against stable SHA256
  `913cb0a5c3ca0cb119eb2c03a4dedb2cd4b79ef60ba30e6a0628d2dfa9e8f4e7`.
  This final source checkpoint differs from the initial SHA; overall review
  remains provisional while Cursor/Grok completes its edits and tests.

## Additional annotation-preservation defects

1. `load_human_annotations` overwrites repeated UIDs within one input table.
   Around lines 1199-1215 in the tested snapshot, assignment to `found[uid]`
   keeps only the last row. Two CSV rows for `252385|003.swc`, first with
   `human_region=IAL` and `human_notes=first decision`, then `IG` and
   `second decision`, return only the second answer. Cross-file conflict
   detection never sees the first answer. Detect conflicting duplicate UIDs
   within each input before writes. Complementary duplicate values also need
   a deliberate merge or rejection rather than implicit last-row precedence.
2. `write_correction_table` retains annotations only for UIDs in the current
   manifest. Around lines 1239-1293, annotations are read but an annotated UID
   missing from the manifest is silently omitted when both tables are replaced.
   A disposable old table containing annotated `252385|004.swc`, with a rebuilt
   manifest containing only `252385|003.swc`, loses the old decision. Before
   overwriting either format, abort for nonblank orphaned annotations or
   preserve them in an explicit archive whose contents and path are recorded.
   A changed cohort must not silently discard a reviewer's work.

## Positive phase-1 checks

- The focused canonical-baseline test detected changed input bytes, raised
  `RuntimeError`, and preserved the original baseline bytes exactly.
- Separate historical normalization preserves `area_44`/`CL_area_44`, while
  portal normalization removes the atlas index from `area_44_76`.
- Physical orthogonal rendering now passes the spacing-derived aspect to
  `imshow` and includes root crosshairs, scale bars and physical dimensions.
- The later isolated verifier test rejected conflicting nonblank CSV/XLSX
  answers with `SystemExit(1)` and left both editable table byte hashes
  unchanged. Nonblank human fields themselves were not treated as a failure.
  Its generated verification report was confined to the temporary directory.

## Exact annotation and baseline reproduction

Run with `C:/Users/laika_yan/miniconda3/envs/projectome/python.exe -B`.
The script compiles an immutable in-memory source snapshot and writes only
under automatically removed `TemporaryDirectory` locations.

```python
import hashlib, tempfile, types
from pathlib import Path
import pandas as pd

path = Path("D:/projectome_analysis/group_analysis/scripts/visual_review_20261002.py")
payload = path.read_bytes()
m = types.ModuleType("independent_phase1_snapshot")
m.__file__ = str(path)
exec(compile(payload, str(path), "exec"), m.__dict__)
print("source_snapshot_sha256", hashlib.sha256(payload).hexdigest())

with tempfile.TemporaryDirectory(prefix="projectome-phase1-audit-") as directory:
    csv = Path(directory) / "duplicate.csv"
    pd.DataFrame([
        {"UID": "252385|003.swc", "human_region": "IAL",
         "human_notes": "first decision"},
        {"UID": "252385|003.swc", "human_region": "IG",
         "human_notes": "second decision"},
    ]).to_csv(csv, index=False)
    print("duplicate UID result", m.load_human_annotations(csv))

with tempfile.TemporaryDirectory(prefix="projectome-phase1-audit-") as directory:
    root = Path(directory)
    (root / "tables").mkdir()
    (root / "manifest").mkdir()
    pd.DataFrame([{
        "SampleID": "252385", "NeuronID": "003.swc",
        "UID": "252385|003.swc", "image_source_status": "source_pending",
    }]).to_csv(root / "manifest/identity_manifest.csv", index=False)
    pd.DataFrame([{
        "UID": "252385|004.swc", "human_region": "IAL",
        "human_notes": "reviewed identity removed by manifest",
    }]).to_csv(root / "tables/correction_table.csv", index=False)
    frame = m.write_correction_table(root)
    print("orphan annotation preserved", (frame.UID == "252385|004.swc").any())

with tempfile.TemporaryDirectory(prefix="projectome-phase1-audit-") as directory:
    root = Path(directory)
    canonical = root / "canonical.txt"
    canonical.write_text("first")
    m.guard_canonical_baseline(root, files=(canonical,))
    saved_path = root / "manifest/canonical_hashes.json"
    baseline = saved_path.read_bytes()
    canonical.write_text("second")
    try:
        m.guard_canonical_baseline(root, files=(canonical,))
        print("canonical change blocked", False)
    except RuntimeError:
        print("canonical change blocked", True,
              "baseline bytes preserved", baseline == saved_path.read_bytes())

print("source_after_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
```

Actual annotation results in the tested snapshot were `IG / second decision`
for the duplicate UID, and `False` for orphan preservation. Canonical change
blocking and baseline-byte preservation both returned `True`.

## Exact isolated verifier reproduction

```python
import hashlib, json, tempfile, types
from pathlib import Path
import pandas as pd

path = Path("D:/projectome_analysis/group_analysis/scripts/visual_review_20261002.py")
payload = path.read_bytes()
m = types.ModuleType("independent_verifier_snapshot")
m.__file__ = str(path)
exec(compile(payload, str(path), "exec"), m.__dict__)
print("source_snapshot_sha256", hashlib.sha256(payload).hexdigest())

with tempfile.TemporaryDirectory(prefix="projectome-phase1-verify-") as directory:
    root = Path(directory)
    (root / "tables").mkdir()
    (root / "manifest").mkdir()
    item = {name: "" for name in list(m.HUMAN_FIELDS) + list(m.PANEL_FILES)}
    item.update(UID="252790|003.swc", SampleID="252790", NeuronID="003.swc",
                image_source_status="source_pending", production_status="source_pending",
                review_category="INS", coverage_status="")
    pd.DataFrame([dict(item, human_region="IAL")]).to_csv(
        root / "tables/correction_table.csv", index=False)
    pd.DataFrame([dict(item, human_region="IG")]).to_excel(
        root / "tables/correction_table.xlsx", index=False)
    pd.DataFrame([item]).to_csv(root / "manifest/identity_manifest.csv", index=False)
    m.canonical_hashes = lambda: {"dummy": "original"}
    (root / "manifest/canonical_hashes.json").write_text(json.dumps(m.canonical_hashes()))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / "tables").iterdir()}
    try:
        m.verify_outputs(root)
        print("conflicting answers rejected", False)
    except SystemExit as exc:
        print("conflicting answers rejected", exc.code == 1)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (root / "tables").iterdir()}
    print("editable table bytes unchanged", before == after)

print("source_after_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
```

Both rejection and editable-table preservation returned `True`. This isolates
annotation verification; it does not claim full production or anatomical
acceptance and does not rerun image loading, downloads or cohort rendering.
