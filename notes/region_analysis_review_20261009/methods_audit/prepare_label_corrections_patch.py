"""Stage factual labels only; never write production or historical outputs."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGED = OUT / "proposed_label_corrections"
PATCH = OUT / "proposed_methods_label_corrections.patch"

TARGETS = ["group_analysis/R_analysis/" + name for name in (
    "v2_combined_primary_pipeline.R", "v2_combined_primary_pipeline.Rmd",
    "combined_lr_primary_analysis.R", "functional_hubs_analysis.R",
    "functional_hubs_L6.R", "intra_insula_connectivity.R", "improved_panel_figures.R")]
WARNING = """
## Historical-results methods warning — 2026-10-09

The earlier results in this report are preserved as historical observations.
They are not newly accepted anatomical or animal-population findings. The
independent audit and numerical receipts are in
`notes/region_analysis_review_20261009/methods_audit/`.

- The reported 58% intra-insula value is a mean share of **normalized
  log10(legacy regional voxel length + 1)**, not an axonal-length budget.
  A diagnostic untransformed-length share for the same 306 IDs is about 64%;
  it is not a replacement biological result. Those legacy lengths retain
  unverified compartment, terminal-target and coordinate semantics.
- Gradient `slope_p` values are ordinary OLS t-test p-values, not permutation
  p-values. Neuron-level tests and their BH adjustments do not establish
  independent animal replication or correct within-animal dependence.
- The L3/L6 hybrid retains L3 ancestors alongside L6 descendants. It is an
  overlapping multiscale transformed-strength profile, not an exclusive
  anatomical partition. Historical stripped Pi labels also have a known
  cortical/subcortical namespace ambiguity.
- The historical output tables use 306 neurons; the current harmonized
  workbook inspected on 2026-10-09 contains 353 common valid IDs. Old tables
  and figures remain version-specific. Their existence/QC flags do not
  establish fresh input-to-output lineage.
- Stratum identifiers ending `_balanced` are retained compatibility names
  for region restrictions. They implement neither animal balancing nor
  independent replication. LOSO drops SampleID; an animal interpretation
  requires a verified registry relationship for that run.

No numerical result, p-value, cohort, original label or historical figure was
changed by this warning. Candidate graph endpoints, legacy regional lengths
and image-reviewed terminal arbors remain distinct measurements. A separate
versioned analysis and agreed sampling design are required for new claims.
"""

NOTICE = """
> **Methods audit (2026-10-09).** Numeric calculations and legacy column/stratum
> identifiers are retained. `prop` denotes normalized log-strength composition,
> not raw axonal-length fraction. The L3/L6 hybrid has overlapping ancestors and
> descendants. OLS slope p-values are ordinary t-test p-values. `_balanced`
> names denote region restrictions, not animal balancing. Historical 306-neuron
> narratives do not describe every future input workbook; no new anatomical,
> terminal-field or animal-population acceptance follows from these outputs.

"""

REPLACEMENTS = [
    ("mean ipsi projection proportion (Wilcoxon)", "mean normalized ipsi log-strength share (neuron-level Wilcoxon; animal dependence unmodeled)"),
    ("OLS slope of target ~ soma_pos with permutation p", "OLS slope of normalized log-strength share ~ soma_pos; ordinary t-test p, animal dependence unmodeled"),
    ('"primary inferential"', '"exploratory neuron contrast"'),
    ('"replication",', '"additional region contrast; not independent replication",'),
    ("only L/R-balanced anatomical stratum", "L/R-sampled region restriction; not animal balancing"),
    ("only L/R-balanced anatomical strata", "L/R-sampled region restrictions; not animal balancing"),
    ("balanced strata", "region-restricted neuron strata"),
    ("balanced stratum", "region-restricted neuron stratum"),
    ("Leave-one-monkey-out", "Leave-one-SampleID-out"),
    ("leave-one-monkey-out", "leave-one-SampleID-out"),
    ("Leave-one-**SampleID**-out", "Leave-one-**SampleID**-out"),
    ("Mean prop\\n(L+R)", "Mean log-strength share\\n(L+R)"),
    ("mean prop\\n(L+R)", "mean log-strength share\\n(L+R)"),
    ("Mean\\nprop", "Mean\\nlog-strength share"),
    ("Projection proportion to target (row-normalized L6)", "Normalized log-strength share to target (L6)"),
    ("Row-normalized ipsi profile: L3 extrinsic + L6 intra-insula; cell = summed target prop per domain.",
     "Normalized ipsi log-strength shares; overlapping L3/L6 features; cells sum assigned feature shares."),
    ("on total L6 axon length", "on legacy L6 regional voxel lengths (unverified compartments/terminal-target selection)"),
    ("on total axon length", "on legacy regional voxel lengths (unverified compartments/terminal-target selection)"),
    ("on axon length (Gou-style)", "on legacy regional voxel lengths; not segmented-arbor replication"),
    ("continuous hemispheric bias (Gou-style)", "continuous legacy voxel-length hemispheric bias"),
    ("58% of axonal budget", "58% of normalized log-strength composition (historical cohort; not a raw-length budget)"),
    ("average insula axonal budget", "average normalized log-strength profile"),
    ("sub-region's axonal budget split", "sub-region's normalized log-strength profile map"),
    ("domain mass", "assigned transformed-strength share"),
    ("Domain mass", "Assigned transformed-strength share"),
    ("row-normalized proportions per neuron", "row-normalized log-strength shares per neuron; overlapping spatial features"),
    ("Row-normalized proportions per neuron", "Row-normalized log-strength shares per neuron; overlapping spatial features"),
    ("Mean summed target proportion", "Mean summed normalized log-strength share"),
    ("Mean L6 proportion", "Mean normalized L6 log-strength share"),
    ("L6 row-normalized proportion", "Normalized L6 log-strength share"),
    ("mean ipsi proportion", "mean normalized ipsi log-strength share"),
]


def code_chunks(text, is_rmd):
    if not is_rmd:
        return text
    return "\n".join(re.findall(r"(?ms)^```\{r[^\n]*\}\n(.*?)^```\s*$", text))


def nontext_code(text, is_rmd):
    """Conservative lexer: compare every R token outside comments/literals."""
    text = code_chunks(text, is_rmd)
    token = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\#[^\n]*''')
    # Literal contents may change labels, while their positions remain bound.
    return re.sub(r"\s+", "", token.sub(lambda m: "" if m.group()[0] == "#" else "<LITERAL>", text))


if STAGED.exists() or PATCH.exists():
    raise FileExistsError("Preserve the existing proposed patch; use a distinct version for revisions")
STAGED.mkdir()
changes, diffs = [], []
for relative in TARGETS:
    original = (ROOT / relative).read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    normalized = text.replace("\r\n", "\n")
    proposed = normalized
    applied = []
    for before, after in REPLACEMENTS:
        count = proposed.count(before)
        if count and before != after:
            proposed = proposed.replace(before, after)
            applied.append({"before": before, "after": after, "occurrences": count})
    if relative.endswith("v2_combined_primary_pipeline.Rmd"):
        proposed = proposed.replace("# 0 · Overview\n", "# 0 · Overview\n" + NOTICE, 1)
        old_policy = next(line for line in proposed.splitlines() if line.startswith("**`p_combo` policy.** Build raw count matrices"))
        new_policy = ("**`p_combo` policy.** `m_l3`, `m_l6` and `m_l6_contra` contain log10(legacy regional voxel length + 1); "
            "`m_len_l6` and `m_len_l6_contra` contain the corresponding untransformed legacy voxel lengths. "
            "`p_l3` and `p_l6` are row-normalized log-strength shares. The hybrid retains L3 ancestors and adds L6 insular leaves, "
            "so its spatial features overlap rather than forming an exclusive partition. `@L3`/`@L6` suffixes disclose feature levels. "
            "`p_combo` normalizes this overlapping transformed-strength profile once per neuron; it is not a raw-length or terminal-arbor budget.")
        proposed = proposed.replace(old_policy, new_policy, 1)
    elif relative.endswith("v2_combined_primary_pipeline.R"):
        proposed = ("# Methods audit 2026-10-09: labels corrected; numerical operations unchanged.\n"
                    "# prop = normalized log-strength share; hybrid L3/L6 features overlap.\n"
                    "# _balanced IDs denote region restrictions, not animal balancing.\n"
                    "# OLS slope p-values are ordinary t-test p-values; animal dependence is unmodeled.\n" + proposed)
    if nontext_code(normalized, relative.endswith(".Rmd")) != nontext_code(proposed, relative.endswith(".Rmd")):
        raise AssertionError(f"Nontext R code changed: {relative}")
    if proposed != normalized:
        destination = STAGED / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = proposed.replace("\n", newline).encode("utf-8")
        destination.write_bytes(payload)
        diffs.extend(difflib.unified_diff(normalized.splitlines(keepends=True), proposed.splitlines(keepends=True),
            fromfile="a/" + relative, tofile="b/" + relative))
        changes.append({"path": relative, "original_sha256": hashlib.sha256(original).hexdigest(),
            "proposed_sha256": hashlib.sha256(payload).hexdigest(), "nontext_R_tokens_identical": True,
            "numeric_operations_changed": False, "replacement_receipts": applied})

relative = "notes/LR_insula_analysis_review.md"
original = (ROOT / relative).read_bytes()
newline = "\r\n" if b"\r\n" in original else "\n"
payload = original + WARNING.replace("\n", newline).encode("utf-8")
destination = STAGED / relative
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(payload)
assert payload.startswith(original)
diffs.extend(difflib.unified_diff(original.decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True),
    payload.decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True), fromfile="a/"+relative, tofile="b/"+relative))
changes.append({"path": relative, "original_sha256": hashlib.sha256(original).hexdigest(),
    "proposed_sha256": hashlib.sha256(payload).hexdigest(), "append_only_original_bytes_preserved": True})
PATCH.write_text("".join(diffs), encoding="utf-8", newline="\n")
validation = {"status": "staged_only_not_applied", "created_utc": datetime.now(timezone.utc).isoformat(),
    "patch_path": str(PATCH.relative_to(ROOT)), "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
    "changed_files": changes, "production_inputs_unchanged": all(
        hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["original_sha256"] for item in changes),
    "historical_output_files_in_patch": [], "numeric_outputs_regenerated": False,
    "validation_limit": "Static R token comparison; no R execution or inferential redesign. String changes require human review of intended labels/specifications."}
(OUT / "proposed_label_patch_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
print(json.dumps({"status": validation["status"], "files": len(changes), "patch_sha256": validation["patch_sha256"]}, indent=2))
