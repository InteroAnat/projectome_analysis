"""Restore an already committed receiver selector in stale canonical Rmd."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'proposed_readable_metric_labels_20261009/group_analysis/R_analysis/v2_combined_primary_pipeline.R'
BASE=HERE/'proposed_readable_metric_labels_20261009/group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd'
OUT=HERE/'proposed_F6_established_workflow_sync_20261009'
TARGET='group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd'
PATCH=HERE/'proposed_F6_established_workflow_sync_20261009.patch'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def block(text):
    start=text.index('f6_df <- bind_rows(f6_rows) %>%')
    final=text.index('ggsave(file.path(OUT_FIGS, "F6_supplement_rank_based_receivers.png")',start)
    end=text.index('\n',text.index('width = 9.5, height = 6.0, dpi = 220)',final))+1
    return start,end,text[start:end]


if __name__=='__main__':
    before=BASE.read_text(encoding='utf-8')
    source=SOURCE.read_text(encoding='utf-8')
    begin,end,oldblock=block(before)
    _,_,fixed=block(source)
    assert 'pmin(' in oldblock and 'best_q < 0.10' in oldblock
    assert 'pmin(' not in fixed and 'q < 0.05' in fixed
    after=before[:begin]+fixed+before[end:]
    old='**Hypothesis.** **H6′:** BH within **family** (intra vs extra-insula) recovers **`Ig`, `LPal`** aligned with **`Ia/Id`** as the strongest asymmetric **receivers** in **`IDD5_balanced`** — **cross-validation** of the composition-LI narrative without assuming a single distance metric.'
    assert old in after
    after=after.replace(old,
      '**Scope.** This supplementary ranking uses the same sampled neurons as the composition contrast. Presence and magnitude are reported as separate test families; this is not independent cross-validation or animal-level inference.')
    old='The bar chart selects targets with `best_q < 0.05` in any region-restricted neuron stratum, takes the top 18, and labels each bar with the strongest `q`, the stratum that gave it, and the family.'
    assert old in after
    after=after.replace(old,
      'Following the established workflow in commit `d27c855b82a280989850d71f9f1eb994f70df58e`, create separate presence and magnitude rows and retain `q < 0.05` within each test family. Rank at most 18 targets per family and facet the display by test family. Do not take a minimum across presence/magnitude q-values. Selecting a best stratum for display does not establish correction across strata or independent animal replication.')
    old='**Outputs.** `F6_supplement_rank_based_receivers.png`, stats: `06_asymmetric_receivers_BH.csv` (all rows, with `p_pres_BH`, `p_mag_BH`, `best_q`, `direction`).'
    assert old in after
    after=after.replace(old,
      '**Outputs.** `F6_supplement_rank_based_receivers.png`; `06_asymmetric_receivers_BH.csv` keeps the separate adjusted p-value columns; `06_asymmetric_receivers_BH_long.csv` adds the established `test_family` and `q` representation. Old saved outputs are preserved until a separate versioned rerun.')
    begin=after.index('**How to read.** This run yields three bars:')
    end=after.index('\n\n',begin)
    after=after[:begin]+('**Historical selector readback.** Applying the established selector to the saved 306-neuron statistics gives presence targets `Ig@L6` and `LPal@L3`, and magnitude targets `Ia/Id@L6`, `Ig@L6` and `LPal@L3`. These are five family-specific display rows for three distinct targets, not a fresh 353-neuron result. The original p-values are unchanged; this audit does not establish a valid animal-level model.')+after[end:]
    destination=OUT/TARGET
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(after,encoding='utf-8',newline='')
    PATCH.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+TARGET,tofile='b/'+TARGET)),encoding='utf-8',newline='')
    git=subprocess.run(['git','-c','safe.directory=D:/projectome_analysis','show','d27c855b82a280989850d71f9f1eb994f70df58e','--','group_analysis/R_analysis/v2_combined_primary_pipeline.R'],cwd=ROOT,capture_output=True,check=True)
    receipt={'status':'staged_sync_of_established_workflow_not_applied','patch_sha256':sha(PATCH),
      'target':TARGET,'base_path':str(BASE.relative_to(ROOT)),'base_sha256':sha(BASE),
      'proposed_sha256':sha(destination),'fixed_R_source':str(SOURCE.relative_to(ROOT)),'fixed_R_sha256':sha(SOURCE),
      'recovered_F6_block_exactly_identical_to_existing_fixed_R':block(after)[2]==fixed,
      'prior_decision_commit':'d27c855b82a280989850d71f9f1eb994f70df58e','prior_commit_show_sha256':hashlib.sha256(git.stdout).hexdigest(),
      'supporting_anchors':['docs/project_note.md:238','notes/whole_insula_lr_continuation_plan_v2.md:15','notes/whole_insula_lr_continuation_plan_v2.md:88','notes/code_review_20260926.md:19','notes/code_review_20260926.md:20','notes/code_review_20260926.md:63'],
      'requires_first':'proposed_readable_metric_labels_20261009.patch','new_statistical_design':False,'inferential_pvalues_recomputed':False,'historical_outputs_changed':False,
      'scope':'Synchronize stale canonical Rmd with preexisting family-specific q<0.05 selector and plot block; animal-independent inference remains unsupported'}
    (HERE/'F6_established_workflow_staging_receipt_20261009.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'patch_sha256':receipt['patch_sha256'],'exact_fixed_block':receipt['recovered_F6_block_exactly_identical_to_existing_fixed_R']}))
