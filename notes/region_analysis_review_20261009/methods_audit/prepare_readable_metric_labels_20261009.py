"""Stage readable metric labels; calculations, columns and filenames preserved."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'proposed_readable_metric_labels_20261009'
BASE = HERE/'readable_metric_labels_base_20261009'
PATCH = HERE/'proposed_readable_metric_labels_20261009.patch'

CONTRACT = '''Metric names (2026-10-09): retained length is the legacy reconstruction
measurement in its declared source units, with compartment/terminal-target
semantics unverified. Display strength = log10(retained length + 1).
A log-scaled length share divides that strength by the sum over selected
features; hybrid L3/L6 features overlap and are not a raw-length budget.
Summary Laterality_Index = Contra/(Ipsi+Contra), range 0..1 (0 ipsi, 1 contra).
Individual Ibias = (Contra-Ipsi)/(Contra+Ipsi), range -1..1 (+1 contra).
Source-group LI = (mean_L-mean_R)/(mean_L+mean_R+epsilon), positive for a
higher LEFT-source neuron mean; this is not an ipsi/contra or right/left target
hemisphere index. Zero-denominator individual balances are unavailable.
Names, columns, thresholds, stratum IDs and output filenames are preserved.'''

SUMMARY_NOTICE = '''Imported Summary metric: the repository producer defines Laterality_Index
as Contra/(Ipsi+Contra), range 0..1 (0 ipsilateral, 1 contralateral).
These labels assume that producer contract; verify an older workbook's source
before reuse. This is retained reconstruction length, not accepted terminal
arbors/synapses. It differs from signed Ibias and from source-group LI.'''


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def notice(text, message, rmd):
    if not rmd:
        return '\n'.join('# '+line for line in message.splitlines())+'\n\n'+text
    lines=text.splitlines(True)
    assert lines[0].strip()=='---'
    end=next(i for i in range(1,len(lines)) if lines[i].strip()=='---')
    return ''.join(lines[:end+1])+'\n\n'+ '\n'.join('> '+line for line in message.splitlines())+'\n\n'+''.join(lines[end+1:])


def active(text, rmd):
    replacements = [
      ('normalized ipsi log-strength shares','ipsilateral shares of log-scaled length'),
      ('normalized ipsi log-strength share','ipsilateral share of log-scaled length'),
      ('Normalized ipsi log-strength shares','Ipsilateral shares of log-scaled length'),
      ('normalized log-strength share','share of log-scaled length'),
      ('log-strength share','log-scaled length share'),
      ('Mean prop\\n(L+R)','Mean log-scaled length share\\n(L+R)'),
      ('mean prop\\nipsi','Mean ipsilateral\\nlog-scaled length share'),
      ('L6 row-normalized mean prop.','Mean ipsilateral log-scaled length shares at L6.'),
      ('Hybrid p_combo (L3 extrinsic + L6 intra-insula); cell = mean summed prop per domain. n_total annotated.',
       'Mean shares of log-scaled length, summed by domain; overlapping L3/L6 features; n_total shown.'),
      ('Per-neuron proportion of ipsi projection going to caudal_OFC.',
       'Per-neuron ipsilateral share of log-scaled retained length assigned to caudal_OFC.'),
      ('Each cell = mean fraction of ipsi projection from source soma sub-region (rows) to target insula sub-region (cols).',
       'Each cell = mean ipsilateral share of log-scaled retained length from source (rows) to target (columns).'),
      ('Hemispheric bias (Ibias)','Retained length balance (Ibias)\\n-1 ipsi; +1 contra'),
      ('Per-neuron continuous legacy voxel-length hemispheric bias','Individual ipsi/contra balance of retained length'),
      ('Laterality index (L−R)/(L+R)','Left vs right SOURCE contrast\\n(mean_L-mean_R)/(mean_L+mean_R+eps)'),
      ('Laterality index (L-R)/(L+R)','Left vs right SOURCE contrast\\n(mean_L-mean_R)/(mean_L+mean_R+eps)'),
      ('y = "Laterality index"','y = "Left vs right SOURCE contrast"'),
      ('name = "Dir"','name = "Higher source-group mean"'),
      ('F6. Target-level LI distributions by stratum family','F6. Left vs right source contrast across targets'),
      ('F6. Bilateral-receiving targets (preferred LI interpretation)','F6. Targets with evidence from both source groups'),
      ('F6 supplement. One-sided / extreme receipt (QC; not symmetric sampling)',
       'F6 supplement. Evidence from one source group only (sampling QC)'),
      ('on projection STRENGTH','on retained regional lengths'),
      ('if both zero we treat as bias = -1 (all ipsi by default)',
       'if both totals are zero, Ibias is unavailable (NA)'),
      ('**total axon length** (L6 length sheets)','retained regional length (L6 length sheets; source units)'),
      ('on **axon length** vs soma AP','on retained regional length vs soma AP'),
      ('on axon length |','on retained regional length |'),
      ('**What.** Target-level **composition LI**', '**What.** Target-level **left vs right source contrast (LI)**'),
      ('on group-mean `p_combo` proportions','on group means of normalized log-scaled retained length shares'),
      ('## F6 — projection laterality index','## F6 — left vs right source contrast'),
      ('## 2.6 Figure 6 — Composition laterality index','## 2.6 Figure 6 — Left vs right source contrast'),
      ('## 2.5 Figure 5 — Ibias (axon length)','## 2.5 Figure 5 — Individual retained length balance (Ibias)'),
      ('F6 (laterality index)','F6 (left vs right source contrast)'),
      ('where does each insula neuron sit on the ipsi-vs-contra axis when measured by **total axon length** (not by branch-tip count)?',
       'where does each neuron sit on the ipsi-vs-contra axis for retained reconstruction length (not candidate endpoint count or segmented arbor count)?'),
      ('Replicating Gou et al. 2025 *Cell*\'s per-neuron hemispheric bias index,',
       'Using a signed retained-length balance, distinct from Gou et al.\'s segmented-arbor measurements,'),
      ('a few outliers in IAL near +218 µm (anterior)',
       'a few historical IAL outliers near NMT Y voxel index 218 (not 218 micrometres)'),
    ]
    for before,after in replacements:
        text=text.replace(before,after)
    if rmd:
        old='| **LI** | Laterality index `(mean_L − mean_R) / (mean_L + mean_R + ε)`; ±1 = exclusive, 0 = symmetric. |'
        new='| **LI** | Left vs right **source-group** contrast `(mean_L − mean_R) / (mean_L + mean_R + ε)`; positive means a higher left-source mean. Zero with both means zero does not establish bilateral innervation. |'
        assert old in text
        text=text.replace(old,new)
        old_hypothesis = '**Hypothesis.** **H5:** Insula **Ibias ≈ −1** for most neurons (**purely ipsilateral length**) — predicting **sparse callosal output** versus PFC ITc in Gou et al. (same metric, different anatomical nucleus). Biological **positive finding** (structure of insula efferents), not a “failed” L vs R test.'
        assert old_hypothesis in text
        text=text.replace(old_hypothesis,
          '**Historical description.** **Ibias ≈ −1** means predominantly ipsilateral retained length under the legacy measurement. It does not by itself establish sparse biological callosal output or replicate a segmented-arbor measurement.')
        start='> ⚠️ **Caveat — insula has sparse contra projection.**'
        if start in text:
            begin=text.index(start)
            end=text.index('\n\n',begin)
            text=text[:begin]+('> **Measurement caveat.** The historical tables reported only ~2% of neurons with a nonzero contralateral retained-length entry, so many Ibias values lie near -1. This is sampled reconstruction evidence under unverified compartment, terminal-target and registration semantics. It is not acceptance of biological absence, sparse callosal anatomy or a current-cohort result. Soma_NII_Y is an index coordinate; an index near 218 is not a distance of 218 micrometres.')+text[end:]
    return notice(text,CONTRACT,rmd)


def summary_plot(text, rmd):
    text=text.replace('Laterality Index Distribution','Contralateral retained length share')
    text=text.replace('Laterality Index by Neuron Type','Contralateral retained length share by neuron type')
    text=text.replace('1 = Purely Ipsilateral, -1 = Purely Contralateral',
                      'Contra/(Ipsi+Contra): 0 ipsilateral, 1 contralateral; unavailable if total is zero')
    text=text.replace('"Laterality Index"','"Contralateral retained length share (0 to 1)"')
    text=text.replace('# 侧向性指数分布 (1=纯同侧, -1=纯对侧)',
                      '# Imported Summary: Contra/(Ipsi+Contra), 0=ipsi and 1=contra.')
    return notice(text,SUMMARY_NOTICE,rmd)


def novel(text):
    text=text.replace('x = "Length Asymmetry Index"','x = "Log-scaled length balance (ipsi-contra)/(ipsi+contra+eps)"')
    text=text.replace('y = "Target Count Asymmetry Index"','y = "Target-count balance (ipsi-contra)/(ipsi+contra+eps)"')
    text=text.replace('title = "Asymmetry Score by Neuron Type"',
                      'title = "Average of log-scaled length and target-count balances"')
    text=text.replace('y = "Asymmetry Score (-1 to 1)"',
                      'y = "Mean balance: positive ipsi; negative contra"')
    explanation = ('Historical tutorial metrics: Length_Asymmetry actually sums Projection_Strength\n'
      '(log10(retained length + 1)), then computes (ipsi-contra)/(ipsi+contra+eps).\n'
      'Count_Asymmetry uses counts of targets with positive strength. Asymmetry_Score\n'
      'averages those two balances. Their positive sign means ipsilateral, the\n'
      'opposite sign to Ibias. Imported Summary Laterality_Index is a separate\n'
      '0..1 contralateral share; abs(Laterality_Index) is not a signed bias magnitude.\n'
      'Calculations and historical variable IDs are preserved; no biological\n'
      'terminal-arbor or independent-animal acceptance follows from this tutorial.')
    return notice(text,explanation,True)


if __name__ == '__main__':
    active_names=['v2_combined_primary_pipeline.R','v2_combined_primary_pipeline.Rmd',
      'combined_lr_primary_analysis.R','functional_hubs_analysis.R','functional_hubs_L6.R',
      'intra_insula_connectivity.R','improved_panel_figures.R']
    summary_paths=['R_analysis/Projectome_Analysis_Tutorial.Rmd',
      'R_analysis/scripts/Projectome_Analysis_Tutorial.Rmd',
      'R_analysis/scripts/Projectome_Tutorials/Projectome_Analysis_Tutorial.Rmd',
      'R_analysis/01_针对251637数据的完整分析.R',
      'R_analysis/tutorials/01_针对251637数据的完整分析.R',
      'R_analysis/tables/251637_single_neuron_analysis.Rmd']
    paths=['group_analysis/R_analysis/'+name for name in active_names]+summary_paths+['R_analysis/Novel_Projectome_Analysis_Tutorial.Rmd']
    records,chunks=[],[]
    for path in paths:
        base=BASE/path
        if not base.exists():
            base.parent.mkdir(parents=True,exist_ok=True)
            base.write_bytes((ROOT/path).read_bytes())
        before=base.read_text(encoding='utf-8-sig').replace('\r\n','\n')
        rmd=path.lower().endswith('.rmd')
        after=(active(before,rmd) if path.startswith('group_analysis/') else
               novel(before) if 'Novel_' in path else summary_plot(before,rmd))
        destination=OUT/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(after,encoding='utf-8',newline='')
        chunks.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
        records.append({'path':path,'base_path':str(base.relative_to(ROOT)),
                        'base_sha256':sha(base),'proposed_sha256':sha(destination)})
    PATCH.write_text(''.join(chunks),encoding='utf-8',newline='')
    receipt={'status':'staged_only_not_applied','files':records,'patch_sha256':sha(PATCH),
      'numeric_operations_changed':False,'column_and_stratum_IDs_changed':False,'output_filenames_changed':False,
      'scope':'7 active group R/Rmd consumers;6 direct legacy Summary-metric plot consumers;1 historical strength-balance tutorial',
      'excluded_entrypoints':'Deprecated multi_monkey_lr_analysis.R archive replay; launcher and package checker have no metric plots',
      'remaining_historical_incompatibility':'Chinese full Rmd initial classification assumes a signed imported index but later reuses its name for a locally computed opposite-sign strength balance; no classification recalculated here'}
    (HERE/'readable_metric_labels_staging_receipt_20261009.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(records),'patch_sha256':receipt['patch_sha256']}))
