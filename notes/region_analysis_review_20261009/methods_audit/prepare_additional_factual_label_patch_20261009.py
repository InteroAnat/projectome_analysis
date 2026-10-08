"""Stage follow-up labels only; Rmd base is the previously reviewed v2 patch."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGED = HERE/'proposed_additional_factual_labels_20261009'
PATCH = HERE/'proposed_additional_factual_labels_20261009.patch'


def sha(data): return hashlib.sha256(data).hexdigest()


def replace(text, old, new, expected=1):
    assert text.count(old) == expected, (old, text.count(old), expected)
    return text.replace(old, new)


def tutorial(text):
    text = replace(text, '  name = "CONTRA",', '  name = "IPSI_DEMO_COPY",')
    text = replace(text, '  row_title = "Contralateral",',
                   '  row_title = "Duplicated ipsilateral example; not contralateral data",')
    text = replace(text, 'decorate_row_title("CONTRA", {',
                   'decorate_row_title("IPSI_DEMO_COPY", {', 2)
    text = replace(text, 'h2 <- build_projection_ht(mat_strength_ipsi_t, "CONTRA", "Contralateral", order = 2)',
                   'h2 <- build_projection_ht(mat_strength_ipsi_t, "IPSI_DEMO_COPY", "Duplicated ipsilateral example; not contralateral data", order = 2)')
    text = replace(text, 'ht_contra <- Heatmap(',
                   '# Demonstration only: this second panel duplicates the ipsilateral input.\nht_contra <- Heatmap(')
    return text


def comparison(text):
    text = replace(text, '**Status:** Comparison note for projectome laterality analysis',
        '**Status:** Historical schematic comparison; factual method labels corrected 2026-10-09\n\n'
        'The deposited Gou implementation distinguishes axon-length laterality from arbor-count laterality. '
        'The raw-length formula below describes one measurement, not every published arbor endpoint. '
        'Contra/ipsi describes a projection relative to its source hemisphere; it does not mean absolute right/left. '
        'In SQ3d, targets index hypotheses and neuron values enter the tests; animal independence is not established by target-wise BH correction.')
    text = replace(text, '**Key:** every statistical unit is a brain region (not a neuron).',
        '**Key:** each brain region defines a separate hypothesis; the tested observations are neuron values in L-source and R-source groups. Neurons nested within animals are not thereby independent animal replicates.')
    text = replace(text, '| **Formula** | `(contra − ipsi) / (contra + ipsi)` on raw lengths | Same formula on group-mean proportions |',
        '| **Formula** | `(contra − ipsi) / (contra + ipsi)` on raw lengths | `(mean_L − mean_R) / (mean_L + mean_R + eps)` on normalized group-mean profiles |')
    text = replace(text, '| **Statistical unit** | Neuron | Region |',
        '| **Test observations / hypothesis** | Neuron LI values / group contrast | Neuron normalized values / one hypothesis per target region |')
    text = replace(text, '"PT neurons are more right-lateralized than CT neurons"',
        '"Higher LI denotes relatively more contralateral than ipsilateral projection"')
    return text


def cohort_narrative(text):
    text = replace(text, '> ⚠️ **Interim dataset.** This pipeline analyses the **partial cohort currently\n'
        '> available (4 macaques, 306 neurons after harmonization)**. The granular Ig\n'
        '> stratum and the agranular L-side IAL stratum will continue to grow as more\n'
        '> samples are added; treat all per-region tests outside the IDD5+IDM balanced\n'
        '> stratum as descriptive until additional bilateral sampling is available.',
        '> **Historical outputs and current input.** The saved output narrative below refers\n'
        '> to the historical 306-neuron cohort from four SampleIDs. The workbook audited\n'
        '> on 2026-10-09 has 353 unique neurons across six SampleIDs (251637, 252383,\n'
        '> 252384, 252385, 252527, 252718). SampleIDs are not asserted to be independent\n'
        '> animals. No fresh 353-neuron result is established by the preserved 306-neuron\n'
        '> tables or captions. All current neuron-level tests remain exploratory pending\n'
        '> an agreed animal sampling model and exact-cohort/namespace/unit preflight.')
    text = replace(text, 'of 306 reconstructed insula neurons from 4 macaques.',
        'of selected reconstructed neurons. The preserved historical outputs used 306 neurons from four SampleIDs; the current audited workbook contains 353 neurons across six SampleIDs, without an established six-animal interpretation.')
    text = replace(text, '## 0.2 Cohort summary (this run)', '## 0.2 Historical 306-neuron cohort summary')
    text = replace(text, 'After harmonizing manual `IAL/IAPM/IDD5/IDM/IDV` labels with atlas labels via `Mapping_Rule` (sheet of `multi_monkey_INS_combined_harmonized.xlsx`):',
        'The following counts describe the historical output cohort. They are preserved, not substituted for the current 353-neuron workbook. Manual labels were harmonized with atlas labels through `Mapping_Rule` (sheet of `multi_monkey_INS_combined_harmonized.xlsx`):')
    text = replace(text, '**balanced** ✅ — primary inferential stratum',
        'L/R sampled — exploratory neuron contrast; not animal balanced')
    text = replace(text, 'reasonably balanced ✅', 'L/R sampled; not animal balanced')
    text = replace(text, '| **Combined (this run)** | **121** | **185** | **306** | 4 macaques (251637 + 252383 + 252384 + 252385) |',
        '| **Historical combined** | **121** | **185** | **306** | Four SampleIDs (251637 + 252383 + 252384 + 252385); no animal-replication claim |')
    text = replace(text, 'The **inferential anchor** is the `IDD5_plus_IDM_balanced` stratum (n_L = 81, n_R = 68 in the new harmonized table). All BH-corrected laterality claims must survive in this stratum — see [§0.5 strata strategy](#sec-strata).',
        'The historical region-restricted contrast is `IDD5_plus_IDM_balanced` (n_L = 81, n_R = 68 in the historical 306-neuron table). These are compatibility labels and historical counts, not animal balancing or independent replication — see [§0.5 strata strategy](#sec-strata).')
    text = replace(text, 'Where do the 306 somata sit on the insula flatmap, by monkey × side?',
        'Where do the selected somata sit on the insula flatmap, by SampleID × side?', 2)
    text = replace(text, 'Re-build a 306×306 FNT distance matrix',
        'Build the exact-cohort square FNT distance matrix (historical run: 306×306)')
    text = replace(text, '> ⚠️ **Interim cohort.** This run analyses the partial dataset currently available (4 macaques, 306 neurons after harmonization). Numbers will change as more samples are added. Always cite this README\'s cohort table next to any figure quoted in a manuscript draft.',
        '> **Historical cohort warning.** The preserved quoted results refer to 306 neurons from four SampleIDs. The workbook audited 2026-10-09 has 353 neurons from six SampleIDs, not an established six-animal cohort. A fresh exact-cohort/namespace/unit preflight and versioned rerun are required before quoting current results.')
    text = replace(text, 'across 4 macaques after harmonization; primary inferential stratum is',
        'after harmonization; independent animal replication is unestablished; exploratory region-restricted stratum is')
    text = replace(text, 'The `All_insula` row pools all 306 neurons.',
        'The `All_insula` row pools the selected neurons; preserved quoted values refer to the historical 306-neuron cohort.')
    text = replace(text, '**Partial cohort.** This run uses 306 neurons across 4 macaques;',
        '**Historical cohort.** Preserved outputs use 306 neurons from four SampleIDs; the workbook audited 2026-10-09 has 353 neurons from six SampleIDs without an established six-animal interpretation;')
    return text


if __name__ == '__main__':
    paths = ['R_analysis/Projectome_Analysis_Tutorial.Rmd',
             'R_analysis/scripts/Projectome_Analysis_Tutorial.Rmd',
             'R_analysis/scripts/Projectome_Tutorials/Projectome_Analysis_Tutorial.Rmd',
             'notes/hemispheric_asymmetry_methods_comparison.md',
             'group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd']
    assert PATCH.parent.resolve() == HERE.resolve() and STAGED.parent.resolve() == HERE.resolve()
    records, chunks = [], []
    for path in paths:
        if path.startswith('group_analysis/'):
            base = HERE/'proposed_label_corrections_v2'/path
        else:
            base = HERE/'additional_factual_labels_base_20261009'/path
            if not base.exists():
                base.parent.mkdir(parents=True,exist_ok=True)
                base.write_bytes((ROOT/path).read_bytes())
        before = base.read_text(encoding='utf-8').replace('\r\n','\n')
        after = (cohort_narrative(before) if path.startswith('group_analysis/') else
                 comparison(before) if path.startswith('notes/') else tutorial(before))
        destination = STAGED/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(after, encoding='utf-8', newline='')
        chunks.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                      fromfile='a/'+path,tofile='b/'+path))
        records.append({'path':path,'base_path':str(base.relative_to(ROOT)),
                        'base_sha256':sha(base.read_bytes()),'proposed_sha256':sha(destination.read_bytes())})
    PATCH.write_text(''.join(chunks),encoding='utf-8',newline='')
    receipt = {'status':'staged_only_not_applied','patch_sha256':sha(PATCH.read_bytes()),
               'requires_first':'proposed_methods_label_corrections_v2.patch',
               'numeric_operations_changed':False,'historical_outputs_changed':False,
               'files':records,'cohort_readback':{'unique_UIDs':353,'SampleIDs':{'251637':260,'252385':47,'252527':23,'252718':13,'252383':5,'252384':5},'animal_count_asserted':False}}
    (HERE/'additional_factual_labels_staging_receipt_20261009.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(records),'patch_sha256':receipt['patch_sha256']}))
