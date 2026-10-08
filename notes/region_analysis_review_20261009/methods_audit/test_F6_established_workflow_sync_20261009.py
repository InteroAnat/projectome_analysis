"""Independent saved-table selector regression; never recompute inferential p."""
import ast
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=Path(os.environ.get('PROJECTOME_F6_SELECTOR_ROOT',str(HERE/'proposed_F6_established_workflow_sync_20261009')))
TARGET='group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd'
SAVED=ROOT/'group_analysis/R_analysis/outputs/combined_primary_v2/stats/06_asymmetric_receivers_BH.csv'
RSCRIPT='C:/Program Files/R/R-4.6.1/bin/Rscript.exe'
STRATA={'IDD5_plus_IDM_balanced','IDD5_balanced','IDM_balanced'}
tree=ast.parse((HERE/'prepare_label_corrections_patch_v2.py').read_text())
namespace={'re':re}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'token-validator','exec'),namespace)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def block(text):
    begin=text.index('f6_df <- bind_rows(f6_rows) %>%')
    final=text.index('ggsave(file.path(OUT_FIGS, "F6_supplement_rank_based_receivers.png")',begin)
    end=text.index('\n',text.index('width = 9.5, height = 6.0, dpi = 220)',final))+1
    return begin,end,text[begin:end]


def oracle(rows):
    selected={}
    for family,column in [('presence','p_pres_BH'),('magnitude','p_mag_BH')]:
        by_target={}
        for row in rows:
            if row['stratum'] not in STRATA or row[column] in ('','NA','NaN'):
                continue
            q=Decimal(row[column])
            if q>=Decimal('0.05'): continue
            by_target.setdefault(row['target_id'],[]).append((q,row))
        ranking=[]
        for target,group in by_target.items():
            best=min(group,key=lambda pair:pair[0])
            effect=max(abs(Decimal(row['cliffs_delta'])) for _,row in group)
            ranking.append((best[0],-effect,target,best[1],effect))
        for q,_,target,row,effect in sorted(ranking)[:18]:
            selected[family,target]={'best_q':q,'best_stratum':row['stratum'],
                'direction_at_best':row['direction'],'abs_delta':effect,'spatial_family':row['family']}
    return selected


def run_existing_R_selector(rows, columns):
    text=(OUT/TARGET).read_text(encoding='utf-8')
    begin=text.index('balanced_strata <- c(',text.index('f6_df <- bind_rows(f6_rows)'))
    end=text.index('if (nrow(top_targets))',begin)
    selector=text[begin:end]
    assert 'p.adjust' not in selector and 'fisher.test' not in selector and 'wilcox.test' not in selector
    with tempfile.TemporaryDirectory(prefix='F6-selector-only-',dir=HERE) as directory:
        temp=Path(directory)
        with (temp/'input.csv').open('w',encoding='utf-8',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        code=('suppressPackageStartupMessages(library(dplyr))\n'
              'args <- commandArgs(trailingOnly=TRUE)\n'
              'f6_df <- read.csv(args[1],stringsAsFactors=FALSE)\n'
              'OUT_STATS <- args[2]\n'+selector+
              'write.csv(top_targets,file.path(OUT_STATS,"selected.csv"),row.names=FALSE)\n')
        driver=temp/'selector_only.R';driver.write_text(code,encoding='utf-8')
        result=subprocess.run([RSCRIPT,'--vanilla',str(driver),str(temp/'input.csv'),str(temp)],
                              cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        assert result.returncode==0,(result.stdout,result.stderr)
        with (temp/'selected.csv').open(encoding='utf-8',newline='') as inp:
            selected=list(csv.DictReader(inp))
    return selected


class EstablishedF6Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hash=sha(SAVED)
        with SAVED.open(encoding='utf-8',newline='') as inp:
            reader=csv.DictReader(inp);cls.rows=list(reader);cls.columns=reader.fieldnames
        cls.expected=oracle(cls.rows)
        cls.actual=run_existing_R_selector(cls.rows,cls.columns)

    def compare(self, expected, actual):
        self.assertEqual(set(expected),{(row['test_family'],row['target_id']) for row in actual})
        for row in actual:
            wanted=expected[row['test_family'],row['target_id']]
            for key in ('best_stratum','direction_at_best','spatial_family'):
                self.assertEqual(row[key],wanted[key])
            for key in ('best_q','abs_delta'):
                self.assertAlmostEqual(float(row[key]),float(wanted[key]),places=14)

    def test_saved_family_specific_targets_match_independent_decimal_oracle(self):
        self.compare(self.expected,self.actual)
        self.assertEqual({target for family,target in self.expected if family=='presence'}, {'Ig@L6','LPal@L3'})
        self.assertEqual({target for family,target in self.expected if family=='magnitude'}, {'Ia/Id@L6','Ig@L6','LPal@L3'})
        self.assertEqual(sha(SAVED),self.original_hash)

    def test_strict_threshold_and_no_cross_family_promotion(self):
        template=self.rows[0]
        rows=[]
        for target,pres,mag in [('exact','0.05','0.5'),('presence_only','0.01','0.07'),('magnitude_only','0.08','0.0499999999')]:
            row=dict(template,target_id=target,stratum='IDD5_balanced',p_pres_BH=pres,p_mag_BH=mag)
            rows.append(row)
        wanted=oracle(rows)
        actual=run_existing_R_selector(rows,self.columns)
        self.compare(wanted,actual)
        self.assertEqual(set(wanted),{('presence','presence_only'),('magnitude','magnitude_only')})

    def test_restored_block_matches_prior_commit_and_existing_fixed_R(self):
        receipt=json.loads((HERE/'F6_established_workflow_staging_receipt_20261009.json').read_text())
        proposed=(OUT/TARGET).read_text(encoding='utf-8')
        fixed=(ROOT/receipt['fixed_R_source']).read_text(encoding='utf-8')
        self.assertEqual(block(proposed)[2],block(fixed)[2])
        committed=subprocess.run(['git','-c','safe.directory=D:/projectome_analysis','show',
            'd27c855b82a280989850d71f9f1eb994f70df58e:group_analysis/R_analysis/v2_combined_primary_pipeline.R'],
            cwd=ROOT,capture_output=True,check=True).stdout.decode('utf-8')
        self.assertEqual(namespace['nontext_code'](block(proposed)[2],False),namespace['nontext_code'](block(committed)[2],False))
        base=(ROOT/receipt['base_path']).read_text(encoding='utf-8')
        begin,end,_=block(proposed)
        neutralized=proposed[:begin]+block(base)[2]+proposed[end:]
        self.assertEqual(namespace['nontext_code'](neutralized,True),namespace['nontext_code'](base,True))

    def test_old_selector_is_different_without_recalculating_pvalues(self):
        old={row['target_id'] for row in self.rows if row['stratum'] in STRATA and
             min((Decimal(row[column]) for column in ('p_pres_BH','p_mag_BH')
                  if row[column] not in ('','NA','NaN')),default=Decimal('Infinity'))<Decimal('0.10')}
        self.assertEqual(old,{'Ia/Id@L6','Ig@L6','LPal@L3','Rh@L3','TE@L3','vlPFC@L3'})
        self.assertEqual(len(self.actual),5)


if __name__=='__main__':
    result=unittest.main(verbosity=2,exit=False).result
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests_run':result.testsRun,
      'failures':len(result.failures),'errors':len(result.errors),'saved_table_sha256':sha(SAVED),
      'saved_table_path':str(SAVED.relative_to(ROOT)),'new_inferential_pvalues':False,
      'selected_rows':EstablishedF6Workflow.actual,'historical_cohort':306,
      'animal_level_inference_supported':False,'Rscript':RSCRIPT}
    receipt.update(selector_source_path=str((OUT/TARGET).resolve()),
      selector_source_sha256=sha(OUT/TARGET),live=OUT.resolve()==ROOT.resolve(),
      Rscript_sha256=sha(Path(RSCRIPT)))
    receipt_name=os.environ.get('PROJECTOME_F6_SELECTOR_RECEIPT','F6_established_workflow_selector_regression_20261009.json')
    assert Path(receipt_name).name==receipt_name and receipt_name.endswith('.json')
    (HERE/receipt_name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
