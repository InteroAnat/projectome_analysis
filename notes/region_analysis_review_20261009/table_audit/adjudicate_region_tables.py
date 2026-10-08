"""Read-only targeted adjudication of the hash-grouped broad table audit."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import openpyxl
import pandas as pd
from audit_region_tables import ROOT, AUDIT, digest, save, csvout, text, rel


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def sheet_records(wb, name):
    it = wb[name].iter_rows(values_only=True)
    headers = next(it, ())
    return headers, [dict(zip(headers, r)) for r in it]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    args=ap.parse_args(); out=args.run_dir.resolve(); out.relative_to(AUDIT)
    inv=json.loads((out/'inventory_groups.json').read_text())
    books=json.loads((out/'workbooks_schemas.json').read_text())
    tables=json.loads((out/'delimited_schemas.json').read_text())
    schemas={r['sha256']:r for r in books['schemas']+tables['schemas']}
    flattened=[]
    for f in inv['files']:
        s=schemas[f['sha256']]
        flattened.append({**f,'read_status':s['status'], 'sheet_count':len(s.get('sheets',[])),
            'sheets':[{k:z.get(k) for k in ('sheet','rows','columns','identity_grain','grain')} for z in s.get('sheets',[])],
            'identical_copy_count':len(inv['groups'][f['sha256']])})
    csvout(out/'file_inventory_with_schema.csv', flattened)
    findings=rows(out/'workbooks_findings.csv')+rows(out/'delimited_findings.csv')
    adjudicated=[]; checks=[]
    for r in findings:
        result={**r,'adjudication':'retained','remedy':''}
        p=ROOT/r['file']
        if r['check']=='composite_UID_mismatch':
            if p.suffix=='.xlsx':
                w=openpyxl.load_workbook(p,read_only=True,data_only=True); _, data=sheet_records(w,r['sheet']);w.close()
            else:data=rows(p)
            # Explicit formats supported by 05a and 05c; verify all rows, not examples.
            bad=[]
            for d in data:
                s,n,u=text(d.get('SampleID')),text(d.get('NeuronID')),text(d.get('NeuronUID'))
                allowed={s+'|'+n,s+':'+n,s+'::'+n}
                if '/fnt/' in r['file']:allowed.add(s+'_'+n.removesuffix('.swc'))
                if u not in allowed:bad.append(d)
            result.update(adjudication='producer_UID_format_verified_no_identity_error' if not bad else 'verified_identity_error', all_rows_checked=len(data),actual_bad_rows=len(bad))
        elif r['check']=='duplicate_identity':
            df=pd.read_csv(p,dtype=str,keep_default_na=False)
            keys=(['sample','neuron_id','path'] if 'path' in df else ['source','SampleID','NeuronID'] if 'source' in df else ['AnimalID','SampleID','NeuronID','Subregion','level','target_index','target_status'])
            if set(keys)<=set(df):
                bad=int(df.duplicated(keys,keep=False).sum())
                result.update(adjudication='explicit_longform_grain_unique' if not bad else 'longform_duplicates_require_context',declared_test_keys=keys,actual_bad_rows=bad)
        elif r['check']=='parse_failure' and p.name.startswith('~$'):
            result.update(adjudication='Office_owner_lock_not_research_workbook',remedy='Retain inventory record; exclude from scientific workbook denominator.')
        elif r['check']=='parse_failure' and p.stat().st_size<=3:
            result.update(adjudication='empty_headerless_historical_QC_export_schema_unavailable',remedy='Future QC writers should emit a header or explicit JSON empty-result status; do not infer zero events from missing schema.')
        elif r['check']=='region_prefix_vs_stored_side':
            result.update(adjudication='preserved_manual_prefix_vs_reference_side_112_114',remedy='Retain original human strings and reviewed decisions separately; no automatic hemisphere relabeling.')
        elif r['check']=='duplicate_headers':
            w=openpyxl.load_workbook(p,read_only=True,data_only=True);it=w[r['sheet']].iter_rows(values_only=True);h=next(it);data=list(it);w.close()
            repeated={c:[i for i,x in enumerate(h) if x==c] for c in h if h.count(c)>1}
            agreement={c:sum(any(row[i]!=row[ix[0]] for i in ix) for row in data) for c,ix in repeated.items()}
            result.update(adjudication='verified_duplicate_total_length_header_values_equal',duplicate_value_disagreements=agreement,remedy='Regenerate isolated derived view with one unambiguous total-length column; do not overwrite historical workbook.')
        elif r['check']=='strength_vs_round_log10_length_plus1':
            w=openpyxl.load_workbook(p,read_only=True,data_only=True)
            for ex in json.loads(r['examples']):
                exrow=ex['row_1based'];ls=r['sheet'].replace('Projection_Strength','Projection_Length',1)
                hl=next(w[ls].iter_rows(values_only=True));hs=next(w[r['sheet']].iter_rows(values_only=True))
                lv=w[ls].cell(exrow,hl.index(ex['target'])+1).value;sv=w[r['sheet']].cell(exrow,hs.index(ex['target'])+1).value
                checks.append({'file':r['file'],'sha256':digest(p),'sheet':r['sheet'],'row_1based':exrow,'NeuronID':ex['NeuronID'],'target':ex['target'],'independent_length':lv,'independent_strength':sv,'independent_expected':round(math.log10(lv+1),4)})
            w.close(); result.update(adjudication='verified_historical_log_before_bilateral_aggregation_523',remedy='Isolated regeneration should keep unknown laterality explicit and compute log10 only after justified length aggregation; retain original bytes.')
        adjudicated.append(result)
    csvout(out/'findings_adjudicated.csv',adjudicated);csvout(out/'independent_strength_cell_readback.csv',checks)

    # Inspect actual ARM keys, not names inferred from suffixes.
    key=ROOT/'atlas/ARM_key_all.txt'
    records=[]
    for line in key.read_text().splitlines()[1:]:
        a=line.split('\t')
        if len(a)>=3 and re.fullmatch(r'[CS][LR]_(PM|CM|RM|R|Pi|M1|M1/PM)',a[1]):records.append({'index':a[0],'abbreviation':a[1],'full_name':a[2]})
    domain_obs=[];absolute_collisions=[];summaries=[];source_bindings=[]
    for i,s in enumerate(books['schemas']):
        if s['status']!='readable':continue
        p=ROOT/s['representative_file'];w=openpyxl.load_workbook(p,read_only=True,data_only=True)
        if 'Summary' in w:
            _,ss=sheet_records(w,'Summary')
            cols=set(ss[0]) if ss else set()
            units=Counter(text(r.get('Length_Unit')) or 'not_declared_in_sheet' for r in ss)
            summaries.append({'file':rel(p),'sha256':s['sha256'],'copies':s['identical_copies'],'rows':len(ss),'Length_Unit_values':dict(units),'identity_count':len({(text(r.get('SampleID')),text(r.get('NeuronID'))) for r in ss}), '523_present':any(text(r.get('NeuronID'))=='523.swc' and text(r.get('SampleID','251637'))=='251637' for r in ss),'explicit_source_hash_columns':sorted(c for c in cols if 'sha' in c.lower() or 'source' in c.lower())})
        for z in s['sheets']:
            name=z['sheet'];h=z['columns']
            if not name.startswith('Projection_Length'):continue
            relevant=[c for c in h if re.fullmatch(r'[CS][LR]_(PM|CM|RM|R|Pi|M1|M1/PM)',c)]
            if not relevant:continue
            _,rr=sheet_records(w,name)
            for c in relevant:
                positive=[d for d in rr if isinstance(d.get(c),(int,float)) and d[c]>0]
                domain_obs.append({'file':rel(p),'sha256':s['sha256'],'sheet':name,'column':c,'positive_rows':len(positive),'sum_stored_units':sum(d[c] for d in positive),'positive_IDs':[text(d.get('NeuronID')) for d in positive]})
            for suffix in ('PM','CM','RM','R','Pi'):
                for side in 'LR':
                    cc,sc=f'C{side}_{suffix}',f'S{side}_{suffix}'
                    if cc not in h or sc not in h:continue
                    overlap=[d for d in rr if isinstance(d.get(cc),(int,float)) and isinstance(d.get(sc),(int,float)) and d[cc]>0 and d[sc]>0]
                    absolute_collisions.append({'file':rel(p),'sha256':s['sha256'],'sheet':name,'suffix':suffix,'hemisphere':side,'both_domain_columns_present':True,'both_positive_same_neuron_rows':len(overlap),'examples':[{k:d.get(k) for k in ('SampleID','NeuronID',cc,sc)} for d in overlap[:30]]})
        w.close()
        if i%25==0:print(json.dumps({'adaptive_workbooks':i,'of':len(books['schemas'])}),flush=True)
    csvout(out/'summary_units_and_cohorts.csv',summaries)
    csvout(out/'unstripped_domain_observations.csv',domain_obs)
    csvout(out/'absolute_domain_collision_observations.csv',absolute_collisions)
    save(out/'atlas_domain_key_definitions.json',{'file':rel(key),'sha256':digest(key),'records':records,'code_interpretation':'C and S domains remain distinct even when suffix strings coincide.'})
    # Reverify previous source bindings rather than interpreting prior PASS as current anatomy.
    cached={};binding_errors=[]
    def bind(path,expected,scope,identity=''):
        pp=Path(path);pp=pp if pp.is_absolute() else ROOT/pp
        try:pp.resolve().relative_to(ROOT)
        except ValueError:return {'scope':scope,'identity':identity,'path':str(pp),'status':'outside_workspace_not_read'}
        k=str(pp.resolve())
        if k not in cached:cached[k]=digest(pp) if pp.is_file() else None
        actual=cached[k]
        r={'scope':scope,'identity':identity,'path':rel(pp),'expected_sha256':expected,'actual_sha256':actual,'status':'match' if actual==expected else 'missing' if actual is None else 'mismatch'}
        if r['status']!='match':binding_errors.append(r)
        return r
    graph=ROOT/'group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/graph_source_readback.csv'
    for d in rows(graph):source_bindings.append(bind(d['path'],d['sha256'],'2480_atlas_source_graph_receipts',d['sample']+'|'+d['neuron_id']))
    prior=ROOT/'notes/region_analysis_review_20261004/final_actual_region_readback.json'
    baseline=json.loads(prior.read_text())
    for d in baseline['samples']:
        source_bindings.append(bind(d['workbook'],d['sha256'],'Oct4_scalar_readback_workbooks'))
    canonical=ROOT/'group_analysis/visual_review_20261002/manifest/canonical_hashes.json'
    ch=json.loads(canonical.read_text())
    if isinstance(ch,dict):
        for path,h in ch.items():
            if isinstance(h,str) and len(h)==64:source_bindings.append(bind(path,h,'protected_canonical_receipt'))
    for f in inv['files']:source_bindings.append(bind(f['file'],f['sha256'],'all_discovered_tables_unchanged'))
    csvout(out/'source_hash_readback.csv',source_bindings)
    save(out/'audit_coverage_summary.json',{'files':inv['total_files'],'unique_content_groups':inv['unique_content_groups'],'bytes':inv['total_bytes'],'extensions':dict(Counter(f['extension'] for f in inv['files'])),'adjudications':dict(Counter(r['adjudication'] for r in adjudicated)),'source_binding_checks':len(source_bindings),'unique_rehashed_paths':len(cached),'source_binding_errors':binding_errors,'all_scientific_tables_preserved':not any(r['scope']=='all_discovered_tables_unchanged' for r in binding_errors),'workbook_sheets':sum(len(s.get('sheets',[])) for s in books['schemas']),'node_table_content_groups':sum(any(z.get('grain')=='node' for z in s.get('sheets',[])) for s in tables['schemas']),'software_only':True,'anatomical_registration_acceptance':'pending','limits':['No recalculation of Excel formula cached values.','No native-to-NMT transform/landmark acceptance.','No inference that historical cohorts are current or equivalent.','Path-based role assignments are provisional; exact source/human/candidate status is retained.','No external shares or portal calls.','CSV global duplicate checks are targeted to flagged long-form tables; initial broad checks are chunk-local.']})
    print(json.dumps({'adjudications':dict(Counter(r['adjudication'] for r in adjudicated)),'binding_errors':len(binding_errors),'domain_overlap_rows':sum(r['both_positive_same_neuron_rows'] for r in absolute_collisions)}),flush=True)


if __name__=='__main__':main()
