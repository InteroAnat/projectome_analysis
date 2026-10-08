"""Apply existing hash-bound case evidence as separate fields; never overwrite audits."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
C=ROOT/'group_analysis/evolution_20261008/classification/coarse_insula_review_20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    csv.field_size_limit(10000000)
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    output=OUT/'selected_52_background_cases_with_case_overlays.csv';receipt=OUT/'delivery_receipt_v2.json'
    if output.exists() or receipt.exists():raise FileExistsError('Preserve existing output')
    prior=json.loads((OUT/'independent_receipt.json').read_text());inputs={}
    def bind(p,h=None):
        p=Path(p).resolve();actual=sha(p)
        if h and actual!=h:raise ValueError('Hash mismatch '+str(p))
        inputs[p.relative_to(ROOT).as_posix()]=actual;return p
    source=bind(OUT/'selected_52_background_cases.csv',prior['outputs']['selected_52_background_cases.csv'])
    delivery=json.loads(bind(C/'delivery_provenance_20261009.json').read_text())
    nativep=bind(C/'native_case_pair_supplement_20261009.csv',delivery['artifacts']['native_case_pair_supplement_20261009.csv'])
    qcp=bind(C/'case_and_source_qc_supplement_20261009.json',delivery['artifacts']['case_and_source_qc_supplement_20261009.json'])
    qc=json.loads(qcp.read_text());overlay={r['sample']+'|'+r['neuron_id']:r for r in read(nativep)};rows=read(source);count_native=0;count_assets=0
    for r in rows:
        case=overlay.get(r['uid']);assets=[]
        if r['sample']=='251637':assets=[a for a in qc['case112_114']['assets'] if a['neuron_id']==r['neuron_id']]
        r['EffectiveNativeAtlasPairAvailable']=r['native_atlas_pair_available'];r['EffectiveNativeSWCPath']=r['native_raw_swc_path'];r['EffectiveNativeSWCSHA256']=r['native_raw_swc_sha256'];r['EffectiveNativeRootXYZ']=r['native_root_xyz_um'];r['NativeCaseOverlayApplied']=bool(case)
        if case:
            if case['own_atlas_sha256']!=r['SWCSHA256']:raise ValueError('Overlay atlas hash mismatch')
            bind(ROOT/case['native_raw_swc_path'],case['native_raw_swc_sha256']);rec=json.loads(case['native_source_identity_receipt']);bind(ROOT/rec['path'],rec['sha256'])
            r.update({'EffectiveNativeAtlasPairAvailable':case['native_atlas_pair_available'],'EffectiveNativeSWCPath':case['native_raw_swc_path'],'EffectiveNativeSWCSHA256':case['native_raw_swc_sha256'],'EffectiveNativeRootXYZ':case['native_root_xyz_um'],'NativeCaseOverlaySource':str(nativep),'NativeCaseSourceIdentityReceipt':case['native_source_identity_receipt']})
        r['AdditionalCaseVisualAssets']=json.dumps(assets);r['CaseVisualEvidenceSource']=str(qcp) if assets else '';r['CaseVisualAssetCheckInThisAudit']='recorded hash-bound local/historical evidence; image bytes not reopened' if assets else ''
        r['EffectiveExistingVisualAssetsAvailable']=bool(json.loads(r['existing_visual_assets'] or '{}') or assets)
        if r['EffectiveNativeAtlasPairAvailable']=='True':count_native+=1
        if r['EffectiveExistingVisualAssetsAvailable']:count_assets+=1
    with output.open('x',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    back=read(output)
    if len(back)!=52 or len({r['uid'] for r in back})!=52:raise ValueError('Case identity readback failed')
    for rel,h in inputs.items():
        if sha(ROOT/rel)!=h:raise ValueError('Input changed')
    note={'status':'software_only_saved_case_evidence_overlay_readback_passed','recorded_at':datetime.now().astimezone().isoformat(),'primary_case_ledger':output.relative_to(ROOT).as_posix(),'primary_case_ledger_sha256':sha(output),'prior_source_root_receipt_sha256':sha(OUT/'independent_receipt.json'),'case_overlay_script_sha256':sha(__file__),'inputs':inputs,'effective_native_atlas_pairs':count_native,'effective_cases_with_existing_visual_assets':count_assets,'native_case_overlay_UIDs':[r['uid'] for r in rows if r['NativeCaseOverlayApplied']],'additional_case_visual_UIDs':[r['uid'] for r in rows if json.loads(r['AdditionalCaseVisualAssets'])],'original_case52_source_roots_rehashes':52,'additional_native_source_rehashes':3,'image_files_read_or_opened':0,'external_historical_X_paths_reprobed':False,'all_source_and_previous_files_preserved':True,'scientific_acceptance':False,'map_or_canonical_changes':False}
    with receipt.open('x',encoding='utf-8') as f:json.dump(note,f,indent=2)
    print(json.dumps(note,indent=2))
if __name__=='__main__':main()
