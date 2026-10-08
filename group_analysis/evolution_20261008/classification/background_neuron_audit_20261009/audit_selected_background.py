"""Read-only selected52 ARM6-background case audit; source-preserving no-clobber."""
import csv, json, hashlib, math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'group_analysis/evolution_20261008'
COARSE=BASE/'classification/coarse_insula_review_20261009'
CORRECTED=COARSE/'distance_priority_v2_20261009'
SELECTED=BASE/'projection_inputs/arm_labels_20261009'
ATLAS=BASE/'atlas_locations/soma_audit_20261009/validated_atlas_lookup'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p):
    csv.field_size_limit(10000000)
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
def truth(v):return str(v).lower()=='true'
def vec(s):
    if not s:return None
    a=np.asarray(json.loads(s),dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Invalid coordinate')
    return a
def main():
    names=['selected_52_background_cases.csv','source_root_readback.csv','sample_coverage.csv','supplementary_unselected_background_queue.csv','summary.json','independent_receipt.json']
    if any((OUT/n).exists() for n in names):raise FileExistsError('Audit outputs already exist; preserve them and use a new variant')
    bound={}
    def bind(p,expected=None,role='input'):
        p=Path(p).resolve();rel=p.relative_to(ROOT).as_posix();actual=sha(p)
        if expected and actual!=expected:raise ValueError('Hash mismatch: '+rel)
        bound[rel]={'sha256':actual,'role':role};return p
    cp=bind(CORRECTED/'correction_provenance.json');cor=json.loads(cp.read_text())
    rp=bind(CORRECTED/'all_neuron_review_manifest_distance_corrected.csv',cor['outputs']['all_neuron_review_manifest_distance_corrected.csv'])
    bind(ROOT/cor['input_delivery']['path'],cor['input_delivery']['sha256'])
    for name,h in cor['protected_prior_file_sha256'].items():bind(COARSE/name,h,'protected_prior_audit')
    prov=json.loads((COARSE/'provenance.json').read_text())
    for name in ['atlas_audit','policies','Henry','animal_registry','segmentation','reference','label_key']:
        v=prov['inputs'][name];bind(ROOT/v['path'],v['sha256'],name)
    ap=bind(ATLAS/'provenance.json');aprov=json.loads(ap.read_text())
    for name in ['ARM6','LR_plane','key','reviewed']:
        v=aprov['inputs'][name];bind(ROOT/v['path'],v['sha256'],name)
    bind(ROOT/aprov['inventory']['path'],aprov['inventory']['sha256'],'complete_inventory')
    ip=bind(BASE/'inventory/live_inventory_20261008/provenance.json');iprov=json.loads(ip.read_text());box=iprov['source_files']['boxes'];bind(ROOT/box['file'],box['sha256'],'existing_folded_rule')
    sd=bind(SELECTED/'delivery_provenance.json');sdel=json.loads(sd.read_text())
    for name in ['combined/projection_manifest.csv','main/projection_manifest.csv','preview/projection_manifest.csv','preparation_provenance.json']:bind(SELECTED/name,sdel['artifacts'][name],'protected_selected_input')
    rows=read(rp);byuid={r['uid']:r for r in rows};selected=read(SELECTED/'combined/projection_manifest.csv');seluid={r['uid'] for r in selected}
    if len(rows)!=8746 or len(byuid)!=8746 or len(selected)!=462 or len(seluid)!=462:raise ValueError('Identity denominator mismatch')
    if any(r['uid']!=r['sample']+'|'+r['neuron_id'] for r in rows):raise ValueError('UID mismatch')
    bg=[r for r in rows if r['own_atlas_id']=='0' and r['own_lookup_status']=='atlas_background'];cases_m=[r for r in selected if r['ARMIndex']=='0'];unselected=[r for r in bg if r['uid'] not in seluid]
    if (len(bg),len(cases_m),len(unselected))!=(295,52,243):raise ValueError('Background denominator mismatch')
    armimg=nib.load(ROOT/aprov['inputs']['ARM6']['path']);arm=np.asanyarray(armimg.dataobj)[...,0,5]
    maskimg=nib.load(ROOT/aprov['inputs']['LR_plane']['path']);mask=np.asanyarray(maskimg.dataobj)
    simg=nib.load(ROOT/prov['inputs']['segmentation']['path']);seg=np.asanyarray(simg.dataobj);ref=nib.load(ROOT/prov['inputs']['reference']['path'])
    for image in [armimg,maskimg,simg]:
        if image.shape[:3]!=ref.shape or not np.array_equal(image.affine,ref.affine):raise ValueError('Reference geometry mismatch')
    classes={}
    for ext in simg.header.extensions:
        c=ext.get_content();xml=c.decode() if isinstance(c,bytes) else str(c)
        for e in ET.fromstring(xml):
            if e.attrib.get('atr_name')=='ATLAS_LABEL_TABLE':classes.update({int(x.attrib['VAL']):x.attrib['STRUCT'] for x in ET.fromstring(e.text.strip().strip('"').strip())})
    if classes!={1:'CSF',2:'GM',3:'scGM',4:'WM',5:'BV'}:raise ValueError('Unexpected official tissue class table')
    with (ROOT/aprov['inputs']['key']['path']).open(encoding='utf-8-sig',newline='') as f:key={int(r['Index']):r for r in csv.DictReader(f,delimiter='\t')}
    anchors=defaultdict(list)
    for r in rows:
        if not truth(r['excluded']) and (truth(r['Henry_coarse_INS_visual_evidence']) or truth(r['atlas_INS'])) and r['portal_coordinate_index_xyz']:anchors[r['sample']].append(r)
    cases=[];fresh=[];native_rehashed=0
    for m in cases_m:
        r=byuid[m['uid']]
        if (m['SampleID'],m['NeuronID'],m['AnimalID'])!=(r['sample'],r['neuron_id'],r['registry_animal']) or r['own_atlas_id']!='0' or truth(r['excluded']):raise ValueError('Selected identity/animal/status mismatch')
        source=bind(m['SWCPath'],m['SWCSHA256'],'selected_background_atlas_SWC')
        if source!=(ROOT/r['map_selected_source']).resolve() or m['SWCSHA256']!=r['map_selected_sha256']:raise ValueError('Selected source mismatch')
        a=np.loadtxt(source,comments='#',ndmin=2)
        if a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Invalid SWC numeric nodes')
        roots=a[a[:,6]==-1]
        if len(roots)!=1:raise ValueError('Invalid root count')
        xyz=roots[0,2:5];index=xyz/250
        if not np.array_equal(xyz,vec(r['own_root_xyz_um'])):raise ValueError('Fresh root mismatch')
        checks={}
        for policy,v in [('current_rint_zero_center',np.rint(index).astype(int)),('published_ceil_edge_one_based_to_zero',(np.ceil(index)-1).astype(int)),('halfopen_floor_zero_center',np.floor(index+.5).astype(int))]:
            if np.any(v<0) or np.any(v>=np.asarray(arm.shape)):raise ValueError('Unexpected outside root')
            idx=tuple(v);aid=int(arm[idx]);side={0:'Unknown',1:'R',2:'L'}[int(mask[idx])];tissue=int(seg[idx]);abbr=key[aid]['Abbreviation'] if aid else '';full=key[aid]['Full_Name'] if aid else 'Atlas background'
            if r['own_root_'+policy+'_label']!=(abbr or 'Unknown_0'):raise ValueError('Fresh policy label mismatch')
            if policy=='current_rint_zero_center' and (aid!=0 or side!=m['Hemisphere']):raise ValueError('Fresh ARM0/side mismatch')
            if policy!='halfopen_floor_zero_center' and str(tissue)!=r['own_root_'+policy+'_tissue_code']:raise ValueError('Fresh tissue mismatch')
            checks.update({policy+'_ARMIndex':aid,policy+'_ARMAbbreviation':abbr,policy+'_ARMFullName':full,policy+'_Hemisphere':side,policy+'_TissueCode':tissue,policy+'_TissueClass':classes.get(tissue,'background_0'),policy+'_VoxelXYZ':json.dumps(v.tolist())})
        pos=vec(r['portal_coordinate_index_xyz']);dists=[(float(np.linalg.norm(pos-vec(a['portal_coordinate_index_xyz']))*.25),a['neuron_id'],a) for a in anchors[r['sample']] if a['uid']!=r['uid']]
        if pos is None or not dists:raise ValueError('Selected case distance unexpectedly missing')
        d,_,anchor=min(dists,key=lambda x:(x[0],x[1]))
        if not math.isclose(d,float(r['nearest_same_exact_sample_INS_anchor_mm']),rel_tol=1e-12,abs_tol=1e-12) or anchor['uid']!=r['nearest_INS_anchor_uid']:raise ValueError('Nearest OTHER anchor mismatch')
        basis=(['Henry_coarse_INS_visual_annotation'] if truth(anchor['Henry_coarse_INS_visual_evidence']) else [])+(['original_portal_INS_label'] if truth(anchor['atlas_INS']) else [])
        if basis!=json.loads(r['nearest_INS_anchor_evidence_basis']):raise ValueError('Anchor evidence mismatch')
        if r['native_raw_swc_path'] and r['native_raw_swc_sha256']:
            bind(ROOT/r['native_raw_swc_path'],r['native_raw_swc_sha256'],'selected_background_native_SWC');native_rehashed+=1
        item=dict(r);item.update({k:m[k] for k in ['SampleID','NeuronID','AnimalID','SourceCohort','EvidenceSourceGroup','ARMLevel','ARMIndex','ARMAbbreviation','ARMFullName','Hemisphere','DisplayLabel','SourceARMStatus','AtlasPath','AtlasKeyPath','AtlasSHA256','AtlasKeySHA256','SWCPath','SWCSHA256']})
        item.update({'BackgroundCaseEvidenceStatus':'Henry_visual_reviewed_coarse_INS' if truth(r['Henry_coarse_INS_visual_evidence']) else 'potential_only_no_new_anatomical_acceptance','SourceHashFreshlyVerified':True,'OwnRootFreshlyRead':True,'TissueSegmentationSHA256':prov['inputs']['segmentation']['sha256'],'AutomaticRelabeling':False,**checks});cases.append(item)
        fresh.append({'uid':r['uid'],'SWCPath':str(source),'SWCSHA256':m['SWCSHA256'],'RootNodeID':int(roots[0,0]),'RootXYZ':json.dumps(xyz.tolist()),'IndexXYZ':json.dumps(index.tolist()),'nearest_OTHER_anchor_uid':anchor['uid'],'nearest_OTHER_anchor_mm':d,'anchor_basis':json.dumps(basis),**checks})
    partition=Counter(r['EvidenceSourceGroup'].rsplit('_',1)[0] for r in cases)
    if dict(partition)!={'HumanINS':13,'Candidate':13,'OriginCandidate':15,'NearINS':11}:raise ValueError('Evidence partition mismatch')
    samples=[]
    for sample in sorted({r['sample'] for r in rows}):
        rs=[r for r in rows if r['sample']==sample];bs=[r for r in bg if r['sample']==sample];us=[r for r in unselected if r['sample']==sample]
        samples.append({'SampleID':sample,'ExactUIDs':len(rs),'OwnGraphVerified':sum(r['map_source_graph_status']=='numeric_graph_verified_all_copies' for r in rs),'OwnSWCMissingUnassessed':sum(r['map_source_graph_status']=='missing_unassessed' for r in rs),'PortalCoordinatesMissing':sum(not r['portal_coordinate_index_xyz'] for r in rs),'OwnARM6Background':len(bs),'SelectedBackground':len(bs)-len(us),'UnselectedBackground':len(us),'UnselectedWithin2mm':sum(bool(r['nearest_same_exact_sample_INS_anchor_mm']) and float(r['nearest_same_exact_sample_INS_anchor_mm'])<=2 for r in us),'UnselectedDistanceUnknown':sum(not r['nearest_same_exact_sample_INS_anchor_mm'] for r in us),'RegistryAnimal':next((r['registry_animal'] for r in rs if r['registry_animal']),'')})
    for r in unselected:
        r.update({'SelectionStatus':'not_in_current_462','CurrentARMFullName':'Atlas background','FreshSourceReadInThisAudit':False,'ProposedMapAddition':False,'DistanceMissingReason':'no_established_INS_anchor_in_same_sample' if not anchors[r['sample']] else ''})
    summary={'observed_at':datetime.now().astimezone().isoformat(),'primary_scope':'52existing_selected_ARM6_background_soma_cases','selected52_evidence_partition':dict(partition),'selected_Henry_reviewed_coarse_INS':13,'selected_potential_only':39,'selected_by_sample':dict(Counter(r['sample'] for r in cases)),'selected_by_animal':dict(Counter(r['AnimalID'] for r in cases)),'selected_mask_sides':dict(Counter(r['Hemisphere'] for r in cases)),'selected_current_tissue':dict(Counter(r['current_rint_zero_center_TissueClass'] for r in cases)),'selected_edge_tissue':dict(Counter(r['published_ceil_edge_one_based_to_zero_TissueClass'] for r in cases)),'selected_tissue_policy':dict(Counter(r['own_root_tissue_policy_status'] for r in cases)),'selected_edge_full_ARM_labels':dict(Counter(r['published_ceil_edge_one_based_to_zero_ARMFullName'] for r in cases)),'selected_halfopen_label_differences':sum(r['current_rint_zero_center_ARMIndex']!=r['halfopen_floor_zero_center_ARMIndex'] for r in cases),'selected_halfopen_voxel_differences':sum(r['current_rint_zero_center_VoxelXYZ']!=r['halfopen_floor_zero_center_VoxelXYZ'] for r in cases),'selected_Henry_UIDs':[r['uid'] for r in cases if truth(r['Henry_coarse_INS_visual_evidence'])],'selected_native_atlas_pairs':sum(truth(r['native_atlas_pair_available']) for r in cases),'selected_visual_manifest_present':sum(truth(r['visual_manifest_present']) for r in cases),'selected_visual_context_coverage':dict(Counter(r['visual_context_coverage_status'] or 'not_recorded' for r in cases)),'selected_reviewed_soma_decisions':{r['uid']:{'region':r['reviewed_region'],'status':r['reviewed_status'],'effective_status':r['reviewed_effective_status']} for r in cases if r['reviewed_region']},'supplementary_complete_UIDs':8746,'supplementary_own_roots':1912,'supplementary_missing_local_SWC':6834,'supplementary_missing_portal_coordinates':sum(not r['portal_coordinate_index_xyz'] for r in rows),'supplementary_own_ARM6_positive':1617,'supplementary_own_ARM6_background':295,'supplementary_unselected_background':243,'supplementary_unselected_current_tissue':dict(Counter(r['own_root_current_rint_zero_center_tissue_class'] for r in unselected)),'supplementary_unselected_within2mm_UIDs':[r['uid'] for r in unselected if r['nearest_same_exact_sample_INS_anchor_mm'] and float(r['nearest_same_exact_sample_INS_anchor_mm'])<=2],'supplementary_unselected_distance_unknown':sum(not r['nearest_same_exact_sample_INS_anchor_mm'] for r in unselected),'primary_policy':'ARM6 [...,0,5], np.rint(SWC XYZ/250), zero-based; no inverse affine','origin_acceptance':'unresolved; ceil(index)-1 and floor(index+.5) sensitivity only','official_segmentation_classes':classes,'reference_shape':list(ref.shape),'reference_affine':ref.affine.tolist(),'saved_folded_candidate_rule':iprov['coordinate_rule'],'saved_folded_padding_mm':iprov['padding_mm'],'fresh_selected_atlas_source_hashes':52,'fresh_selected_root_reads':52,'fresh_selected_native_source_hashes':native_rehashed,'fresh_nearest_OTHER_anchor_checks':52,'network_requests':0,'image_reads':0,'map_or_canonical_writes':0,'automatic_relabeling':False,'new_scientific_acceptance':False}
    write(OUT/names[0],cases);write(OUT/names[1],fresh);write(OUT/names[2],samples);write(OUT/names[3],unselected)
    with (OUT/'summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    reread=read(OUT/names[0]);saved_samples=read(OUT/names[2])
    if len(reread)!=52 or {r['uid'] for r in reread}!={r['uid'] for r in cases_m}:raise ValueError('Saved case UID readback mismatch')
    if sum(int(r['ExactUIDs']) for r in saved_samples)!=8746 or sum(int(r['OwnARM6Background']) for r in saved_samples)!=295:raise ValueError('Saved sample conservation mismatch')
    for rel,rec in bound.items():
        if sha(ROOT/rel)!=rec['sha256']:raise ValueError('Input/source changed during audit: '+rel)
    receipt={'status':'software_and_source_readback_passed','observed_at':datetime.now().astimezone().isoformat(),'audit_script_sha256':sha(__file__),'inputs_and_selected_sources':bound,'outputs':{n:sha(OUT/n) for n in names[:-1]},'fresh_selected_source_hashes':52,'fresh_root_reads':52,'fresh_nearest_OTHER_anchor_checks':52,'saved_primary_unique_UIDs':len({r['uid'] for r in reread}),'all_protected_inputs_and_selected_sources_unchanged':True,'scope_conservation':'52selected+243unselected=295background;1617positive+295background=1912own;1912own+6834missing=8746','prior_full_graph_equivalence':'reuse hash-bound prior full2480source/1912UID readback; no new full graph comparison','fresh_broad_unselected_graph_reads':0,'image_reads':0,'network_requests':0,'map_writes':0,'canonical_writes':0,'scientific_acceptance':False}
    with (OUT/names[-1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
