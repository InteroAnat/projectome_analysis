"""Independent actual-file verification; no preparer/helper lookup reused."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    provenance=OUT/'preparation_provenance.json';report=json.loads(provenance.read_text())
    for entry in report['inputs'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed bound input')
    for name,expected in report['outputs'].items():
        if sha(OUT/name)!=expected:raise ValueError('Changed output')
    if sha(ROOT/'group_analysis/scripts/prepare_arm_labeled_projection_manifest.py')!=report['preparer_sha256']:raise ValueError('Preparer code hash changed')
    atlas=np.asanyarray(nib.load(report['inputs']['atlas']['path']).dataobj)[...,0,5]
    mask=np.asanyarray(nib.load(report['inputs']['hemisphere_mask']['path']).dataobj)
    reference=nib.load(report['inputs']['reference']['path'])
    key={int(d['Index']):d for d in csv.DictReader(open(report['inputs']['atlas_key']['path'],encoding='utf-8'),delimiter='\t')}
    # Independently derive mask semantics from exact known ARM prefixes.
    value_to_side={}
    for value in np.unique(mask):
        if not value:continue
        codes=np.unique(atlas[mask==value]);sides=set()
        for code in codes:
            abbreviation=key.get(int(code),{}).get('Abbreviation','')
            if abbreviation.startswith(('CL_','SL_')):sides.add('L')
            if abbreviation.startswith(('CR_','SR_')):sides.add('R')
        if len(sides)!=1:raise ValueError('Ambiguous independent mask semantics')
        value_to_side[int(value)]=sides.pop()
    fields=report['label_schema'];cohort_sets={};failures=[];checked=[]
    combined=pd.read_csv(OUT/'combined/projection_manifest.csv',dtype=str,keep_default_na=False)
    for cohort in ('main','preview'):
        before=pd.read_csv(report['inputs'][cohort+'_manifest']['path'],dtype=str,keep_default_na=False)
        after=pd.read_csv(OUT/cohort/'projection_manifest.csv',dtype=str,keep_default_na=False)
        old_ids=list(zip(before.SampleID,before.NeuronID));new_ids=list(zip(after.SampleID,after.NeuronID))
        if old_ids!=new_ids:raise ValueError('Changed exact selected source identities/order')
        cohort_sets[cohort]=set(new_ids)
        for c in before.columns:
            expected=after.EvidenceSourceGroup if c=='Subregion' else after[c]
            if not before[c].equals(expected):raise ValueError('Changed pre-existing evidence/source metadata')
        part=combined[combined.SourceCohort.eq(cohort)]
        for c in after.columns:
            if after[c].tolist()!=part[c].tolist():raise ValueError('Combined rows differ from separate cohort')
    if cohort_sets['main']&cohort_sets['preview']:raise ValueError('Main/preview overlap')
    for cohort in ('main','preview','combined'):
        frame=pd.read_csv(OUT/cohort/'projection_manifest.csv',dtype=str,keep_default_na=False)
        labels=pd.read_csv(OUT/cohort/'label_map.csv',dtype=str,keep_default_na=False)
        if labels.Subregion.duplicated().any() or set(labels.Subregion)!=set(frame.Subregion):raise ValueError('Label-map coverage/identity mismatch')
        for name,group in frame.groupby('Subregion'):
            label=labels[labels.Subregion.eq(name)].iloc[0]
            for c in fields:
                if set(group[c])!={label[c]}:raise ValueError('Per-row full-name label map differs')
    for row in combined.to_dict('records'):
        source=Path(row['SWCPath'])
        if sha(source)!=row['SWCSHA256']:raise ValueError('Source hash changed')
        graph=np.loadtxt(source,comments='#',usecols=range(7),ndmin=2)
        roots=graph[graph[:,6]==-1]
        if len(roots)!=1:raise ValueError('Independent source root count differs')
        raw=roots[0,2:5]
        if row['CoordinateFrame']=='atlas_index_um':index=raw/np.array([float(v) for v in row['IndexScaleUm'].split(';')])
        else:
            inverse=np.linalg.inv(reference.affine);index=inverse[:3,:3]@raw+inverse[:3,3]
        voxel=np.rint(index).astype(int);inside=np.all(voxel>=0)&np.all(voxel<atlas.shape)
        code=int(atlas[tuple(voxel)]) if inside else -1
        side=value_to_side.get(int(mask[tuple(voxel)]),'Unknown') if inside else 'Unknown'
        status='mapped' if code>0 else 'zero_unassigned' if code==0 else 'out_of_FOV'
        official=key[code]['Full_Name'] if code>0 else 'Atlas background' if code==0 else 'Outside reference'
        abbreviation=key[code]['Abbreviation'] if code>0 else ''
        display=official+' ('+{'L':'Left','R':'Right','Unknown':'Unknown'}[side]+')'
        expected={'ARMLevel':'6','ARMIndex':str(code),'ARMFullName':official,'ARMAbbreviation':abbreviation,'Hemisphere':side,'DisplayLabel':display,'SourceARMStatus':status,'Subregion':f'ARM6_{code}_{side}'}
        for c,value in expected.items():
            if row[c]!=value:failures.append({'uid':row['SampleID']+'|'+row['NeuronID'],'field':c,'actual':row[c],'expected':value})
        if json.loads(row['SourceRootVoxelXYZ'])!=voxel.tolist():raise ValueError('Recorded root voxel differs')
        if not np.allclose(json.loads(row['SourceRootIndexXYZ']),index,rtol=0,atol=1e-12):raise ValueError('Recorded root index differs')
        if str(int(roots[0,0]))!=row['SourceRootNodeID']:raise ValueError('Recorded root node identity differs')
        if sha(row['AtlasPath'])!=row['AtlasSHA256'] or sha(row['AtlasKeyPath'])!=row['AtlasKeySHA256']:raise ValueError('Label-row atlas/key binding differs')
        checked.append({'SampleID':row['SampleID'],'NeuronID':row['NeuronID'],'SourceCohort':row['SourceCohort'],'ARMIndex':code,'ARMFullName':official,'Hemisphere':side,'SourceARMStatus':status})
    if failures:raise ValueError(f'Independent label failures: {failures[:5]}')
    evidence={'status':'passed','software_only':True,'fresh_independent_source_root_reads':len(checked),
        'exact_main_neurons':len(cohort_sets['main']),'exact_preview_neurons':len(cohort_sets['preview']),
        'exact_combined_neurons':len(combined),'animals':sorted(combined.AnimalID.unique().tolist()),
        'full_label_groups':combined[fields].drop_duplicates().to_dict('records'),
        'lookup_status':dict(Counter(d['SourceARMStatus'] for d in checked)),
        'mask_value_to_side':value_to_side,'all_original_non_grouping_metadata_unchanged':True,
        'new_source_grouping_only':'ARM6 direct root lookup; old manual/atlas/candidate flags remain provenance',
        'original_sources_and_old_manifest_hashes_match':True,'independent_root_lookup_failures':failures,
        'preparation_provenance_sha256':sha(provenance),'independent_checker_sha256':sha(Path(__file__)),
        'source_and_atlas_files_written':False,'anatomical_registration_or_origin_acceptance':False}
    with (OUT/'independent_readback.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2)
    print(json.dumps({k:evidence[k] for k in ('status','fresh_independent_source_root_reads','lookup_status')}))


if __name__=='__main__':main()
