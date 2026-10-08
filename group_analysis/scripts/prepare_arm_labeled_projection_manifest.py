"""Create new ARM-only source groups for already selected, hash-bound neurons.

The selection, sources and evidence remain unchanged. Source groups use a
fresh SWC-root lookup in the actual pinned ARM level-6 volume, with the
existing np.rint center-origin policy. Other origin policies are sensitivity
metadata only. Background/outside roots remain explicitly unresolved.
No registration, source download, canonical relabeling or map computation occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'main_scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_projection_maps import checked_manifest, sha256
from endpoint_atlas import EndpointAtlas, coded_mm_grid
from swc_validation import parse_swc

PRIMARY='current_rint_zero_center'
POLICIES=(PRIMARY,'published_ceil_edge_one_based_to_zero','halfopen_floor_zero_center')
LABEL_COLUMNS=['Subregion','ARMLevel','ARMIndex','ARMAbbreviation','ARMFullName','Hemisphere',
               'DisplayLabel','AtlasSHA256','AtlasKeySHA256','AtlasPath','AtlasKeyPath','SourceARMStatus']
IDENTITY=['SampleID','NeuronID']


def _write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def _write_csv(path,frame):
    with path.open('x',encoding='utf-8',newline='') as stream:frame.to_csv(stream,index=False)


def _bound_run(manifest,provenance,readback):
    run=json.loads(provenance.read_text(encoding='utf-8'))
    review=json.loads(readback.read_text(encoding='utf-8'))
    if run.get('status')!='software_verified_candidate_endpoints' or review.get('status')!='passed':
        raise ValueError('Require complete endpoint run and passed matching software readback')
    if review.get('run_provenance_sha256')!=sha256(provenance):raise ValueError('Endpoint readback does not bind run provenance')
    expected=run['inputs']['manifest']['sha256']
    if sha256(manifest)!=expected or review.get('manifest_sha256')!=expected:raise ValueError('Selected manifest differs from bound endpoint run')
    for key,entry in run['inputs'].items():
        if sha256(Path(entry['path']))!=entry['sha256']:raise ValueError(f'Changed endpoint input: {key}')
    return run


def policy_voxel(index_xyz,policy):
    xyz=np.asarray(index_xyz,dtype=float)
    if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('Root coordinates must be a finite triple')
    if policy==PRIMARY:value=np.rint(xyz)
    elif policy==POLICIES[1]:value=np.ceil(xyz)-1
    elif policy==POLICIES[2]:value=np.floor(xyz+.5)
    else:raise ValueError('Unknown explicit root-origin policy')
    # Bound before integer conversion, avoiding overflow for remote finite coordinates.
    return tuple(int(v) for v in value)


def source_index_xyz(root,record,grid):
    xyz=np.asarray(root[2:5],dtype=float)
    if record['CoordinateFrame']=='atlas_index_um':return xyz/np.asarray(record['index_scale_um'])
    if record['CoordinateFrame']=='nifti_world_mm':
        inverse=np.linalg.inv(grid.affine_mm)
        return inverse[:3,:3]@xyz+inverse[:3,3]
    raise ValueError('Unsupported declared source coordinate frame')


def lookup_root(atlas,index_xyz,policy,atlas_hash,key_hash):
    voxel=policy_voxel(index_xyz,policy)
    inside=all(0<=v<atlas.data.shape[i] for i,v in enumerate(voxel))
    if inside:
        side,targets=atlas.lookup(voxel);target=targets[5]
    else:
        side,targets=atlas.lookup(None);target=targets[5]
    index=target['index'];status=target['target_status']
    if status=='mapped':
        name=target['full_name'];abbreviation=target['abbreviation']
    elif status=='zero_unassigned':name,abbreviation='Atlas background',''
    elif status=='out_of_FOV':name,abbreviation,index='Outside reference','',-1
    else:raise ValueError(f'Positive source ARM6 label lacks a valid exact key entry: {index}: {status}')
    region=f'ARM6_{index}_{side}'
    full_side={'L':'Left','R':'Right','Unknown':'Unknown'}[side]
    metadata=dict(Subregion=region,ARMLevel=6,ARMIndex=index,ARMAbbreviation=abbreviation,
        ARMFullName=name,Hemisphere=side,DisplayLabel=f'{name} ({full_side})',AtlasSHA256=atlas_hash,AtlasKeySHA256=key_hash)
    metadata.update(SourceARMStatus=status,SourceRootVoxelXYZ=json.dumps(voxel),
        SourceARMLabelHemisphere=target['label_side'],SourceARMHemisphereConflict=target['hemisphere_conflict'])
    return metadata


def prepare(main_manifest,preview_manifest,main_run,main_readback,preview_run,preview_readback,output,
            *,input_root=ROOT,root_policy=PRIMARY):
    paths={k:Path(v).resolve() for k,v in dict(main_manifest=main_manifest,preview_manifest=preview_manifest,
        main_run=main_run,main_readback=main_readback,preview_run=preview_run,preview_readback=preview_readback).items()}
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new ARM-only manifest destination')
    if root_policy!=PRIMARY:raise ValueError('This selection variant retains current_rint_zero_center; alternatives are sensitivity only')
    initial={k:sha256(p) for k,p in paths.items()}
    main=_bound_run(paths['main_manifest'],paths['main_run'],paths['main_readback'])
    preview=_bound_run(paths['preview_manifest'],paths['preview_run'],paths['preview_readback'])
    asset_names=('reference','atlas','atlas_key','hemisphere_mask')
    for name in asset_names:
        if main['inputs'][name]['sha256']!=preview['inputs'][name]['sha256']:raise ValueError(f'Main/preview use different {name}')
    assets={name:Path(main['inputs'][name]['path']).resolve() for name in asset_names}
    hashes={name:main['inputs'][name]['sha256'] for name in asset_names}
    grid=coded_mm_grid(nib.load(str(assets['reference'])))
    atlas=EndpointAtlas(assets['atlas'],assets['atlas_key'],assets['hemisphere_mask'],grid)
    frames={};sources=set();source_hashes={};identities=set();roots=[];labels={}
    for cohort in ('main','preview'):
        manifest=paths[cohort+'_manifest'];frame=pd.read_csv(manifest,dtype=str,keep_default_na=False)
        records=checked_manifest(manifest,input_root)
        if len(frame)!=len(records):raise ValueError('Checked source manifest row count changed')
        reserved=set(LABEL_COLUMNS)-{'Subregion'}|{'EvidenceSourceGroup','SourceCohort','SourceRootNodeID','SourceRootXYZ','SourceRootIndexXYZ','SourceARMStatus','SourceRootVoxelXYZ','SourceARMLabelHemisphere','SourceARMHemisphereConflict','SourceRootLookupPolicy'}
        if reserved.intersection(frame.columns):raise ValueError('Input already contains reserved ARM preparation columns')
        rows=[]
        for original,record in zip(frame.to_dict('records'),records):
            identity=(record['SampleID'],record['NeuronID'])
            if identity in identities:raise ValueError('Main/preview exact neuron identity overlap')
            identities.add(identity)
            if record['source_path'] in sources:raise ValueError('Main/preview source path overlap')
            sources.add(record['source_path'])
            source_hashes[record['source_path']]=record['expected_sha256']
            for alias,target in (('sample','SampleID'),('neuron_id','NeuronID')):
                if alias in original and original[alias]!=original[target]:raise ValueError('Exact sample/channel or neuron alias mismatch')
            if 'uid' in original and original['uid']!='|'.join(identity):raise ValueError('Source UID does not match exact sample/channel+neuron')
            payload=record['source_path'].read_bytes()
            if hashlib.sha256(payload).hexdigest()!=record['expected_sha256']:raise ValueError('SWC changed before root readback')
            graph=parse_swc(payload.decode('utf-8-sig'),str(record['source_path']))
            root=next(node for node in graph if node[6]==-1)
            index_xyz=source_index_xyz(root,record,grid)
            current=lookup_root(atlas,index_xyz,PRIMARY,hashes['atlas'],hashes['atlas_key'])
            current.update(AtlasPath=str(assets['atlas']),AtlasKeyPath=str(assets['atlas_key']))
            item={**original,'EvidenceSourceGroup':original['Subregion'],'SourceCohort':cohort,**current,
                'SourceRootNodeID':root[0],'SourceRootXYZ':json.dumps(root[2:5]),'SourceRootIndexXYZ':json.dumps(index_xyz.tolist()),
                'SourceRootLookupPolicy':PRIMARY}
            for policy in POLICIES[1:]:
                alternative=lookup_root(atlas,index_xyz,policy,hashes['atlas'],hashes['atlas_key'])
                for field in ('ARMIndex','ARMAbbreviation','ARMFullName','Hemisphere','SourceARMStatus','SourceRootVoxelXYZ'):
                    item[policy+'_'+field]=alternative[field]
                item[policy+'_label_differs']=(alternative['ARMIndex'],alternative['SourceARMStatus'])!=(current['ARMIndex'],current['SourceARMStatus'])
                item[policy+'_hemisphere_differs']=alternative['Hemisphere']!=current['Hemisphere']
            label={k:current[k] for k in LABEL_COLUMNS}
            old=labels.setdefault(current['Subregion'],label)
            if old!=label:raise ValueError('Stable ARM group has conflicting display metadata')
            rows.append(item)
            roots.append({k:item[k] for k in ('SampleID','NeuronID','uid','AnimalID','SWCPath','SWCSHA256','EvidenceSourceGroup','SourceCohort') if k in item}|{k:v for k,v in item.items() if k.startswith(('ARM','Source','Display','Hemisphere','published_','halfopen_'))})
        frames[cohort]=pd.DataFrame(rows)
    frames['combined']=pd.concat([frames['main'],frames['preview']],ignore_index=True).fillna('')
    # Preserve all pre-existing fields exactly; only the declared primary grouping changes.
    for cohort in ('main','preview'):
        old=pd.read_csv(paths[cohort+'_manifest'],dtype=str,keep_default_na=False)
        for field in old.columns:
            if field!='Subregion' and old[field].tolist()!=frames[cohort][field].tolist():raise ValueError(f'Changed original metadata: {field}')
    if any(sha256(path)!=initial[key] for key,path in paths.items()):raise ValueError('A selection/run/readback changed during preparation')
    if any(sha256(path)!=hashes[key] for key,path in assets.items()):raise ValueError('An ARM/reference input changed during preparation')
    if any(sha256(path)!=expected for path,expected in source_hashes.items()):raise ValueError('An SWC source changed during preparation')
    output.mkdir(parents=True,exist_ok=False)
    for cohort,frame in frames.items():
        folder=output/cohort;folder.mkdir()
        _write_csv(folder/'projection_manifest.csv',frame)
        local=sorted(set(frame.Subregion))
        _write_csv(folder/'label_map.csv',pd.DataFrame([labels[label] for label in local],columns=LABEL_COLUMNS))
    _write_csv(output/'label_map.csv',pd.DataFrame([labels[label] for label in sorted(labels)],columns=LABEL_COLUMNS))
    _write_csv(output/'root_readback.csv',pd.DataFrame(roots))
    outputs={str(p.relative_to(output)).replace('\\','/'):sha256(p) for p in output.rglob('*.csv')}
    report={'status':'software_verified_ARM_only_source_group_manifests','created_utc':datetime.now(timezone.utc).isoformat(),
        'inputs':{key:{'path':str(path),'sha256':initial[key]} for key,path in paths.items()}|{key:{'path':str(path),'sha256':hashes[key]} for key,path in assets.items()},
        'preparer_sha256':sha256(Path(__file__)),'helper_code_sha256':{name:sha256(ROOT/path) for name,path in {'checked_manifest':'group_analysis/scripts/build_projection_maps.py','ARM_lookup':'main_scripts/endpoint_atlas.py','SWC_validation':'main_scripts/swc_validation.py'}.items()},
        'coordinate_policy':{'primary':PRIMARY,'formula':'np.rint(declared source XYZ / IndexScaleUm); ties to even','world_coordinate_sources':'inverse reference affine before voxel assignment','alternatives':'same ARM6 only; published ceil(index)-1 and half-open floor(index+.5) are sensitivity metadata, never primary groups','export_origin_and_registration_acceptance':'unverified'},
        'atlas_policy':'Only actual pinned NMT ARM6 data/key used for primary source lookup; no other atlas file or hierarchy inference.',
        'label_schema':LABEL_COLUMNS,'hemisphere_mask_value_to_side':atlas.value_to_side,'hemisphere_contingency':atlas.contingency,
        'selection':{cohort:{'neurons':len(frame),'animals':sorted(frame.AnimalID.unique().tolist()),'groups':dict(Counter(frame.Subregion)),'lookup_status':dict(Counter(frame.SourceARMStatus))} for cohort,frame in frames.items()},
        'published_origin_label_differences':int(frames['combined'][POLICIES[1]+'_label_differs'].sum()),
        'halfopen_tie_label_differences':int(frames['combined'][POLICIES[2]+'_label_differs'].sum()),
        'source_selection_changed':False,'evidence_metadata_promoted':False,'canonical_tables_modified':False,'existing_map_or_figure_provenance_modified':False,
        'source_graph_validation':'Every selected SWC hash reverified and full seven-column topology freshly validated; one exact root read.',
        'outputs':outputs,'maps_computed':False}
    _write_json(output/'preparation_provenance.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('main-manifest','preview-manifest','main-run','main-readback','preview-run','preview-readback','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--input-root',type=Path,default=ROOT)
    parser.add_argument('--root-policy',choices=(PRIMARY,),required=True,help='Retain the audited current center-origin source lookup; alternatives remain sensitivity only.')
    args=parser.parse_args()
    result=prepare(args.main_manifest,args.preview_manifest,args.main_run,args.main_readback,args.preview_run,args.preview_readback,args.output,input_root=args.input_root,root_policy=args.root_policy)
    print(json.dumps({'status':result['status'],'selection':result['selection']},indent=2))
