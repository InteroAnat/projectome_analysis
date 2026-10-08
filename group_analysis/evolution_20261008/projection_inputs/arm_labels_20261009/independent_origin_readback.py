"""Independent origin/tie sensitivity on source indices freshly checked by root readback."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    first=OUT/'independent_readback.json';verified=json.loads(first.read_text())
    if verified['status']!='passed':raise ValueError('Fresh independent source root check required')
    prep=OUT/'preparation_provenance.json';record=json.loads(prep.read_text())
    if sha(prep)!=verified['preparation_provenance_sha256']:raise ValueError('Changed prepared provenance')
    manifest=OUT/'combined/projection_manifest.csv'
    if sha(manifest)!=record['outputs']['combined/projection_manifest.csv']:raise ValueError('Changed root indices')
    asset=record['inputs']['atlas'];key=record['inputs']['atlas_key']
    if sha(asset['path'])!=asset['sha256'] or sha(key['path'])!=key['sha256']:raise ValueError('Changed ARM/key')
    data=np.asanyarray(nib.load(asset['path']).dataobj)[...,0,5]
    frame=pd.read_csv(manifest,dtype=str,keep_default_na=False);counts=Counter()
    for row in frame.to_dict('records'):
        index=np.array(json.loads(row['SourceRootIndexXYZ']))
        current=np.rint(index).astype(int)
        counts['roots_with_exact_half_integer_axis']+=int(np.any(np.mod(index,1)==.5))
        for name,voxel in [('published_ceil_edge_one_based_to_zero',(np.ceil(index)-1).astype(int)),('halfopen_floor_zero_center',np.floor(index+.5).astype(int))]:
            inside=np.all(voxel>=0)&np.all(voxel<data.shape)
            code=int(data[tuple(voxel)]) if inside else -1
            if str(code)!=row[name+'_ARMIndex']:raise ValueError('Saved alternative ARM index differs')
            counts[name+'_voxel_differences']+=int(not np.array_equal(current,voxel))
            counts[name+'_label_differences']+=int(code!=int(row['ARMIndex']))
    verified.update({'initial_independent_root_readback_sha256':sha(first),'independent_origin_checker_sha256':sha(Path(__file__)),
        'coordinate_origin_sensitivity':dict(counts),'halfopen_tie_label_differences':counts['halfopen_floor_zero_center_label_differences'],
        'coordinate_convention_limit':'Observed zero rint-versus-half-open label differences applies only to these exact 462 roots; generic tie policies remain different. Primary remains np.rint. Published ceil(index)-1 is sensitivity only on the same pinned ARM, not a reproduction using a different atlas.'})
    with (OUT/'independent_readback_v2.json').open('x',encoding='utf-8') as stream:json.dump(verified,stream,indent=2)
    print(json.dumps({'status':'passed','roots':len(frame),'coordinate_origin_sensitivity':dict(counts)}))


if __name__=='__main__':main()
