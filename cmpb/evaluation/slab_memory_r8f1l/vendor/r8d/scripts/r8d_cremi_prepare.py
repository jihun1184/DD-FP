"""D5 input preparation only; refuses implicit threshold or dataset guessing.

An explicit manifest supplies source SHA256, binary-mask .npy path, source
axis order (e.g. zyx), provenance of extraction and foreground mapping.
Creates real nested 125x{256,384,512}^2 crops from one source region. A source
smaller than a target is refused; it is never tiled or resized. No GPU runs.
"""
import argparse
from pathlib import Path
from r8d_common import output, readj, sha, writej, seal


def prepare(manifest,out):
    import numpy as np
    m=readj(manifest)
    for k in ('source','sha256','axis_order','extraction_provenance','foreground_value','origin_zyx'):
        if k not in m: raise ValueError('missing explicit field '+k)
    if not m['extraction_provenance'] or sorted(m['axis_order'])!=['x','y','z']:
        raise ValueError('invalid axis/provenance')
    source=Path(m['source'])
    if sha(source)!=m['sha256']: raise ValueError('source hash mismatch')
    arr=np.load(source,mmap_mode='r',allow_pickle=False)
    if arr.ndim!=3: raise ValueError('3D mask required')
    arr=arr.transpose(*(m['axis_order'].index(a) for a in 'zyx'))
    z,y,x=m['origin_zyx']
    if any(type(v) is not int or v<0 for v in (z,y,x)): raise ValueError('invalid crop origin')
    if z+125>arr.shape[0] or y+512>arr.shape[1] or x+512>arr.shape[2]: raise ValueError('source too small for nested crop set')
    if m['foreground_value'] not in (1,255): raise ValueError('explicit foreground 1 or 255 required')
    out=output(out); out.mkdir(parents=True,exist_ok=False); records=[]
    for size in (256,384,512):
        crop=np.asarray(arr[z:z+125,y:y+size,x:x+size])
        if not np.isin(crop,[0,m['foreground_value']]).all(): raise ValueError('not the declared binary mask')
        vol=np.ascontiguousarray((crop==m['foreground_value']).transpose(2,1,0).astype(np.uint8)*255)
        p=out/f'cremi_125x{size}x{size}.npy'; np.save(p,vol,allow_pickle=False)
        records.append({'path':str(p),'sha256':sha(p),'vol_shape_xyz':list(vol.shape),'source_origin_zyx':[z,y,x]})
    writej(out/'inputs.json',{'status':'PREPARED_NOT_EXECUTED','manifest':m,'source_manifest_sha256':sha(manifest),'crops':records,
        'interpretation':'nested crops share voxels; size and content effects remain confounded; not independent subjects'})
    seal(out); return records


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(prepare(a.manifest,a.out))


if __name__=='__main__': main()
