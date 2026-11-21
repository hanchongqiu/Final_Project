"""Extract only the 2,000 genuine global prediction-extreme chart images.

Works on a local Windows/macOS/Linux data folder. No pyarrow is required;
CSV manifest was precomputed from all 1,491,887 OOS baseline predictions.

python sq4_extract_top2000.py --data-dir "D:/.../img_data" \
  --manifest top2000_from_full_oos_manifest.csv \
  --out SQ4_Top2000_Images.zip
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile
from collections import Counter
import numpy as np

H,W=64,60

def resolve_dir(path: str | Path) -> Path:
    p=Path(path).expanduser().resolve()
    candidates=(p,p/'monthly_20d',p/'data'/'monthly_20d',p/'img_data'/'monthly_20d')
    for q in candidates:
        if (q/'20d_month_has_vb_[20]_ma_2000_images.dat').is_file():return q
    raise FileNotFoundError(f'Could not find yearly .dat files below {p}. Select img_data or monthly_20d.')

def read_manifest(path:Path)->list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    required={'group','year','row_idx','Date','y_true','prob_up'}
    if not rows or not required.issubset(rows[0]):raise ValueError('Manifest columns are missing or file is empty')
    for r in rows:
        r['year']=int(r['year']);r['row_idx']=int(r['row_idx']);r['y_true']=int(r['y_true']);r['prob_up']=float(r['prob_up'])
        if r['year']<2000 or r['year']>2019 or r['row_idx']<0:raise ValueError('Year or row offset out of bounds')
    keys=[(r['year'],r['row_idx']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate images in manifest')
    groups=Counter(r['group'] for r in rows)
    print('Selected records:',len(rows),'groups:',dict(groups))
    if len(rows)==2000 and groups!={'Top1000_UP':1000,'Top1000_DOWN':1000}:
        raise ValueError('Expected 1,000 images in each of the two global extreme groups')
    return rows

def extract(data_dir:Path,rows:list[dict])->np.ndarray:
    x=np.empty((len(rows),H,W),dtype=np.uint8)
    years=sorted(set(r['year'] for r in rows))
    for y in years:
        path=data_dir/f'20d_month_has_vb_[20]_ma_{y}_images.dat'
        if not path.is_file():raise FileNotFoundError(f'Missing original image file: {path}')
        sz=path.stat().st_size
        if sz%(H*W):raise ValueError(f'Invalid .dat size for year {y}: {sz}')
        count=sz//(H*W)
        indices=np.array([i for i,r in enumerate(rows) if r['year']==y],dtype=np.int64)
        offs=np.array([rows[i]['row_idx'] for i in indices],dtype=np.int64)
        if np.any(offs>=count):raise IndexError(f'Year {y} has {count} images, but request up to {offs.max()}')
        mm=np.memmap(path,mode='r',dtype=np.uint8,shape=(count,H,W))
        x[indices]=mm[offs]
        del mm
        print(f'  {y}: {len(indices):4d} extracted from {count:,} original images')
    return x

def main(argv=None):
    ap=argparse.ArgumentParser(description='Extract 2,000 full-OOS prediction extrema from original .dat files')
    ap.add_argument('--data-dir',required=True,help='img_data, monthly_20d, or project data directory')
    ap.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'outputs'/'top2000_from_full_oos_manifest.csv')
    ap.add_argument('--out',type=Path,default=Path('SQ4_Top2000_Images.zip'))
    args=ap.parse_args(argv)
    folder=resolve_dir(args.data_dir);rows=read_manifest(args.manifest)
    imgs=extract(folder,rows)
    buf=io.BytesIO();np.save(buf,imgs,allow_pickle=False)
    mem=io.StringIO();writer=csv.DictWriter(mem,fieldnames=['group','year','row_idx','Date','y_true','prob_up']);writer.writeheader();writer.writerows(rows)
    m={'dataset':'Full 2000–2019 OOS baseline_seed42.parquet; global 1,000 largest and 1,000 smallest prob_up',
       'n':len(rows),'image_shape':[H,W],'pixel_dtype':'uint8','data_dir_at_extraction':str(folder),
       'model':'CNN20D baseline_seed42.pt', 'note':'Rows refer to original per-year .dat image offsets. Not a random sub-sample.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('images.npy',buf.getvalue());z.writestr('manifest.csv',mem.getvalue());z.writestr('extraction_info.json',json.dumps(m,ensure_ascii=False,indent=2))
    print(f'\nDONE: {args.out.resolve()} ({args.out.stat().st_size/1024/1024:.2f} MB)')
    return args.out

if __name__=='__main__':main()
