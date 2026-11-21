"""Finish the exact Top-1000 UP and Top-1000 DOWN Grad-CAM analysis.

Uses the small SQ4_Top2000_Images.zip created on the machine holding original .dat data.
Model inference is checked against baseline_seed42.parquet probabilities in the manifest.

  python sq4_run_top2000_gradcam.py --archive SQ4_Top2000_Images.zip \
    --checkpoint baseline_seed42.pt --repo Final_Project-main --output outputs/sq4_top2000
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

H,W=64,60

def archive_data(path):
    with zipfile.ZipFile(path) as z:
        x=np.load(io.BytesIO(z.read('images.npy')),allow_pickle=False)
        df=pd.read_csv(io.BytesIO(z.read('manifest.csv')),dtype={'group':str})
    if x.shape!=(len(df),H,W) or x.dtype!=np.uint8:raise ValueError('Archive image dimensions or dtype invalid')
    if df.duplicated(['year','row_idx']).any():raise ValueError('Archive image indices not unique')
    df['Date']=pd.to_datetime(df['Date'])
    if not set(df.group.unique()).issubset({'Top1000_UP','Top1000_DOWN'}):
        raise ValueError('Expected Top1000_UP / Top1000_DOWN groups')
    df['image_index']=np.arange(len(df))
    df['period']=np.where(df.year<2010,'2000s','2010s')
    df['predicted_up']=(df.prob_up>=.5).astype('int8')
    df['correct']=(df.predicted_up==df.y_true).astype('int8')
    df['confidence']=np.abs(df.prob_up-.5)
    df['StockID']='not_required_for_parity'
    if (df.group.eq('Top1000_UP') & df.predicted_up.eq(0)).any() or (df.group.eq('Top1000_DOWN') & df.predicted_up.eq(1)).any():
        raise ValueError('Up/Down group probability inconsistent')
    return x,df

def ci_by_month(df,n_boot=3000,seed=42):
    """Whole-month cluster bootstrap for UP minus DOWN, not randomized causal inference."""
    rng=np.random.default_rng(seed)
    metrics=['price_share','volume_share','recent_10d_share','recent_5d_share']
    frame=df[df.valid_cam.eq(1)].copy()
    frame['month']=pd.to_datetime(frame['Date']).dt.to_period('M').astype(str)
    periods=sorted(frame.month.unique())
    results=[]
    for k in metrics:
        a=frame.groupby(['month','group'])[k].agg(['sum','count']).unstack('group').reindex(periods,fill_value=0)
        sums={g:a[('sum',g)].fillna(0).to_numpy(float) if ('sum',g) in a.columns else np.zeros(len(periods)) for g in ('Top1000_UP','Top1000_DOWN')}
        ns={g:a[('count',g)].fillna(0).to_numpy(float) if ('count',g) in a.columns else np.zeros(len(periods)) for g in ('Top1000_UP','Top1000_DOWN')}
        means={g:sums[g].sum()/ns[g].sum() if ns[g].sum() else np.nan for g in sums}
        diffs=[];blocklen=3;nblocks=int(np.ceil(len(periods)/blocklen))
        for _ in range(n_boot):
            starts=rng.integers(0,len(periods),size=nblocks)
            idx=((starts[:,None]+np.arange(blocklen)[None,:])%len(periods)).ravel()[:len(periods)]
            d={g:sums[g][idx].sum()/ns[g][idx].sum() if ns[g][idx].sum() else np.nan for g in sums}
            diffs.append(d['Top1000_UP']-d['Top1000_DOWN'])
        lo,hi=np.nanquantile(diffs,[.025,.975])
        results.append({'metric':k,'up_mean':means['Top1000_UP'],'down_mean':means['Top1000_DOWN'],
                'up_minus_down':means['Top1000_UP']-means['Top1000_DOWN'],
                'ci95_lower':lo,'ci95_upper':hi,
                'method':'3-month circular block bootstrap across calendar months; observational cohort comparison'})
    return pd.DataFrame(results)

def make_figures(x,df,feat,cams,out):
    # Grad-CAM scores are individually normalized; zero-mass CAMs omitted from means.
    group_names=('Top1000_UP','Top1000_DOWN')
    maps={}
    for grp in group_names:
        idx=(feat['group']==grp)&feat.valid_cam.eq(1)
        maps[grp]=cams[idx.to_numpy()].mean(axis=0) if idx.any() else np.zeros((H,W))
    vmax=max(float(maps[g].max()) for g in maps)
    fig,axs=plt.subplots(1,2,figsize=(10,5.7),layout='constrained')
    for ax,g in zip(axs,group_names):
        im=ax.imshow(maps[g],cmap='magma',vmin=0,vmax=vmax,aspect='equal',interpolation='nearest')
        ax.axhline(50.5,color='cyan',ls='--',lw=1,label='Price/volume separator')
        ax.axhline(51.5,color='cyan',ls=':',lw=.6)
        ax.axvline(29.5,color='white',ls='--',lw=.6)
        ax.set_xticks([0,15,30,45,59],['1','6','11','16','20'])
        ax.set_xlabel('Trading day (oldest → most recent)')
        ax.set_ylabel('Image row')
        n=int((df.group==g).sum());valid=int(((feat.group==g)&feat.valid_cam.eq(1)).sum())
        ax.set_title(('High-confidence UP' if g.endswith('UP') else 'High-confidence DOWN')+f'\nselected n={n}, valid CAM={valid}')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Mean normalized Grad-CAM intensity',fraction=.03)
    fig.suptitle('Full 2000–2019 OOS: global Top-1000 per predicted class',fontweight='bold')
    fig.savefig(out/'fig4_sq4_top1000_global_gradcam.png',dpi=220,bbox_inches='tight')
    fig.savefig(out/'fig4_sq4_top1000_global_gradcam.pdf',bbox_inches='tight')
    plt.close(fig)
    # Four correct extreme examples, ranked by confidence within each era and direction.
    case_rows=[]
    for grp in group_names:
        for era in ('2000s','2010s'):
            t=feat[(feat.group==grp)&feat.period.eq(era)&feat.correct.eq(1)&feat.valid_cam.eq(1)]
            if len(t):case_rows.append(t.sort_values('confidence',ascending=False).iloc[0])
    if case_rows:
        fig,axes=plt.subplots(len(case_rows),2,figsize=(8,2.3*len(case_rows)),layout='constrained',squeeze=False)
        id_to_cam={i:j for j,i in enumerate(feat.image_index.to_numpy())}
        for k,r in enumerate(case_rows):
            i=int(r.image_index);cm=cams[id_to_cam[i]];raw=x[i]
            for ax in axes[k]:
                ax.imshow(raw,vmin=0,vmax=255,cmap='gray',interpolation='nearest')
                ax.axhline(50.5,color='cyan',ls='--',lw=.6)
                ax.set_xticks([]);ax.set_yticks([])
            axes[k,1].imshow(cm,cmap='jet',alpha=.7*np.clip(cm/max(float(cm.max()),1e-12),0,1))
            axes[k,0].set_title(f'{r.group}, {r.period}, {r.Date}, p={r.prob_up:.4f}',fontsize=9)
            axes[k,1].set_title('Predicted-class Grad-CAM',fontsize=9)
        fig.savefig(out/'sq4_full_top1000_case_studies.png',dpi=210,bbox_inches='tight')
        plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description='Validate and compute Grad-CAM for global Top1000 up/down cases')
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--repo',required=True,type=Path,help='Unzipped Final_Project-main source folder')
    p.add_argument('--output',default=Path('outputs/sq4_top2000'),type=Path)
    p.add_argument('--batch-size',type=int,default=12)
    args=p.parse_args(argv)
    sys.path.insert(0,str(args.repo/'src'))
    from visual_alpha.models import CNN20D
    # Import original verified attribution implementation for consistent methods.
    import importlib.util
    script=Path(__file__).with_name('exp_sq4_drift_gradcam.py')
    spec=importlib.util.spec_from_file_location('sq4_experiment',script)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    x,df=archive_data(args.archive)
    model,ck=mod.load_checkpoint(args.checkpoint)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    parity=mod.check_parity(model,x,df,max_n=None)
    print('Parity check on all images:',json.dumps(parity,indent=2))
    feat,cams=mod.gradcam(model,x,df,args.batch_size)
    feat=feat.merge(df[['image_index','group']],on='image_index',validate='one_to_one')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    feat.to_csv(out/'sq4_top2000_individual_gradcam.csv',index=False)
    summary=feat[feat.valid_cam.eq(1)].groupby('group').agg(
         n_valid=('image_index','size'),price_mass=('price_share','mean'),volume_mass=('volume_share','mean'),
         recent_10d_mass=('recent_10d_share','mean'),recent_5d_mass=('recent_5d_share','mean')).reset_index()
    summary=summary.merge(feat.groupby('group').size().rename('n_selected'),on='group',validate='one_to_one')
    summary.to_csv(out/'sq4_top2000_global_roi.csv',index=False)
    ci_by_month(feat).to_csv(out/'sq4_top2000_bootstrap.csv',index=False)
    make_figures(x,df,feat,cams,out)
    np.savez_compressed(out/'sq4_top2000_cam_arrays.npz',cams=cams.astype(np.float16))
    (out/'top2000_verification.json').write_text(json.dumps({
        'n':len(df),'group_counts':df.group.value_counts().to_dict(), 'label_source':'baseline_seed42.parquet y_true',
        'ranking_source':'All 1,491,887 OOS predictions, global top/bottom prob_up',
        'probability_parity':parity,'checkpoint_best_epoch':ck.get('best_epoch'),
        'limitation':'Attributions are post-hoc, not evidence of causal market mechanisms.'},indent=2,default=str))
    print('Completed Top-1000 each direction. Outputs:',out.resolve())
    print(summary.to_string(index=False))

if __name__=='__main__':main()
