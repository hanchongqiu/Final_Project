"""Member D SQ4: verified CNN20D Grad-CAM, temporal drift and modern retraining.

Examples (from repository root):
  python experiments/exp_sq4_drift_gradcam.py sample \
    --zip-file SQ4_sample_2000_2019_50peryear.zip \
    --checkpoint baseline_seed42.pt \
    --predictions baseline_seed42.parquet \
    --sample-predictions outputs/tables/sample_manifest_with_baseline_predictions.csv

  # Requires the COMPLETE img_data/monthly_20d archive, not the sampled ZIP:
  python experiments/exp_sq4_drift_gradcam.py modern \
    --data-dir data/monthly_20d --predictions baseline_seed42.parquet

  # Exact full-OOS top-1000 per class, requiring complete image .dat files:
  python experiments/exp_sq4_drift_gradcam.py topfull \
    --data-dir data/monthly_20d --predictions baseline_seed42.parquet \
    --checkpoint baseline_seed42.pt

Implements Grad-CAM for the exact model from src/visual_alpha/models.py.
Treats dataset as observational and CAM as a post-hoc attribution, NOT causality.
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
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visual_alpha.models import CNN20D
from visual_alpha.dataset import ChartDataset,build_train_val_test_splits
from visual_alpha.trainer import train_model,predict_dataset
from visual_alpha.evaluate import evaluate_predictions

H,W=64,60

def read_parquet(path,cols=None):
    return pd.read_parquet(path,columns=cols)

def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=CNN20D()
    model.load_state_dict(ck['model_state_dict'],strict=True)
    return model.eval(),ck

def load_sample(zip_file,predictions=None,manifest_predictions=None):
    with zipfile.ZipFile(zip_file) as z:
        manifest=pd.read_csv(io.BytesIO(z.read('selection_manifest.csv')),dtype={'StockID':str})
        manifest=manifest.sort_values(['year','subset_row']).reset_index(drop=True)
        if 'y_true' in manifest.columns:manifest=manifest.rename(columns={'y_true':'y_sample_ret'})
        imgs=[]
        for year in sorted(manifest.year.unique()):
            sub=manifest.loc[manifest.year==year]
            assert np.array_equal(sub.subset_row.to_numpy(),np.arange(len(sub)))
            raw=z.read(f'monthly_20d/20d_month_has_vb_[20]_ma_{year}_images.dat')
            if len(raw)!=len(sub)*H*W:raise ValueError(f'Image-label alignment failed for {year}')
            imgs.append(np.frombuffer(raw,dtype=np.uint8).reshape(-1,H,W).copy())
    x=np.concatenate(imgs)
    if manifest_predictions and Path(manifest_predictions).exists():
        full=pd.read_csv(manifest_predictions,dtype={'StockID':str})
        full=full[['year','original_row','prob_up','y_true_y']].rename(columns={'y_true_y':'y_true'})
        frame=manifest.merge(full,on=['year','original_row'],validate='one_to_one')
    elif predictions:
        pred=read_parquet(predictions,['year','row_idx','prob_up','y_true'])
        frame=manifest.merge(pred,left_on=['year','original_row'],right_on=['year','row_idx'],how='left',validate='one_to_one')
    else:raise ValueError('Supply --predictions or --sample-predictions')
    if len(frame)!=len(x) or frame.prob_up.isna().any():raise ValueError('Missing predictions or misaligned sample')
    frame['image_index']=np.arange(len(frame));frame['period']=np.where(frame.year<2010,'2000s','2010s')
    frame['predicted_up']=(frame.prob_up>=0.5).astype(int)
    frame['confidence']=np.abs(frame.prob_up-0.5)
    frame['correct']=(frame.predicted_up==frame.y_true).astype(int)
    frame['Date']=pd.to_datetime(frame.Date)
    return x,frame

def load_full_top(data_dir,predictions,top_k):
    pred=read_parquet(predictions,['year','row_idx','prob_up','y_true','Date','StockID'])
    if pred.duplicated(['year','row_idx']).any():raise ValueError('Duplicate keys in predictions')
    up=pred.nlargest(top_k,'prob_up').copy();down=pred.nsmallest(top_k,'prob_up').copy()
    df=pd.concat([down,up],ignore_index=True)
    x=np.empty((len(df),H,W),dtype=np.uint8)
    for year in sorted(df.year.unique()):
        pos=np.flatnonzero(df.year.to_numpy()==year)
        dat_path=Path(data_dir)/f'20d_month_has_vb_[20]_ma_{year}_images.dat'
        if not dat_path.exists():raise FileNotFoundError(f'Cannot run full-top analysis without {dat_path}')
        sz=dat_path.stat().st_size
        if sz%(H*W):raise ValueError(f'Invalid {dat_path} size')
        mm=np.memmap(dat_path,dtype=np.uint8,mode='r',shape=(sz//(H*W),H,W))
        rows=df.row_idx.to_numpy()[pos]
        if rows.min()<0 or rows.max()>=len(mm):raise ValueError(f'Invalid original row index {year}')
        x[pos]=mm[rows]
    df['image_index']=np.arange(len(df));df['period']=np.where(df.year<2010,'2000s','2010s')
    df['predicted_up']=(df.prob_up>=0.5).astype(int);df['confidence']=np.abs(df.prob_up-0.5)
    df['correct']=(df.predicted_up==df.y_true).astype(int);df['Date']=pd.to_datetime(df.Date)
    return x,df

def check_parity(model,x,frame,batch_size=48,max_n=None):
    model.eval()
    if max_n and max_n<len(x):idx=np.random.default_rng(42).choice(len(x),max_n,replace=False)
    else:idx=np.arange(len(x))
    out=[]
    with torch.inference_mode():
        for b in range(0,len(idx),batch_size):
            xx=torch.from_numpy(np.ascontiguousarray(x[idx[b:b+batch_size]])).float().unsqueeze(1)/255.0
            pred=model(xx).softmax(1)[:,1]
            out.append(pred.numpy())
    estimate=np.concatenate(out)
    actual=frame.prob_up.to_numpy()[idx]
    result={'n':int(len(idx)), 'mae':float(np.mean(np.abs(estimate-actual))),
            'max_abs_error':float(np.max(np.abs(estimate-actual))),
            'pearson':float(np.corrcoef(estimate,actual)[0,1]),
            'normalization':'uint8 / 255', 'architecture':'visual_alpha.models.CNN20D'}
    if result['mae']>1e-5 or result['max_abs_error']>1e-4 or result['pearson']<0.999:
        raise RuntimeError(f'Checkpoint parity check failed; REFUSING Grad-CAM: {result}')
    return result

def balanced_subset(frame,n_per_era=500,seed=42):
    a=frame.copy()
    a['conf_bin']=pd.qcut(a.confidence,5,labels=False,duplicates='drop').fillna(0).astype(int)
    rng=np.random.default_rng(seed)
    chosen=[];bins=sorted(a.conf_bin.unique());quota=max(1,n_per_era//(len(bins)*2))
    for cls in (0,1):
        for bin_id in bins:
            a0=a[(a.period=='2000s')&(a.predicted_up==cls)&(a.conf_bin==bin_id)]
            a1=a[(a.period=='2010s')&(a.predicted_up==cls)&(a.conf_bin==bin_id)]
            n=min(len(a0),len(a1),quota)
            if n:
                chosen.extend(rng.choice(a0.index.to_numpy(),size=n,replace=False))
                chosen.extend(rng.choice(a1.index.to_numpy(),size=n,replace=False))
    return a.loc[sorted(chosen)].copy()

def gradcam(model,x,frame,batch_size=16):
    """Predicted-class logit Grad-CAM: grads w.r.t. block3[0] conv activations."""
    model.eval()
    hook_layer=model.last_conv_layer
    acts=[];features=[];all_cams=[];all_images=[];idxs=[]
    handle=hook_layer.register_forward_hook(lambda m,i,o:acts.append(o))
    try:
        for beg in range(0,len(frame),batch_size):
            f=frame.iloc[beg:beg+batch_size];ii=f.image_index.to_numpy(dtype=int)
            xx=torch.from_numpy(np.ascontiguousarray(x[ii])).float().unsqueeze(1)/255.0
            targets=torch.tensor(f.predicted_up.to_numpy(dtype=np.int64),dtype=torch.long)
            acts.clear()
            with torch.enable_grad():
                logits=model(xx)
                score=logits.gather(1,targets[:,None]).sum()
                activation=acts[-1]
                grads=torch.autograd.grad(score,activation)[0]
                weights=grads.mean((2,3),keepdim=True)
                heat=F.relu((weights*activation).sum(1,keepdim=True))
                ups=F.interpolate(heat,size=(H,W),mode='bilinear',align_corners=False)[:,0].detach().cpu().numpy()
            # Mass normalization, not max normalization: ROI fractions sum to one
            sums=ups.sum(axis=(1,2),keepdims=True)
            valid=(sums[:,0,0]>1e-12)
            norm=np.divide(ups,sums,out=np.zeros_like(ups),where=sums>1e-12)
            for j,(_,r) in enumerate(f.iterrows()):
                cam=norm[j]; ok=bool(valid[j]);
                features.append(dict(image_index=int(ii[j]),year=int(r.year),Date=str(r.Date.date()),
                    StockID=str(r.StockID),period=r.period,predicted_up=int(r.predicted_up),
                    y_true=int(r.y_true),correct=int(r.correct),prob_up=float(r.prob_up),
                    confidence=float(r.confidence),valid_cam=int(ok),
                    price_share=float(cam[:51,:].sum()) if ok else np.nan,
                    separator_share=float(cam[51,:].sum()) if ok else np.nan,
                    volume_share=float(cam[52:,:].sum()) if ok else np.nan,
                    recent_10d_share=float(cam[:,30:].sum()) if ok else np.nan,
                    recent_5d_share=float(cam[:,45:].sum()) if ok else np.nan,
                    nonzero_pixel_frac=float((x[ii[j]]>0).mean())))
            all_cams.append(norm.astype(np.float32)); idxs.extend(ii)
    finally:handle.remove()
    arr=np.concatenate(all_cams)
    assert len(arr)==len(frame)
    return pd.DataFrame(features),arr

def moving_month_bootstrap(frame,metric,n_boot=3000,block_len=3,seed=42):
    """Consecutive-month moving-block bootstrap. Confidence matched cohort fixed.

    Resampling monthly panels preserves within-month stock dependence and (approximately)
    neighboring-month serial dependence. Returns later-minus-earlier by predicted class.
    """
    rng=np.random.default_rng(seed)
    results=[]
    for cls in (0,1):
        group_means={};boots={};ns={}
        for era in ('2000s','2010s'):
            s=frame[(frame.period==era)&(frame.predicted_up==cls)].copy()
            s=s[np.isfinite(s[metric])]
            ns[era]=len(s);group_means[era]=float(s[metric].mean())
            s['month']=pd.PeriodIndex(pd.to_datetime(s.Date),freq='M')
            a=s.groupby('month',sort=True)[metric].agg(['sum','count'])
            # Periods are ordinal and can be sparse (sampled ZIP). Expand missing months.
            idx=pd.period_range(a.index.min(),a.index.max(),freq='M')
            a=a.reindex(idx,fill_value=0)
            sums=a['sum'].to_numpy(); counts=a['count'].to_numpy();M=len(a)
            nblocks=int(np.ceil(M/block_len))
            shifts=np.arange(block_len)
            bt=np.empty(n_boot,dtype=float)
            for j in range(n_boot):
                starts=rng.integers(0,M,size=nblocks)
                sel=((starts[:,None]+shifts[None,:])%M).ravel()[:M]
                denom=counts[sel].sum();bt[j]=sums[sel].sum()/denom if denom else np.nan
            boots[era]=bt
        delta=boots['2010s']-boots['2000s']
        ci=np.nanquantile(delta,[.025,.975]);d=group_means['2010s']-group_means['2000s']
        results.append({'metric':metric,'predicted_class':'UP' if cls else 'DOWN',
           'n_2000s':ns['2000s'],'n_2010s':ns['2010s'],
           'mean_2000s':group_means['2000s'],'mean_2010s':group_means['2010s'],
           'late_minus_early':d,'ci95_lo':float(ci[0]),'ci95_hi':float(ci[1]),
           'bootstrap':'3-month moving-block, resampled per era'})
    return pd.DataFrame(results)

def plots(frame,cam,images,yearly_csv,out,source):
    out.mkdir(parents=True,exist_ok=True)
    means={}
    for era in ('2000s','2010s'):
        for cls in (0,1):
            mask=(frame.period==era)&(frame.predicted_up==cls)
            means[(era,cls)]=cam[mask.to_numpy()].mean(axis=0) if mask.any() else np.zeros((H,W))
    mx=max(z.max() for z in means.values())
    # Separate 2x2 heatmap with identical color scale.
    fig,axs=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for row,era in enumerate(('2000s','2010s')):
        for col,cls in enumerate((1,0)):
            ax=axs[row,col];z=means[(era,cls)]
            im=ax.imshow(z,cmap='magma',vmin=0,vmax=mx,aspect='auto',interpolation='nearest')
            ax.axhline(50.5,ls='--',lw=.9,color='cyan');ax.axhline(51.5,ls=':',lw=.6,color='cyan')
            ax.axvline(29.5,ls='--',lw=.8,color='white',alpha=.8)
            n=int(((frame.period==era)&(frame.predicted_up==cls)).sum())
            ax.set(title=f'{era} | Predicted {"UP" if cls else "DOWN"} | n={n}',xlabel='Older days  ←  time  →  Recent days',ylabel='Image pixel row')
    fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,label='Mean Grad-CAM probability mass / pixel')
    fig.suptitle('Authentic baseline seed42 attribution · matched, image-level normalized Grad-CAM',fontsize=13)
    fig.savefig(out/'sq4_mean_gradcam_era_class.png',dpi=200,bbox_inches='tight');plt.close(fig)
    # Four most confident CORRECT predictions within the observational sample.
    cases=[]
    for era,cls in [('2000s',1),('2000s',0),('2010s',1),('2010s',0)]:
        f=frame[(frame.period==era)&(frame.predicted_up==cls)&(frame.correct==1)]
        if len(f):cases.append(f.sort_values('confidence',ascending=False).iloc[0])
    fig,axs=plt.subplots(len(cases),2,figsize=(8,2.7*len(cases)),layout='constrained',squeeze=False)
    for j,r in enumerate(cases):
        idx=int(r.image_index);raw=images[idx]
        zz=cam[frame.image_index.to_numpy()==idx][0]
        axs[j,0].imshow(raw,cmap='gray',vmin=0,vmax=255,aspect='auto',interpolation='nearest')
        axs[j,1].imshow(raw,cmap='gray',vmin=0,vmax=255,aspect='auto',interpolation='nearest')
        axs[j,1].imshow(zz,cmap='jet',alpha=.68*np.clip(zz/max(float(zz.max()),1e-12),0,1),aspect='auto')
        for ax in axs[j]:ax.axhline(50.5,color='cyan',ls='--',lw=.6);ax.set_xticks([]);ax.set_yticks([])
        axs[j,0].set_title(f'{r.period} {"UP" if r.predicted_up else "DOWN"} | {r.Date} | P(up)={r.prob_up:.3f}',fontsize=10)
        axs[j,1].set_title('Predicted-class Grad-CAM overlay',fontsize=10)
    fig.suptitle('High-confidence correct cases within 1,000-image convenience sample (NOT global OOS extremes)')
    fig.savefig(out/'sq4_case_studies.png',dpi=200,bbox_inches='tight');plt.close(fig)
    # Figure 4: faithful, near-square 64x60 attribution maps (no aspect distortion).
    fig=plt.figure(figsize=(13.8,9.2),layout='constrained')
    gs=fig.add_gridspec(2,3,width_ratios=[1.43,1,1],height_ratios=[1,1],wspace=.09,hspace=.15)
    ax=fig.add_subplot(gs[0,0])
    try:
        annual=pd.read_csv(yearly_csv)
        for mdl,color in [('Seed42','#1769aa'),('Ensemble5','#ed8734')]:
            tab=annual[annual.model==mdl].sort_values('year')
            ax.plot(tab.year,tab.auc,label=mdl,marker='o',markersize=3,lw=1.8,color=color)
        ax.set_xlim(1999.5,2019.5);ax.set_ylim(.49,.61)
        ax.axvspan(2009.5,2019.5,color='#e8eff7',alpha=.8)
        ax.axhline(.5,ls=':',lw=1,c='gray');ax.legend(frameon=False,fontsize=9)
        ax.set_ylabel('AUC');ax.set_xlabel('Out-of-sample year')
        ax.grid(axis='y',alpha=.17)
        ax.set_title('(a) Temporal OOS AUC',loc='left',fontweight='bold',fontsize=11)
    except (FileNotFoundError,KeyError) as e:
        ax.text(.5,.5,f'Annual metrics unavailable: {e}',ha='center',transform=ax.transAxes)
    ax=fig.add_subplot(gs[1,0])
    g=frame.groupby(['period','predicted_up'],observed=True)['volume_share'].mean().unstack()
    xaxis=np.arange(2);barw=.32
    for i,cls in enumerate((0,1)):
        v=[g.loc[era,cls] if era in g.index and cls in g.columns else 0 for era in ('2000s','2010s')]
        ax.bar(xaxis+(i-.5)*barw,v,width=barw,
          label='Predicted DOWN' if cls==0 else 'Predicted UP',alpha=.88)
    ax.set_xticks(xaxis,['2000s','2010s']);ax.set_ylim(0,.34)
    ax.axhline(12/64,ls=':',color='gray',lw=1,label='12 / 64 uniform reference')
    ax.set_ylabel('Volume CAM mass / total CAM mass')
    ax.set_title('(b) CAM in volume region',loc='left',fontweight='bold',fontsize=11)
    ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.14)
    for col,cls in [(1,1),(2,0)]:
        for row,era in enumerate(('2000s','2010s')):
            ax=fig.add_subplot(gs[row,col]);a=means[(era,cls)]
            ax.imshow(a,cmap='magma',vmin=0,vmax=mx,aspect='equal',interpolation='nearest')
            ax.axhline(50.5,color='cyan',ls='--',lw=.75)
            ax.axvline(29.5,color='white',ls='--',lw=.75)
            ax.set_ylabel(f'{era} pixel row');ax.set_xlabel('Old  ←  time  →  recent')
            ax.set_xticks([0,15,30,45,59]);ax.set_yticks([0,20,40,63])
            if row==0:ax.set_title(f'({"c" if cls else "d"}) Mean CAM: predicted {"UP" if cls else "DOWN"}',
                                  loc='left',fontweight='bold',fontsize=11)
    fig.suptitle('SQ4 | Fixed-CNN predictive decay and authentic Grad-CAM (matched chart sample)',
                 fontsize=14,fontweight='bold')
    fig.savefig(out/'fig4_sq4_drift_gradcam.png',dpi=195,bbox_inches='tight')
    fig.savefig(out/'fig4_sq4_drift_gradcam.pdf',bbox_inches='tight');plt.close(fig)

def run_cam(args,full=False):
    if full:
        images,all_frame=load_full_top(args.data_dir,args.predictions,args.top_k)
        selected=all_frame.copy()
        source='Top-'+str(args.top_k)+' from ALL full-OOS predictions'
    else:
        images,all_frame=load_sample(args.zip_file,args.predictions,args.sample_predictions)
        selected=balanced_subset(all_frame,args.per_era,args.seed)
        source='Confidence/class-matched convenience sample (NOT full-OOS top-1000)'
    model,ck=load_checkpoint(args.checkpoint)
    parity=check_parity(model,images,all_frame,max_n=None if not full else 120)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'model_parity_check.json').write_text(json.dumps({'source':source,'checkpoint':str(args.checkpoint),'checkpoint_best_epoch':ck.get('best_epoch'),**parity},indent=2),encoding='utf8')
    features,cams=gradcam(model,images,selected,args.batch_size)
    features.to_csv(out/'sq4_gradcam_individual_metrics.csv',index=False)
    mean=features.groupby(['period','predicted_up'],observed=True).agg(
       n=('image_index','size'),valid_cam_fraction=('valid_cam','mean'),
       price_share=('price_share','mean'),volume_share=('volume_share','mean'),
       separator_share=('separator_share','mean'),recent_10d_share=('recent_10d_share','mean'),
       recent_5d_share=('recent_5d_share','mean'),avg_pixel_density=('nonzero_pixel_frac','mean')).reset_index()
    mean.to_csv(out/'sq4_gradcam_group_summary.csv',index=False)
    ci=pd.concat([moving_month_bootstrap(features,metric,seed=args.seed) for metric in ('volume_share','price_share','recent_10d_share','recent_5d_share')],ignore_index=True)
    ci.to_csv(out/'sq4_gradcam_month_bootstrap.csv',index=False)
    np.savez_compressed(out/'sq4_gradcam_arrays.npz',cams=cams.astype(np.float16),image_index=features.image_index.to_numpy())
    plots(features,cams,images,args.yearly_csv,out,source)
    (out/'sample_methodology.json').write_text(json.dumps({'source':source,'selected_n':len(features),'years':sorted(features.year.unique().tolist()),
       'class_target':'predicted class logit','conv_layer':'block3[0]',
       'per_image_cam_normalization':'ReLU CAM / sum of all 64x60 pixels',
       'confidence_matching':not full,'volume_rows':'52:64','price_rows':'0:51',
       'recent10_columns':'30:60', 'input_normalization':'uint8/255',
       'caveat':'Grad-CAM is post-hoc attribution, not proof of a causal financial mechanism; sample mode is NOT full-OOS top1000.'},indent=2),encoding='utf8')
    print('PARITY',json.dumps(parity,indent=2));print('GROUPS\n',mean.to_string(index=False));print('CI\n',ci.to_string(index=False));print('Outputs:',out)

def run_modern(args):
    """Strictly fair 2006-12 retrain of original architecture, held-out 2013-19."""
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_dir)
    for year in list(range(2006,2013))+list(range(2013,2020)):
        p=data/f'20d_month_has_vb_[20]_ma_{year}_images.dat'
        if not p.exists():raise FileNotFoundError(f'Full data required for modern experiment: {p}')
    spl=build_train_val_test_splits(data_dir=data,train_val_years=range(2006,2013),test_years=range(2013,2020),split_seed=42)
    if len(spl.train)<100000 or len(spl.test)<100000:
        raise RuntimeError('Modern retraining REQUIRES COMPLETE raw data, not the 50-image-per-year SQ4 sample')
    from visual_alpha.utils import get_device
    model=CNN20D()
    ckpt=out/'sq4_modern_seed42.pt'
    res=train_model(model,spl.train,spl.val,seed=42,lr=1e-5,batch_size=args.train_batch_size,
       eval_batch_size=args.eval_batch_size,max_epochs=args.max_epochs,patience=2,
       checkpoint_path=ckpt,device=get_device(),verbose=True)
    predictions=predict_dataset(model,spl.test,batch_size=args.eval_batch_size,device=get_device())
    predictions.to_parquet(out/'sq4_modern_seed42_predictions_2013_2019.parquet',index=False)
    base=read_parquet(args.predictions)
    base=base[base.year.between(2013,2019)].reset_index(drop=True)
    if len(base)!=len(predictions):raise ValueError('Different test set sample counts')
    if not np.array_equal(base[['year','row_idx']].to_numpy(),predictions[['year','row_idx']].to_numpy()):
        raise ValueError('Baseline and Modern test keys are not identically ordered')
    rows=[]
    for name,df in [('Model_1990s',base),('Model_Modern',predictions)]:
        rows.append({'model':name, 'train_years':'1993-1999' if name=='Model_1990s' else '2006-2012',
                    'test_years':'2013-2019',**evaluate_predictions(df)})
    pd.DataFrame(rows).to_csv(out/'tab3_concept_drift.csv',index=False)
    res.history.to_csv(out/'modern_training_history.csv',index=False)
    print('MODERN VS HISTORICAL\n',pd.DataFrame(rows).to_string(index=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode',choices=['sample','topfull','modern'])
    ap.add_argument('--zip-file',type=Path,default=ROOT/'SQ4_sample_2000_2019_50peryear.zip')
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'baseline_seed42.pt')
    ap.add_argument('--predictions',type=Path,default=ROOT/'baseline_seed42.parquet')
    ap.add_argument('--sample-predictions',type=Path,default=ROOT/'outputs/tables/sample_manifest_with_baseline_predictions.csv')
    ap.add_argument('--yearly-csv',type=Path,default=ROOT/'outputs/tables/sq4_yearly_performance.csv')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data/monthly_20d')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/sq4')
    ap.add_argument('--per-era',type=int,default=500)
    ap.add_argument('--top-k',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-batch-size',type=int,default=256)
    ap.add_argument('--eval-batch-size',type=int,default=1024)
    ap.add_argument('--max-epochs',type=int,default=15)
    args=ap.parse_args()
    if args.mode=='modern':run_modern(args)
    else:run_cam(args,full=args.mode=='topfull')

if __name__=='__main__':main()
