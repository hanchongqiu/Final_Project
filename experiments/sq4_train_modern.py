"""Actual full-sample SQ4 modern-era retraining, no sampling substitution.

Trains CNN20D on 2006-2012 (70:30 train:val) and evaluates on 2013-2019
alongside the ORIGINAL 1993-1999 baseline. Requires original 14 annual
.dat + .feather pairs on this computer. Seed 42, Adam 1e-5, 15 epoch limit,
patience 2, original preprocessing uint8 / 255.

Example:
  python sq4_train_modern.py --repo Final_Project-main \
    --data-dir "D:/.../img_data" --baseline baseline_seed42.parquet \
    --out outputs/sq4_modern
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def resolve_data(path:Path):
    p=path.expanduser().resolve()
    for q in (p,p/'monthly_20d',p/'data'/'monthly_20d',p/'img_data'/'monthly_20d'):
        if (q/'20d_month_has_vb_[20]_ma_2006_images.dat').exists():return q
    raise FileNotFoundError(f'Could not find 2006 original image .dat under {p}')

def verify_years(directory,lo=2006,hi=2019):
    report=[]
    for year in range(lo,hi+1):
        dat=directory/f'20d_month_has_vb_[20]_ma_{year}_images.dat'
        fea=directory/f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather'
        if not dat.is_file() or not fea.is_file():raise FileNotFoundError(f'Missing raw original files for {year}: {dat} {fea}')
        count,rem=divmod(dat.stat().st_size,64*60)
        if rem:raise ValueError(f'Incorrect raw image size for {year}')
        if count<1000:raise ValueError(f'Found suspiciously small file for {year}: {count} images; full original required')
        report.append({'year':year,'image_rows':int(count),'dat_bytes':int(dat.stat().st_size),'feather_bytes':int(fea.stat().st_size)})
    return report

def month_paired_bootstrap(merged,n_boot=3000,seed=42,block_months=3):
    """Unweighted calendar-month performance differences; respects broad time clustering."""
    merged=merged.copy()
    merged['month']=pd.to_datetime(merged.Date).dt.to_period('M')
    rec=[]
    for month,g in merged.groupby('month',sort=True):
        y=g.y_true.to_numpy(int);p0=g.prob_up_old.to_numpy(float);p1=g.prob_up_new.to_numpy(float)
        d={'month':str(month),'n':len(g)}
        for metric in ('acc','brier','auc'):
            def stat(p):
                if metric=='acc':return np.mean((p>=.5)==y)
                if metric=='brier':return np.mean((p-y)**2)
                return roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
            d[f'{metric}_1990s']=stat(p0);d[f'{metric}_modern']=stat(p1)
            d[f'{metric}_delta']=d[f'{metric}_modern']-d[f'{metric}_1990s']
        rec.append(d)
    months=pd.DataFrame(rec)
    rng=np.random.default_rng(seed)
    out=[]
    for metric in ('auc','acc','brier'):
        diffs=months[f'{metric}_delta'].to_numpy(float);M=len(diffs)
        m=math.ceil(M/block_months);samples=np.empty(n_boot)
        for j in range(n_boot):
            starts=rng.integers(0,M,size=m)
            take=((starts[:,None]+np.arange(block_months)[None,:])%M).ravel()[:M]
            samples[j]=np.nanmean(diffs[take])
        lo,hi=np.nanquantile(samples,[.025,.975]);
        out.append({'metric':metric,'monthly_equal_weight_delta_modern_minus_1990s':float(np.nanmean(diffs)),
                    'ci95_lower':float(lo),'ci95_upper':float(hi),
                    'n_months':M,'bootstrap':'3-month moving-block (monthly means, same test samples)'})
    return months,pd.DataFrame(out)

def generate_annual_figure(df,out):
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for metric,ax in [('auc',axs[0]),('accuracy_pct',axs[1])]:
        for name,label in [('Model_1990s','1993–1999-trained'),('Model_Modern','2006–2012-trained')]:
            t=df[df.model.eq(name)].sort_values('year')
            ax.plot(t.year,t[metric],'-o',lw=2,ms=4,label=label)
        ax.axhline(.5 if metric=='auc' else 50.,ls=':',color='gray')
        ax.set(xlabel='OOS test year',ylabel='AUC' if metric=='auc' else 'Accuracy (%)',title=('2013–2019 AUC' if metric=='auc' else '2013–2019 classification accuracy'))
        ax.set_xticks(range(2013,2020));ax.grid(alpha=.18)
    axs[0].legend(loc='best',fontsize=8,frameon=False)
    fig.savefig(out/'fig_sq4_modern_vs_historical.png',dpi=210,bbox_inches='tight')
    fig.savefig(out/'fig_sq4_modern_vs_historical.pdf',bbox_inches='tight')
    plt.close(fig)

def main(argv=None):
    ap=argparse.ArgumentParser(description='SQ4: train CNN20D on all 2006-12 and test on 2013-19')
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('outputs/sq4_modern'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--eval-batch-size',type=int,default=1024)
    ap.add_argument('--max-epochs',type=int,default=15)
    ap.add_argument('--patience',type=int,default=2)
    ap.add_argument('--reuse-checkpoint',action='store_true',help='Skip training only when an existing modern checkpoint is present')
    args=ap.parse_args(argv)
    src=args.repo/'src'
    if not (src/'visual_alpha'/'models.py').exists():raise FileNotFoundError(f'Original repo not found at {src}')
    sys.path.insert(0,str(src.resolve()))
    from visual_alpha.dataset import build_train_val_test_splits
    from visual_alpha.models import CNN20D
    from visual_alpha.trainer import train_model,predict_dataset
    from visual_alpha.evaluate import evaluate_predictions
    from visual_alpha.utils import get_device,seed_everything
    data_dir=resolve_data(args.data_dir)
    checks=verify_years(data_dir)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    (out/'data_preflight.json').write_text(json.dumps({'raw_dir':str(data_dir),'years':checks,'seed':args.seed,
               'test_period':'2013-2019','train_val_period':'2006-2012'},indent=2))
    print('Device:',get_device(),' | Starting full 2006–2012 modern model experiment',flush=True)
    splits=build_train_val_test_splits(data_dir=data_dir,
          train_val_years=range(2006,2013),test_years=range(2013,2020),
          split_seed=args.seed,val_ratio=.30)
    ns=[len(splits.train),len(splits.val),len(splits.test)]
    print('train, val, test:',ns,flush=True)
    if sum(ns[:2])<400000 or ns[2]<400000:raise RuntimeError('Data set too small; refusing sampled proxy training')
    device=get_device()
    ckpath=out/f'sq4_modern_seed{args.seed}.pt'
    seed_everything(args.seed) # seed BEFORE initializing weights
    model=CNN20D()
    if args.reuse_checkpoint and ckpath.exists():
        checkpoint=torch.load(ckpath,map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        model.to(device)
        print('Reusing previously trained modern checkpoint',ckpath,flush=True)
        best_epoch=checkpoint.get('best_epoch')
    else:
        res=train_model(model,splits.train,splits.val,seed=args.seed,
             lr=1e-5,batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
             max_epochs=args.max_epochs,patience=args.patience,device=device,
             checkpoint_path=ckpath,verbose=True)
        res.history.to_csv(out/'sq4_modern_training_history.csv',index=False)
        best_epoch=res.best_epoch
    modern=predict_dataset(model,splits.test,batch_size=args.eval_batch_size,device=device)
    modern.to_parquet(out/'sq4_modern_predictions_2013_2019.parquet',index=False)
    old=pd.read_parquet(args.baseline)
    old=old[old.year.between(2013,2019)].copy()
    # Align by immutable source coordinates rather than assuming any sort order.
    keys=['year','row_idx']
    if old.duplicated(keys).any() or modern.duplicated(keys).any():raise ValueError('Duplicate historical/modern keys')
    if len(old)!=len(modern):raise ValueError(f'Non-identical OOS test counts old={len(old)} modern={len(modern)}')
    older=old.sort_values(keys,kind='stable').reset_index(drop=True)
    newer=modern.sort_values(keys,kind='stable').reset_index(drop=True)
    if not np.array_equal(older[keys].to_numpy(),newer[keys].to_numpy()):raise ValueError('Test row IDs differ')
    if not np.array_equal(older.y_true.to_numpy(),newer.y_true.to_numpy()):raise ValueError('Label mismatch, cannot compare')
    rows=[];years=[]
    for name,d in [('Model_1990s',older),('Model_Modern',newer)]:
        rows.append({'model':name,'training_years':'1993-1999' if name=='Model_1990s' else '2006-2012',
            'test_years':'2013-2019',**evaluate_predictions(d)})
        for year,g in d.groupby('year'):
            mt=evaluate_predictions(g)
            years.append({'model':name,'year':int(year),'n':len(g),'auc':mt['auc'],'accuracy_pct':mt['accuracy_pct'],
                'ce_loss':mt['ce_loss']})
    pd.DataFrame(rows).to_csv(out/'tab3_concept_drift.csv',index=False)
    annual=pd.DataFrame(years);annual.to_csv(out/'sq4_modern_annual_comparison.csv',index=False)
    match=older[['year','row_idx','Date','y_true','prob_up']].merge(newer[['year','row_idx','prob_up']],on=keys,suffixes=('_old','_new'),validate='one_to_one')
    monthly,bs=month_paired_bootstrap(match)
    monthly.to_csv(out/'sq4_modern_paired_monthly_metrics.csv',index=False)
    bs.to_csv(out/'sq4_modern_paired_bootstrap.csv',index=False)
    generate_annual_figure(annual,out)
    (out/'modern_verification.json').write_text(json.dumps({
         'full_original_training':True,'n_training':ns[0],'n_validation':ns[1],'n_oos':ns[2],
         'seed':args.seed,'init_seeded_before_model':True,'best_epoch':best_epoch,
         'checkpoint':str(ckpath),'same_test_keys':True,'same_y_true':True,
         'device':str(device)},indent=2))
    bundle=out/'SQ4_Modern_Actual_Results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in out.iterdir():
            if path.is_file() and path!=bundle:
                z.write(path,arcname=path.name)
    print('COMPLETE MODERN:',bundle.resolve(),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Paired monthly bootstrap:\n',bs.to_string(index=False),flush=True)

if __name__=='__main__':main()
