"""Regenerate SQ4 Figure 4 + training-diagnostics figure from the *included* results.

No 8.6 GB raw archive, torch, or parquet decoder needed. Original Grad-CAM
arrays and selected raw images are bundled. The script does NOT retrain a model.

  python 02_Code/scripts/rebuild_figures.py
"""
import io,zipfile,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BASE);args=p.parse_args()
 root=args.root.resolve();tables=root/'outputs/tables'; figs=root/'outputs/figures';figs.mkdir(exist_ok=True,parents=True)
 a=pd.read_csv(tables/'full_year_metrics.csv')
 b=pd.read_csv(tables/'sq4_modern_annual_comparison.csv')
 hist=pd.read_csv(tables/'sq4_modern_training_history.csv')
 feat=pd.read_csv(tables/'sq4_top2000_individual_gradcam.csv')
 roi=pd.read_csv(tables/'sq4_top2000_global_roi.csv')
 cams=np.load(root/'outputs/gradcam/sq4_top2000_cam_arrays.npz')['cams'].astype('float32')
 with zipfile.ZipFile(root/'outputs/gradcam/SQ4_Top2000_Images.zip') as z:
  images=np.load(io.BytesIO(z.read('images.npy')),allow_pickle=False)
  mf=pd.read_csv(io.BytesIO(z.read('manifest.csv')))
 assert cams.shape==images.shape==(2000,64,60) and len(feat)==len(mf)==2000
 assert np.array_equal(feat.image_index.to_numpy(),np.arange(2000))
 assert np.array_equal(feat.group.to_numpy(),mf.group.to_numpy())
 fig=plt.figure(figsize=(13.6,8.6),facecolor='white'); gs=GridSpec(2,4,figure=fig,height_ratios=[.87,1.13],hspace=.35,wspace=.38,left=.06,right=.92,bottom=.10,top=.93)
 ax=fig.add_subplot(gs[0,:2])
 for model,color,label in [('Seed42','#305f79','1990s CNN'),('Ensemble5','#a6723a','1990s 5-seed ensemble')]:
  t=a[a.model.eq(model)].sort_values('year');ax.plot(t.year,t.auc,'-o',lw=2,ms=3.4,color=color,label=label)
 t=b[b.model.eq('Model_Modern')].sort_values('year');ax.plot(t.year,t.auc,'-s',lw=1.7,ms=3.9,color='#9c528b',label='Modern CNN (2013–2019)')
 ax.axhline(.5,color='gray',ls='--',lw=.8);ax.axvspan(2012.5,2019.25,alpha=.11,color='#78929c')
 ax.set(xlim=(1999.7,2019.4),ylabel='Out-of-sample AUC',xlabel='Test year',title='(a) Predictive discrimination, 2000–2019')
 ax.set_xticks(range(2000,2020,2));ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=7.5,frameon=False)
 for i,g in enumerate(['Top1000_UP','Top1000_DOWN']):
  ax2=fig.add_subplot(gs[0,2+i]);t=feat[(feat.group==g)&feat.correct.eq(1)&feat.valid_cam.eq(1)].sort_values('confidence',ascending=False)
  r=t.iloc[0];idx=int(r.image_index);cm=cams[idx]
  ax2.imshow(images[idx],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
  ax2.imshow(cm,cmap='magma',alpha=.8*np.clip(cm/(cm.max()+1e-9),0,1),interpolation='nearest')
  ax2.axhline(50.5,color='#66f4dd',ls='--',lw=.8);ax2.set_xticks([]);ax2.set_yticks([])
  ax2.set_title('(b) '+('Bullish' if i==0 else 'Bearish')+' case\n'+str(r.Date)[:10]+f' · p(UP)={r.prob_up:.3f}',fontsize=10)
 maps={g:cams[feat.group.to_numpy()==g].mean(axis=0) for g in ['Top1000_UP','Top1000_DOWN']};vmax=max(float(v.max()) for v in maps.values())
 for i,g in enumerate(['Top1000_UP','Top1000_DOWN']):
  ax3=fig.add_subplot(gs[1,i*2:(i+1)*2]);im=ax3.imshow(maps[g],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest',aspect='equal')
  ax3.axhline(50.5,color='#58f0e2',ls='--',lw=1);ax3.axhline(51.5,color='#58f0e2',ls=':',lw=.6);ax3.axvline(29.5,color='white',ls=':',lw=.75)
  ax3.set_xticks([0,15,30,45,59],['1','6','11','16','20']);ax3.set_xlabel('Trading day (oldest to most recent)',fontsize=9);ax3.set_ylabel('Image row',fontsize=9)
  r=roi[roi.group==g].iloc[0];ax3.set_title(('(c) Top-1000 UP' if i==0 else '(d) Top-1000 DOWN')+f' · volume share {r.volume_mass*100:.2f}%',fontsize=11)
 bar_ax=fig.add_axes([.935,.14,.013,.39]);cb=fig.colorbar(im,cax=bar_ax);cb.set_label('Mean normalized CAM mass / pixel',fontsize=8);cb.ax.tick_params(labelsize=7)
 fig.suptitle('SQ4  |  Temporal transfer and Grad-CAM attribution',fontsize=16,fontweight='bold',y=.985)
 fig.text(.5,.035,'Observed predictions and original CNN. CAM is a post-hoc attribution, not causal evidence.',ha='center',fontsize=9,color='#444444')
 for ext in ['png','pdf']:fig.savefig(figs/f'fig4_sq4_drift_gradcam.{ext}',dpi=210,facecolor='white',bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
 for ax,train,val,yl,title in [(axes[0],'train_loss','val_loss','Cross-entropy','Training and validation loss'),(axes[1],'train_acc_pct','val_acc_pct','Accuracy (%)','Training and validation accuracy')]:
  ax.plot(hist.epoch,hist[train],'-o',color='#305f79',label='Train');ax.plot(hist.epoch,hist[val],'-o',color='#a6723a',label='Validation')
  ax.axvline(11,color='gray',ls='--',lw=1,label='Best epoch 11');ax.set(xlabel='Epoch',ylabel=yl,title=title);ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
 fig.savefig(figs/'modern_training_diagnostics.png',dpi=210,bbox_inches='tight');plt.close(fig)
 print('Rebuilt: Figure 4 and training diagnostics in',figs)
if __name__=='__main__':main()
