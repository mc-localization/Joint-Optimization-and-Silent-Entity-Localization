"""Independent l=50 convergence measurements; original l=100 curves are preserved."""
import copy,json,math,sys
from pathlib import Path
import numpy as np
from experiment_common import ROOT,load,save,sha,report,mark_done,write_csv,read_csv

MODELS=['cnn','dnn','gru','transformer']
OUT=ROOT/'paper_figures/fig7_l50'

def cfg_for(model):
    adapter=__import__('gru_workers' if model=='gru' else 'transformer_workers' if model=='transformer' else 'experiment_workers')
    cfg=copy.deepcopy(adapter.cfg_for(model))
    cfg['training'].update(seed=2025,batch_size=256,long_window_batch_size=256)
    cfg['paths']['checkpoint_cache_dir']=str(OUT/model/'checkpoints')
    return cfg,adapter

def convergence(job):
    import torch
    import experiment_workers as common
    common.configure();base=common.base;model=job['model'];length=50
    cfg,adapter=cfg_for(model)
    frozen=load(ROOT/'search'/model/'frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
    assert base.batch_size_for(cfg,length)==256
    start=int(frozen['starts'][str(length)])
    folder=OUT/model;folder.mkdir(parents=True,exist_ok=True)
    save(folder/'training_config.json',cfg)
    report(job['key'],'loading_data',model=model,length=50,start=start,seed=2025,batch_size=256)
    tr,va=adapter.full_data(cfg);scaler=base.TargetScaler(tr)
    assert len(tr['x'])==410640 and len(va['x'])==25500
    td,vd=[base.make_dataset(d,scaler,length,start,cfg) for d in [tr,va]]
    curve=folder/'curve.csv';rows=[]
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,model=model,length=length,start=start,seed=2025,batch_size=256,
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);report(job['key'],'training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,'convergence_l50_60',model,length,start,
        2025,td,vd,torch.device('cuda'),60,61)
    if cached:rows=read_csv(curve)
    assert [int(r['epoch']) for r in rows]==list(range(1,61))
    checkpoint=base.checkpoint_path(cfg,'convergence_l50_60',model,length,start,2025,60,61)
    record=dict(model=model,length=50,start=start,seed=2025,batch_size=256,epochs=60,early_stopping=False,
        loss='mse_physical_um2',deterministic=False,optimizer_sha256=frozen['optimizer_sha256'],
        training_config_sha256=sha(folder/'training_config.json'),curve_sha256=sha(curve),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),best_validation_epoch=epoch,
        best_validation_mse_um2=loss,training_seconds=elapsed)
    save(folder/'result.json',record);mark_done(job['key'],**record)

def render(job):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'mathtext.fontset':'stix'})
    records={m:read_csv(OUT/m/'curve.csv') for m in MODELS}
    for model,rows in records.items():
        assert [int(r['epoch']) for r in rows]==list(range(1,61))
        assert all(int(r['length'])==50 and int(r['batch_size'])==256 and int(r['seed'])==2025 for r in rows)
    outputs={};values={}
    for field,name,label,smooth,minimum in [
        ('train_mse_um2','fig7_training_loss_l50',r'Training loss ($\mu$m$^2$)',False,25),
        ('val_mse_um2','fig8_validation_loss_l50',r'Validation loss ($\mu$m$^2$)',True,5)]:
        fig,ax=plt.subplots(figsize=(4,3));fig.subplots_adjust(left=.18,right=.97,bottom=.20,top=.96)
        values[field]={}
        for i,model in enumerate(MODELS):
            ys=[float(r[field]) for r in records[model]]
            if smooth:
                for k in range(1,len(ys)):ys[k]=.2*ys[k]+.8*ys[k-1]
            assert all(math.isfinite(v) and v>=0 for v in ys)
            values[field][model]=ys
            ax.plot(range(1,61),ys,color=['#d62728','#1f77b4','#d8a700','#2ca02c'][i],
                label=['CNN','DNN','GRU','Transformer'][i],linewidth=1.25)
        upper=max(minimum,5*math.ceil(max(max(ys) for ys in values[field].values())/5))
        ax.set(xlim=(0,60),ylim=(0,upper),xticks=range(0,61,10),xlabel='Epoch',ylabel=label)
        ax.set_yticks(np.linspace(0,upper,6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_axisbelow(True);ax.grid(which='major',linestyle=':',linewidth=.5,color='#b8b8b8')
        ax.tick_params(which='both',direction='in',top=True,right=True)
        for spine in ax.spines.values():spine.set_linewidth(.8)
        lg=ax.legend(loc='upper right',ncol=1,framealpha=1,edgecolor='black');lg.get_frame().set_linewidth(.6)
        fig.canvas.draw();path=OUT/(name+'.png');fig.savefig(path,dpi=600,bbox_inches='tight',pad_inches=.04);plt.close(fig)
        outputs[name]=dict(path=str(path),sha256=sha(path))
    save(OUT/'manifest.json',dict(length=50,models=MODELS,epochs=60,batch_size=256,seed=2025,
        train_smoothing=False,validation_ema_alpha=.2,outputs=outputs,
        curve_sha256={m:sha(OUT/m/'curve.csv') for m in MODELS}))
    mark_done(job['key'],outputs=outputs)

def run(job):
    p=load(ROOT/'convergence_l50_protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if job['kind']=='convergence':convergence(job)
    elif job['kind']=='paper_figure':render(job)
    else:raise ValueError(job)
