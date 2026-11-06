"""Ten hidden layers, exactly matched to CNN parameters at l=50; test l=250."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from experiment_common import ROOT, load, save, sha, utc, identity, alive, read_csv, write_csv

OUT=ROOT/'ablations/dnn10_matched_cnn50_l250_seed2034_20261029'
OLD=ROOT/'ablations/dnn_depth_l250_seed2034_20261029'
CNN=ROOT/'paper_figures/fig7_seed2034/l250/cnn'
DNN=ROOT/'paper_figures/fig7_seed2034/l250/dnn'
WIDTHS=[99,99,98,97,97,97,97,96,96,96]
CONFIG_ID='ten_layers_exact_cnn50_budget'


def verify():
    import dnn_depth_l250
    dnn_depth_l250.verify()
    p=load(ROOT/'dnn10_matched_cnn50_protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert p['hidden_dims']==WIDTHS
    return p


def status(phase,**kw):
    rec=dict(phase=phase,pid=os.getpid(),at_utc=utc(),**kw)
    save(OUT/'status.json',rec);print(json.dumps(rec),flush=True)


def install_model(base):
    import torch.nn as nn
    from retest_models_4rx import _preprocess, _init_xavier
    native=base.build_localizer
    class TenLayerDNN(nn.Module):
        def __init__(self,input_dim,widths,dropout):
            super().__init__();self.input_bn=nn.BatchNorm1d(input_dim)
            layers=[];previous=input_dim
            for width in widths:
                layers.extend([nn.Linear(previous,width),nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(dropout)])
                previous=width
            self.backbone=nn.Sequential(*layers);self.head=nn.Linear(previous,2)
            _init_xavier(self)
        def forward(self,x):return self.head(self.backbone(_preprocess(x,self.input_bn)))
    def build(cfg,model,length):
        item=cfg['models'][model][0]
        if model=='dnn' and item['id']==CONFIG_ID:
            assert item['hidden_dims']==WIDTHS
            return TenLayerDNN(cfg['data']['n_rx']*length,item['hidden_dims'],item['dropout'])
        return native(cfg,model,length)
    base.build_localizer=build


def run():
    import numpy as np
    import torch
    import experiment_workers as common
    verify();common.configure();base=common.base;install_model(base)
    assert torch.cuda.is_available()
    reference=load(DNN/'result.json');cnn_reference=load(CNN/'result.json')
    for folder,rec in [(DNN,reference),(CNN,cnn_reference)]:
        assert rec['seed']==2034 and rec['length']==250 and rec['start']==1 and rec['epochs']==60
        assert sha(folder/'training_config.json')==rec['training_config_sha256']
        assert sha(folder/'curve.csv')==rec['curve_sha256']
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    frozen=load(ROOT/'search/dnn/frozen_optimizer.json')
    assert int(frozen['starts']['250'])==1
    assert sha(frozen['optimizer_path'])==reference['optimizer_sha256']==frozen['optimizer_sha256']
    cfg=copy.deepcopy(load(DNN/'training_config.json'))
    cfg['models']['dnn']=[dict(id=CONFIG_ID,hidden_dims=WIDTHS,dropout=.05)]
    cfg['paths']['checkpoint_cache_dir']=str(OUT/'deep/checkpoints')
    assert cfg['training']['seed']==2034 and cfg['training']['lr']==.001
    assert base.batch_size_for(cfg,250)==256
    assert cfg['training']['loss']=='mse_physical_um2' and cfg['training']['deterministic'] is False
    cnn_cfg=load(CNN/'training_config.json');counts={}
    for model,settings in [('dnn',cfg),('cnn',cnn_cfg)]:
        for length in [50,250]:
            net=base.build_localizer(settings,model,length)
            counts[f'{model}_l{length}']=sum(w.numel() for w in net.parameters() if w.requires_grad)
            del net
    assert counts==dict(dnn_l50=108290,dnn_l250=189090,cnn_l50=108290,cnn_l250=109890),counts
    # Verify the additive model supports finite gradients; training resets its seed afterwards.
    net=base.build_localizer(cfg,'dnn',250).cuda()
    prediction=net(torch.rand(8,1000,device='cuda'))
    assert prediction.shape==(8,2) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(torch.isfinite(w.grad).all() for w in net.parameters() if w.grad is not None)
    del net,prediction;torch.cuda.empty_cache()
    previous=read_csv(OLD/'comparison_with_cnn.csv')
    assert {r['architecture'] for r in previous}=={'three_layers','five_layers','cnn'}
    assert all(int(r['seed'])==2034 and int(r['length'])==250 and int(r['start'])==1 for r in previous)
    save(OUT/'deep/training_config.json',cfg)
    for label,source in [('cnn',CNN/'curve.csv'),('three_layers',DNN/'curve.csv'),('five_layers',OLD/'deep/curve.csv')]:
        p=OUT/'references'/f'{label}_curve.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes())
    save(OUT/'preflight.json',dict(at_utc=utc(),parameters=counts,hidden_dims=WIDTHS,
        reference_length=50,test_length=250,gradient_check=True,
        source_comparison_sha256=sha(OLD/'comparison_with_cnn.csv'),
        optimizer_sha256=frozen['optimizer_sha256'],optimizer_retrained=False,
        protocol_sha256=sha(ROOT/'dnn10_matched_cnn50_protocol.json')))
    status('loading_data',hidden_dims=WIDTHS,parameters=counts,batch_size=256,lr=.001,seed=2034,length=250,start=1)
    tr,va=common.full_data(cfg);scaler=base.TargetScaler(tr)
    td,vd=[base.make_dataset(d,scaler,250,1,cfg) for d in [tr,va]]
    curve=OUT/'deep/curve.csv';rows=[]
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,length=250,start=1,seed=2034,batch_size=256,
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);status('training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,seconds,cached=base.train_or_load(cfg,'dnn10_cnn50_budget60','dnn',250,1,
        2034,td,vd,torch.device('cuda'),60,61)
    if cached:rows=read_csv(curve)
    assert [int(r['epoch']) for r in rows]==list(range(1,61))
    checkpoint=base.checkpoint_path(cfg,'dnn10_cnn50_budget60','dnn',250,1,2034,60,61)
    result=dict(length=250,start=1,seed=2034,batch_size=256,lr=.001,epochs=60,early_stopping=False,
        hidden_dims=WIDTHS,parameters=counts['dnn_l250'],best_validation_mse_um2=loss,
        best_validation_epoch=epoch,training_seconds=seconds,reused_checkpoint=cached,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),curve_sha256=sha(curve))
    save(OUT/'deep/result.json',result);status('evaluating')
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    assert all(r['test_sha256']==sha(ROOT/'data/independent_test.npz') for r in previous)
    metrics,pred,truth,errors=common.infer(cfg,state,'dnn',250,1,test,scaler)
    np.savez_compressed(OUT/'ten_layers_predictions.npz',predictions=pred,truth=truth,errors_um=errors)
    row=dict(architecture='ten_layers_cnn50_budget',parameters=counts['dnn_l250'],length=250,start=1,
        seed=2034,batch_size=256,lr=.001,epochs=60,n_test=len(errors),
        final_train_mse_um2=float(rows[-1]['train_mse_um2']),final_validation_mse_um2=float(rows[-1]['val_mse_um2']),
        best_validation_mse_um2=loss,best_validation_epoch=epoch,test_sha256=sha(ROOT/'data/independent_test.npz'),
        checkpoint_sha256=sha(checkpoint),**metrics)
    write_csv(OUT/'comparison.csv',previous+[row]);render()
    status('complete',result=row,optimizer_retrained=False)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix'})
    histories={'CNN':read_csv(OUT/'references/cnn_curve.csv'),
        'DNN (3 hidden layers)':read_csv(OUT/'references/three_layers_curve.csv'),
        'DNN (5 hidden layers)':read_csv(OUT/'references/five_layers_curve.csv'),
        'DNN (10 hidden layers)':read_csv(OUT/'deep/curve.csv')}
    for field,name,label in [('train_mse_um2','training_loss',r'Training loss ($\mu$m$^2$)'),
                             ('val_mse_um2','validation_loss',r'Validation loss ($\mu$m$^2$)')]:
        fig,ax=plt.subplots(figsize=(4.5,3.3));fig.subplots_adjust(left=.18,right=.97,bottom=.18,top=.96)
        for (title,rows),color in zip(histories.items(),['#d62728','#1f77b4','#2ca02c','#9467bd']):
            ax.plot(range(1,61),[float(r[field]) for r in rows],label=title,color=color,lw=1.25)
        ax.set(xlim=(0,60),ylim=(0,None),xlabel='Epoch',ylabel=label,xticks=range(0,61,10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.grid(linestyle=':',linewidth=.5,color='#b8b8b8');ax.legend(loc='upper right',fontsize=8)
        fig.savefig(OUT/f'{name}_l250.png',dpi=600,bbox_inches='tight');plt.close(fig)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'launch.json'
    if target.exists() and alive(load(target)['identity']):print(json.dumps(load(target)));return
    assert not (OUT/'status.json').exists(),'Inspect existing run before relaunch'
    assert not any(alive(v['identity']) for v in load(ROOT/'active.json').values())
    prior=ROOT/'CURRENT_DNN_DEPTH_ABLATION.json'
    if prior.exists():assert not alive(load(prior)['identity'])
    with (OUT/'worker.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    rec=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),
        protocol=str(ROOT/'dnn10_matched_cnn50_protocol.json'))
    save(target,rec);save(prior,rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except Exception:status('failed',error=traceback.format_exc());raise
