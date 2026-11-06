"""Isolated five-vs-three-hidden-layer DNN comparison at l=250."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiment_common import ROOT, load, save, sha, utc, identity, alive, read_csv, write_csv

OUT = ROOT/'ablations/dnn_depth_l250_seed2034_20261029'
BASELINE = ROOT/'paper_figures/fig7_seed2034/l250/dnn'


def verify():
    import run_with_convergence_seed2034 as controller
    controller.check_sources(load(ROOT/'deployment_protocol.json'))
    protocol=load(ROOT/'dnn_depth_l250_protocol.json')
    for name,digest in protocol['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    return protocol


def status(phase, **fields):
    record=dict(phase=phase,pid=os.getpid(),at_utc=utc(),**fields)
    save(OUT/'status.json',record)
    print(__import__('json').dumps(record),flush=True)


def run():
    import numpy as np
    import torch
    import experiment_workers as common
    protocol=verify();common.configure();base=common.base
    assert torch.cuda.is_available()
    reference=load(BASELINE/'result.json')
    assert reference['seed']==2034 and reference['length']==250 and reference['epochs']==60
    assert sha(BASELINE/'curve.csv')==reference['curve_sha256']
    assert sha(BASELINE/'training_config.json')==reference['training_config_sha256']
    assert sha(reference['checkpoint'])==reference['checkpoint_sha256']
    frozen=load(ROOT/'search/dnn/frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']==reference['optimizer_sha256']
    start=int(frozen['starts']['250']);assert start==reference['start']==1
    old_cfg=load(BASELINE/'training_config.json');cfg=copy.deepcopy(old_cfg)
    assert old_cfg['models']['dnn'][0]['num_hidden_layers']==3
    assert old_cfg['models']['dnn'][0]['hidden_dim']==192
    cfg['models']['dnn']=[dict(id='depth5_width161_matched_l250',hidden_dim=161,num_hidden_layers=5,dropout=.05)]
    cfg['paths']['checkpoint_cache_dir']=str(OUT/'deep/checkpoints')
    assert base.batch_size_for(cfg,250)==256 and cfg['training']['lr']==.001
    assert cfg['training']['seed']==2034 and cfg['training']['loss']=='mse_physical_um2'
    counts={}
    for label,settings in [('three_layers',old_cfg),('five_layers',cfg)]:
        network=base.build_localizer(settings,'dnn',250)
        counts[label]=sum(p.numel() for p in network.parameters() if p.requires_grad)
        del network
    assert counts=={'three_layers':269842,'five_layers':269423},counts
    assert abs(counts['five_layers']/counts['three_layers']-1)<.002
    save(OUT/'preflight.json',dict(at_utc=utc(),parameters=counts,baseline=reference,
        optimizer_retrained=False,start=start,gpu=torch.cuda.get_device_name(0),protocol_sha256=sha(ROOT/'dnn_depth_l250_protocol.json')))
    save(OUT/'deep/training_config.json',cfg)
    save(OUT/'baseline/result.json',reference)
    (OUT/'baseline/curve.csv').write_bytes((BASELINE/'curve.csv').read_bytes())
    status('loading_data',parameters=counts,batch_size=256,lr=.001,seed=2034,start=start)
    tr,va=common.full_data(cfg);scaler=base.TargetScaler(tr)
    td,vd=[base.make_dataset(data,scaler,250,start,cfg) for data in [tr,va]]
    curve=OUT/'deep/curve.csv';rows=[]
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,model='dnn',hidden_layers=5,hidden_width=161,length=250,
            start=start,seed=2034,batch_size=256,train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);status('training',**row)
    base.TRAINING_OBSERVER=observe
    state,best_loss,best_epoch,elapsed,cached=base.train_or_load(cfg,'depth_ablation60','dnn',250,start,
        2034,td,vd,torch.device('cuda'),60,61)
    if cached:rows=read_csv(curve)
    assert [int(r['epoch']) for r in rows]==list(range(1,61))
    checkpoint=base.checkpoint_path(cfg,'depth_ablation60','dnn',250,start,2034,60,61)
    result=dict(length=250,start=start,seed=2034,batch_size=256,lr=.001,epochs=60,
        early_stopping=False,hidden_layers=5,hidden_width=161,parameters=counts['five_layers'],
        best_validation_mse_um2=best_loss,best_validation_epoch=best_epoch,
        training_seconds=elapsed,reused_checkpoint=cached,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),curve_sha256=sha(curve))
    save(OUT/'deep/result.json',result)
    status('evaluating')
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:
        test={k:z[k].copy() for k in ['X','x','y']}
    comparison=[]
    old_state=torch.load(reference['checkpoint'],map_location='cpu',weights_only=False)['state']
    for label,settings,weights,info in [('three_layers',old_cfg,old_state,reference),('five_layers',cfg,state,result)]:
        metrics,pred,truth,errors=common.infer(settings,weights,'dnn',250,start,test,scaler)
        np.savez_compressed(OUT/(label+'_predictions.npz'),predictions=pred,truth=truth,errors_um=errors)
        history=read_csv(OUT/('baseline' if label=='three_layers' else 'deep')/'curve.csv')
        comparison.append(dict(architecture=label,parameters=counts[label],length=250,start=start,
            seed=2034,batch_size=256,lr=.001,epochs=60,n_test=len(errors),
            final_train_mse_um2=float(history[-1]['train_mse_um2']),
            final_validation_mse_um2=float(history[-1]['val_mse_um2']),
            best_validation_mse_um2=info['best_validation_mse_um2'],
            best_validation_epoch=info['best_validation_epoch'],
            test_sha256=sha(ROOT/'data/independent_test.npz'),checkpoint_sha256=info['checkpoint_sha256'],**metrics))
    write_csv(OUT/'comparison.csv',comparison)
    render()
    status('complete',comparison=comparison,optimizer_retrained=False)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix'})
    histories={label:read_csv(OUT/folder/'curve.csv') for label,folder in [('3 hidden layers (192 each)','baseline'),('5 hidden layers (161 each)','deep')]}
    for field,name,ylabel in [('train_mse_um2','training_loss',r'Training loss ($\mu$m$^2$)'),
                              ('val_mse_um2','validation_loss',r'Validation loss ($\mu$m$^2$)')]:
        fig,ax=plt.subplots(figsize=(4,3));fig.subplots_adjust(left=.18,right=.97,bottom=.2,top=.96)
        for (label,history),color in zip(histories.items(),['#1f77b4','#d62728']):
            ax.plot([int(r['epoch']) for r in history],[float(r[field]) for r in history],label=label,color=color,linewidth=1.25)
        ax.set(xlim=(0,60),ylim=(0,None),xlabel='Epoch',ylabel=ylabel,xticks=range(0,61,10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.grid(linestyle=':',linewidth=.5,color='#b8b8b8');ax.legend(loc='upper right',fontsize=8)
        fig.savefig(OUT/f'{name}_l250.png',dpi=600,bbox_inches='tight');plt.close(fig)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'launch.json'
    if target.exists() and alive(load(target)['identity']):
        print(__import__('json').dumps(load(target)));return
    assert not (OUT/'status.json').exists(),'Inspect existing run before relaunch'
    active=load(ROOT/'active.json');assert not any(alive(v['identity']) for v in active.values())
    with (OUT/'worker.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    rec=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),
        protocol=str(ROOT/'dnn_depth_l250_protocol.json'))
    save(target,rec);save(ROOT/'CURRENT_DNN_DEPTH_ABLATION.json',rec)
    print(__import__('json').dumps(rec),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except Exception:
            status('failed',error=traceback.format_exc());raise
