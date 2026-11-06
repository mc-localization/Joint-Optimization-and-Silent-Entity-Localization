"""Independent fits, policy search adapters, validation and frozen test evaluation."""
from pathlib import Path
import copy,json,sys,time
import numpy as np
import torch
from experiment_common import ROOT,load,save,sha,report,mark_done,write_csv,read_csv
DATA_ROOT=ROOT.parent
sys.path.insert(0,str(DATA_ROOT/'runtime'))
import run_experiment_4rx as base

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def cfg_for(model,stage='final'):
    assert model in ['cnn','transformer']
    assert stage=='final','No start-optimizer or candidate-search training is authorized in this rerun.'
    cfg=copy.deepcopy(load(ROOT/'protocol.json')['cases'][model][stage+'_cfg'])
    return cfg

def micro_data(cfg):
    folder=DATA_ROOT/'micro_dataset';manifest=load(folder/'sampling_manifest.json')
    assert manifest['generation_signature']==cfg['data']['dataset_protocol_sha256']
    result=[]
    for name,count in [('micro_train.npz',cfg['data']['search_train_samples']),('micro_valid.npz',cfg['data']['search_valid_samples'])]:
        assert sha(folder/name)==manifest['artifacts_sha256'][name]
        with np.load(folder/name,allow_pickle=False) as z:result.append({k:z[k].copy() for k in ['X','x','y']})
        assert len(result[-1]['x'])==count
    return result

def full_data(cfg):
    ready=load(DATA_ROOT/'preflight/readiness.json')
    assert ready['generation_signature']==cfg['data']['dataset_protocol_sha256']
    train,valid,_,_,_=base.load_splits(cfg,final=True,include_test=False)
    assert len(train['x'])==cfg['data']['final_train_samples']
    assert len(valid['x'])==cfg['data']['final_valid_samples']
    return train,valid

def infer(cfg,state,model,length,start,raw,scaler):
    ds=base.make_dataset(raw,scaler,length,start,cfg)
    net=base.build_localizer(cfg,model,length).cuda();net.load_state_dict(state);net.eval()
    predictions=[]
    with torch.inference_mode():
        for xb,_ in base.make_loader(ds,cfg,torch.device('cuda'),length,shuffle=False):
            with torch.amp.autocast(device_type='cuda',enabled=cfg['training']['amp']):
                predictions.append(base.forward_model(net,xb.cuda(non_blocking=True)).float().cpu().numpy())
    pred=scaler.physical(np.concatenate(predictions));truth=np.column_stack((raw['x'],raw['y'])).astype(np.float64)
    errors=np.linalg.norm(pred-truth,axis=1)
    assert np.isfinite(errors).all()
    return dict(mean_error_um=float(errors.mean()),median_error_um=float(np.median(errors)),
        p90_error_um=float(np.quantile(errors,.9)),success_rate=float((errors<.4).mean()),
        accuracy025=float((errors<.25).mean())),pred,truth,errors

def train(job):
    stage='search' if job['kind']=='search_fit' else 'final'
    cfg=cfg_for(job['model'],stage);model=job['model'];length=int(job['length']);start=int(job['start'])
    train_data,valid_data=micro_data(cfg) if stage=='search' else full_data(cfg)
    scaler=base.TargetScaler(train_data)
    td,vd=[base.make_dataset(d,scaler,length,start,cfg) for d in [train_data,valid_data]]
    checkpoint=base.checkpoint_path(cfg,stage,model,length,start,cfg['training']['seed'],60,10)
    curve_path=checkpoint.with_suffix('.curve.csv');rows=[]
    def observer(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,model=model,length=length,start=start,seed=cfg['training']['seed'],
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve_path,rows)
        report(job['key'],'training',stage=stage,**row)
    base.TRAINING_OBSERVER=observer
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,stage,model,length,start,
        cfg['training']['seed'],td,vd,torch.device('cuda'),60,10)
    metrics,pred,truth,errors=infer(cfg,state,model,length,start,valid_data,scaler)
    row=dict(model=model,length=length,start=start,seed=cfg['training']['seed'],
        best_epoch=epoch,val_mse_um2=loss,val_mean_error_um=metrics['mean_error_um'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_seconds=elapsed,reused_checkpoint=bool(cached),
        generation_signature=cfg['data']['dataset_protocol_sha256'],
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**metrics)
    target=ROOT/'search'/model/'score_cache'/f'l{length}_s{start}.json' if stage=='search' else ROOT/'fits'/f'{model}_l{length}_s{start}.json'
    save(target,row);mark_done(job['key'],result=str(target),**row)

def evaluate(job):
    model=job.get('model','cnn');length=int(job['length']);role=job['role']
    cfg=cfg_for(model)
    frozen=load(ROOT/'search'/model/'frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
    if role=='optimized':start=int(frozen['starts'][str(length)])
    elif role=='fixed':start=1
    elif role=='coarse':start=int(load(ROOT/'coarse'/f'l{length}'/'selected_start.json')['start'])
    else:raise ValueError(role)
    fit=load(ROOT/'fits'/f'{model}_l{length}_s{start}.json')
    checkpoint=Path(fit['checkpoint_source']);assert sha(checkpoint)==fit['checkpoint_sha256']
    # Fit the coordinate scaler on training labels only; do not load full X again.
    labels={k:[] for k in ['x','y']}
    for p in sorted((DATA_ROOT/'data/data_train').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for k in labels:labels[k].append(z[k].astype(np.float32))
    labels={k:np.concatenate(v) for k,v in labels.items()};scaler=base.TargetScaler(labels)
    with np.load(DATA_ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state']
    metrics,pred,truth,errors=infer(cfg,state,model,length,start,test,scaler)
    folder=ROOT/'results'/f'{model}_{role}'/f'l{length}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.npz';np.savez_compressed(path,predictions=pred,truth=truth,errors_um=errors)
    row=dict(model=model,condition=role,length=length,start=start,n_test=len(errors),seed=cfg['training']['seed'],batch_size=256,deterministic=False,optimizer_reused=True,optimizer_training_seed=frozen['seed'],
        training_loss='mse_physical_um2',best_epoch=fit['best_epoch'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=fit['checkpoint_sha256'],
        test_sha256=sha(DATA_ROOT/'data/independent_test.npz'),predictions_sha256=sha(path),
        optimizer_sha256=frozen['optimizer_sha256'] if role=='optimized' else None,**metrics)
    write_csv(folder/'final_test.csv',[row]);mark_done(job['key'],result=row)


def run(job):
    configure();report(job['key'],'started',kind=job['kind'])
    if job['kind']=='formal_fit':train(job)
    elif job['kind']=='evaluate':evaluate(job)
    else:raise ValueError(job)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();run(load(args.job))
