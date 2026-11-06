"""Independent fits, policy search adapters, validation and frozen test evaluation."""
from pathlib import Path
import copy,json,sys,time
import numpy as np
import torch
from experiment_common import ROOT,load,save,sha,report,mark_done,write_csv,read_csv
sys.path.insert(0,str(ROOT/'runtime'))
import run_experiment_4rx as base

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def cfg_for(model,stage='final'):
    assert model=='transformer'
    cfg=copy.deepcopy(load(ROOT/'transformer_protocol.json')['case'][stage+'_cfg'])
    cfg['training']['seed']=2026 if stage=='search' else 2025
    return cfg

def micro_data(cfg):
    folder=ROOT/'micro_dataset';manifest=load(folder/'sampling_manifest.json')
    assert manifest['generation_signature']==cfg['data']['dataset_protocol_sha256']
    result=[]
    for name,count in [('micro_train.npz',cfg['data']['search_train_samples']),('micro_valid.npz',cfg['data']['search_valid_samples'])]:
        assert sha(folder/name)==manifest['artifacts_sha256'][name]
        with np.load(folder/name,allow_pickle=False) as z:result.append({k:z[k].copy() for k in ['X','x','y']})
        assert len(result[-1]['x'])==count
    return result

def full_data(cfg):
    ready=load(ROOT/'preflight/readiness.json')
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

def request_search(model,length,start):
    key=f'search_{model}_l{length}_s{start}'
    save(ROOT/'requests'/f'{key}.json',dict(key=key,kind='search_fit',model=model,
        length=int(length),start=int(start),resource='gpu',gpu_gib=4.,memory_gib=3.,priority=1,deps=['micro_ready']))

def policy(job):
    import policy_search as search
    cfg=cfg_for(job['model'],'search');model=job['model'];folder=ROOT/'search'/model
    cfg['paths']['output_dir']=str(folder);cfg['_localizer']=model;cfg['_variant']='full'
    cfg['_signature']=sha(ROOT/'transformer_protocol.json')+':'+model
    folder.mkdir(parents=True,exist_ok=True);save(folder/'server_config.json',cfg)
    tr,va=micro_data(cfg)
    base.load_splits=lambda local,final,include_test=False:(tr,va,None,tr,va)
    native=search.candidates;budgets={}
    budget_path=folder/'candidate_budgets.csv'
    if budget_path.exists():budgets={(int(r['round']),int(r['slot'])):r for r in read_csv(budget_path)}
    calls={}
    def candidates(local,method,rng,center,length,round_idx,visited):
        slot=calls.get(round_idx,0);calls[round_idx]=slot+1
        picks=native(local,method,rng,center,length,round_idx,visited)
        budgets[round_idx+1,slot]=dict(round=round_idx+1,slot=slot,length=length,
            center=center,evaluations=len(picks),candidate_start_steps=json.dumps(picks))
        write_csv(budget_path,list(budgets.values()))
        return picks
    search.candidates=candidates
    search.prepare_candidates=lambda local,length,picks:[request_search(model,length,start) for start in picks]
    def score(local,length,start,*args):
        target=folder/'score_cache'/f'l{length}_s{start}.json';key=f'search_{model}_l{length}_s{start}'
        if not target.exists():request_search(model,length,start)
        while not target.exists():
            failed=ROOT/'failed'/f'{key}.json'
            if failed.exists():raise RuntimeError('Candidate failed: '+failed.read_text())
            report(job['key'],'waiting_candidate',model=model,length=length,start=start)
            time.sleep(5)
        row=load(target);assert sha(row['checkpoint_source'])==row['checkpoint_sha256']
        return row,0.,True
    search.score=score;search.refit=lambda *args,**kw:None
    search.search(cfg,'guided')
    net=base.PolicyNet(cfg['policy']['hidden_dims'])
    net.load_state_dict(torch.load(folder/'policy_raw.pt',map_location='cpu',weights_only=True))
    starts={str(l):search.predict(net,l,250) for l in range(1,251)}
    assert all(1<=s<=251-int(l) for l,s in starts.items())
    frozen=dict(starts=starts,optimizer_path=str(folder/'policy_raw.pt'),
        optimizer_sha256=sha(folder/'policy_raw.pt'),seed=2026,
        generation_signature=cfg['data']['dataset_protocol_sha256'],
        candidate_position_plan_sha256=sha(ROOT/'candidate_position_plan.json'))
    save(folder/'frozen_optimizer.json',frozen);mark_done(job['key'],**frozen)

def formal_key(model,length,start):return f'fit_{model}_l{length}_s{start}'

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
    for p in sorted((ROOT/'data/data_train').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for k in labels:labels[k].append(z[k].astype(np.float32))
    labels={k:np.concatenate(v) for k,v in labels.items()};scaler=base.TargetScaler(labels)
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state']
    metrics,pred,truth,errors=infer(cfg,state,model,length,start,test,scaler)
    folder=ROOT/'results'/f'{model}_{role}'/f'l{length}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.npz';np.savez_compressed(path,predictions=pred,truth=truth,errors_um=errors)
    row=dict(model=model,condition=role,length=length,start=start,n_test=len(errors),seed=2025,
        training_loss='mse_physical_um2',best_epoch=fit['best_epoch'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=fit['checkpoint_sha256'],
        test_sha256=sha(ROOT/'data/independent_test.npz'),predictions_sha256=sha(path),
        optimizer_sha256=frozen['optimizer_sha256'] if role=='optimized' else None,**metrics)
    write_csv(folder/'final_test.csv',[row]);mark_done(job['key'],result=row)

def select_coarse(job):
    length=job['length'];frozen=load(ROOT/'search/cnn/frozen_optimizer.json')
    wanted=sorted(set(range(1,252-length,5))|{251-length,int(frozen['starts'][str(length)])})
    records=[]
    for start in wanted:
        row=load(ROOT/'fits'/f'cnn_l{length}_s{start}.json')
        records.append(dict(length=length,start=start,validation_mean_error_um=row['mean_error_um'],
            val_mse_um2=row['val_mse_um2'],checkpoint_source=row['checkpoint_source'],checkpoint_sha256=row['checkpoint_sha256']))
    best=min(records,key=lambda r:(r['validation_mean_error_um'],r['start']))
    folder=ROOT/'coarse'/f'l{length}';write_csv(folder/'candidate_validation.csv',records)
    save(folder/'selected_start.json',dict(best,candidates=len(records),learned_start=frozen['starts'][str(length)],selection_metric='mean Euclidean validation error'))
    mark_done(job['key'],start=best['start'],candidates=len(records))

def run(job):
    configure();report(job['key'],'started',kind=job['kind'])
    if job['kind'] in ['search_fit','formal_fit']:train(job)
    elif job['kind']=='policy':policy(job)
    elif job['kind']=='evaluate':evaluate(job)
    elif job['kind']=='coarse_select':select_coarse(job)
    else:raise ValueError(job)
