"""Additive paper experiments using the frozen simulation and training primitives."""
from pathlib import Path
import copy,json,sys,time
import numpy as np
import torch
from experiment_common import ROOT,load,save,sha,report,mark_done,write_csv,read_csv
sys.path.insert(0,str(ROOT/'runtime'))
import run_experiment_4rx as base
import experiment_workers as common

def specification(case):return load(ROOT/'paper_suite_protocol.json')['parameter_cases'][case]
def folder(case):return ROOT/'parameter_experiments'/case
def data_root(case):return ROOT/specification(case)['dataset_directory']

def cfg_for(case,stage='final'):
    spec=specification(case);root=data_root(case)
    meta=load(root/'dataset_metadata.json');ready=load(root/'readiness.json')
    assert ready['passed'] and sha(root/'dataset_metadata.json')==ready['metadata_sha256']
    assert meta['signature']==spec['generation_signature']==ready['signature']
    assert meta['features']==spec['M'] and meta['n_rx']==spec['n_rx']
    cfg=copy.deepcopy(load(ROOT/'deployment_protocol.json')['cases']['cnn'][stage+'_cfg'])
    cfg['paths'].update(data_dir=str(root),micro_dataset_dir=str(root/'micro_dataset'),
        checkpoint_cache_dir=str(folder(case)/'checkpoints'/stage),output_dir=str(folder(case)/'search'))
    cfg['data'].update(M=spec['M'],source_M=spec['M'],n_rx=spec['n_rx'],molecules=spec['Q'],
        readiness_file=str(root/'readiness.json'),dataset_protocol_sha256=spec['generation_signature'])
    cfg['training']['seed']=2026 if stage=='search' else 2025
    cfg['simulation'].update(features=spec['M'],molecules=spec['Q'])
    cfg['geometry']['receivers']=cfg['geometry']['receivers'][:spec['n_rx']]
    cfg['final']['lengths']=spec['lengths'];cfg['comparison']['cache_sources']=[]
    return cfg

def micro(case,cfg):
    root=data_root(case);meta=load(root/'dataset_metadata.json');result=[]
    for name,count in [('micro_train.npz',12840),('micro_valid.npz',3424)]:
        path=root/'micro_dataset'/name;assert sha(path)==meta['micro_sha256'][name]
        with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ['X','x','y']}
        assert d['X'].shape==(count,cfg['data']['M']*cfg['data']['n_rx'])
        with np.load(ROOT/'micro_dataset'/name,allow_pickle=False) as z:
            assert np.array_equal(d['x'],z['x']) and np.array_equal(d['y'],z['y'])
        result.append(d)
    return result

def full(case,cfg):
    root=data_root(case);meta=load(root/'dataset_metadata.json')
    for rel,digest in meta['shards_sha256'].items():
        if rel.startswith(('data_train/','data_valid/')):assert sha(root/rel)==digest
    tr,va,_,_,_=base.load_splits(cfg,final=True,include_test=False)
    assert (len(tr['x']),len(va['x']))==(410640,25500)
    assert tr['X'].shape[1]==va['X'].shape[1]==cfg['data']['M']*cfg['data']['n_rx']
    return tr,va

def convergence(job):
    model=job['model'];length=100
    adapter=__import__('gru_workers' if model=='gru' else 'transformer_workers' if model=='transformer' else 'experiment_workers')
    cfg=adapter.cfg_for(model);frozen=load(ROOT/'search'/model/'frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
    start=int(frozen['starts'][str(length)])
    out=ROOT/'paper_figures/fig7'/model;out.mkdir(parents=True,exist_ok=True)
    cfg['paths']['checkpoint_cache_dir']=str(out/'checkpoints')
    tr,va=adapter.full_data(cfg);scaler=base.TargetScaler(tr)
    td,vd=[base.make_dataset(d,scaler,length,start,cfg) for d in [tr,va]]
    curve=out/'curve.csv';rows=[]
    def observer(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,model=model,length=length,start=start,seed=2025,
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);report(job['key'],'training',**row)
    base.TRAINING_OBSERVER=observer
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,'convergence60',model,length,start,
        2025,td,vd,torch.device('cuda'),60,61)
    if cached:rows=read_csv(curve)
    assert [int(r['epoch']) for r in rows]==list(range(1,61))
    checkpoint=base.checkpoint_path(cfg,'convergence60',model,length,start,2025,60,61)
    record=dict(model=model,length=length,start=start,epochs=60,early_stopping=False,
        loss='mse_physical_um2',seed=2025,optimizer_sha256=frozen['optimizer_sha256'],
        curve_sha256=sha(curve),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        best_validation_epoch=epoch,best_validation_mse_um2=loss,training_seconds=elapsed)
    save(out/'result.json',record);mark_done(job['key'],**record)

def parameter_train(job):
    case=job['case_id'];stage='search' if job['kind']=='parameter_search_fit' else 'final'
    cfg=cfg_for(case,stage);length=job['length'];start=job['start']
    tr,va=micro(case,cfg) if stage=='search' else full(case,cfg)
    scaler=base.TargetScaler(tr);td,vd=[base.make_dataset(d,scaler,length,start,cfg) for d in [tr,va]]
    checkpoint=base.checkpoint_path(cfg,stage,'cnn',length,start,cfg['training']['seed'],60,10)
    curve=checkpoint.with_suffix('.curve.csv');rows=[]
    def observer(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,case_id=case,model='cnn',length=length,start=start,
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);report(job['key'],'training',**row)
    base.TRAINING_OBSERVER=observer
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,stage,'cnn',length,start,
        cfg['training']['seed'],td,vd,torch.device('cuda'),60,10)
    metrics,_,_,_=common.infer(cfg,state,'cnn',length,start,va,scaler)
    row=dict(case_id=case,model='cnn',length=length,start=start,best_epoch=epoch,
        val_mse_um2=loss,val_mean_error_um=metrics['mean_error_um'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_seconds=elapsed,reused_checkpoint=cached,**metrics)
    target=folder(case)/('search/score_cache' if stage=='search' else 'fits')/f'l{length}_s{start}.json'
    save(target,row);mark_done(job['key'],result=str(target),**row)

def request(case,length,start):
    key=f'parameter_search_{case}_l{length}_s{start}'
    save(ROOT/'requests'/f'{key}.json',dict(key=key,kind='parameter_search_fit',case_id=case,
        model='cnn',length=int(length),start=int(start),resource='gpu',gpu_gib=2.5,memory_gib=3.,
        priority=30,deps=['main_supplement_complete','dataset_ready_'+case]))

def parameter_policy(job):
    import policy_search as search
    case=job['case_id'];cfg=cfg_for(case,'search');out=folder(case)/'search';M=cfg['data']['M']
    out.mkdir(parents=True,exist_ok=True)
    cfg['_localizer']='cnn';cfg['_variant']='full';cfg['_signature']=sha(ROOT/'paper_suite_protocol.json')+':'+case
    save(out/'server_config.json',cfg);tr,va=micro(case,cfg)
    base.load_splits=lambda local,final,include_test=False:(tr,va,None,tr,va)
    native=search.candidates;calls={};budgets={};budgetfile=out/'candidate_budgets.csv'
    if budgetfile.exists():budgets={(int(r['round']),int(r['slot'])):r for r in read_csv(budgetfile)}
    def candidates(local,method,rng,center,length,round_idx,visited):
        slot=calls.get(round_idx,0);calls[round_idx]=slot+1
        picks=native(local,method,rng,center,length,round_idx,visited)
        budgets[round_idx+1,slot]=dict(round=round_idx+1,slot=slot,length=length,
            center=center,evaluations=len(picks),candidate_start_steps=json.dumps(picks))
        write_csv(budgetfile,[budgets[k] for k in sorted(budgets)]);return picks
    search.candidates=candidates
    search.prepare_candidates=lambda local,length,picks:[request(case,length,start) for start in picks]
    def score(local,length,start,*args):
        target=out/'score_cache'/f'l{length}_s{start}.json'
        key=f'parameter_search_{case}_l{length}_s{start}'
        if not target.exists():request(case,length,start)
        while not target.exists():
            failed=ROOT/'failed'/f'{key}.json'
            if failed.exists():raise RuntimeError(failed.read_text())
            report(job['key'],'waiting_candidate',case_id=case,length=length,start=start);time.sleep(5)
        row=load(target);assert sha(row['checkpoint_source'])==row['checkpoint_sha256']
        return row,0.,True
    search.score=score;search.refit=lambda *args,**kw:None;search.search(cfg,'guided')
    net=base.PolicyNet(cfg['policy']['hidden_dims'])
    net.load_state_dict(torch.load(out/'policy_raw.pt',map_location='cpu',weights_only=True))
    starts={str(l):search.predict(net,l,M) for l in range(1,M+1)}
    assert all(1<=s<=M-int(l)+1 for l,s in starts.items())
    frozen=dict(starts=starts,optimizer_path=str(out/'policy_raw.pt'),optimizer_sha256=sha(out/'policy_raw.pt'),
        generation_signature=cfg['data']['dataset_protocol_sha256'],M=M,case_id=case,seed=2026,
        suite_protocol_sha256=sha(ROOT/'paper_suite_protocol.json'))
    save(out/'frozen_optimizer.json',frozen);mark_done(job['key'],**frozen)

def parameter_evaluate(job):
    case=job['case_id'];cfg=cfg_for(case);length=job['length'];root=data_root(case)
    frozen=load(folder(case)/'search/frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
    start=int(frozen['starts'][str(length)])
    fit=load(folder(case)/'fits'/f'l{length}_s{start}.json')
    checkpoint=Path(fit['checkpoint_source']);assert sha(checkpoint)==fit['checkpoint_sha256']
    labels={k:[] for k in ['x','y']}
    for p in sorted((root/'data_train').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for k in labels:labels[k].append(z[k].astype(np.float32))
    scaler=base.TargetScaler({k:np.concatenate(v) for k,v in labels.items()})
    meta=load(root/'dataset_metadata.json');assert sha(root/'independent_test.npz')==meta['test_sha256']
    with np.load(root/'independent_test.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in ['X','x','y']}
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:
        assert np.array_equal(raw['x'],z['x']) and np.array_equal(raw['y'],z['y'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state']
    metrics,pred,truth,errors=common.infer(cfg,state,'cnn',length,start,raw,scaler)
    out=folder(case)/'results'/f'l{length}';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',predictions=pred,truth=truth,errors_um=errors)
    row=dict(case_id=case,model='cnn',condition='optimized',M=cfg['data']['M'],
        n_rx=cfg['data']['n_rx'],Q=cfg['data']['molecules'],length=length,start=start,n_test=len(errors),
        seed=2025,training_loss='mse_physical_um2',optimizer_sha256=frozen['optimizer_sha256'],
        test_sha256=meta['test_sha256'],checkpoint_sha256=fit['checkpoint_sha256'],
        predictions_sha256=sha(out/'predictions.npz'),**metrics)
    write_csv(out/'final_test.csv',[row]);mark_done(job['key'],result=row)

def mle_supplement(job):
    import run_mle_current as mle
    mle.LENGTHS=list(range(25,251,25));mle.run()
    mark_done(job['key'],lengths=mle.LENGTHS,optimizer_reused=True,templates_reused=True)

def run(job):
    common.configure();report(job['key'],'started',kind=job['kind'])
    if job['kind']=='convergence':convergence(job)
    elif job['kind']=='parameter_policy':parameter_policy(job)
    elif job['kind'] in ['parameter_search_fit','parameter_fit']:parameter_train(job)
    elif job['kind']=='parameter_evaluate':parameter_evaluate(job)
    elif job['kind']=='mle_supplement':mle_supplement(job)
    elif job['kind'] in ['paper_figure','barrier']:
        if job['kind']=='paper_figure':
            import paper_suite_figures as figures;figures.run(job)
        mark_done(job['key'],figure=job.get('figure'))
    else:raise ValueError(job)
