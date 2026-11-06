"""Full-data localization tests for the frozen ten-layer DNN start optimizer."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

OUT=ROOT/'dnn10_width100_optimized_tests_seed2034_20261029'
POLICY=ROOT/'start_optimizer_dnn10_width100_20261029'
PREVIOUS=ROOT/'ablations/dnn10_width100_l250_seed2034_20261029'
PROTOCOL=ROOT/'dnn10_optimized_tests_protocol.json'
LENGTHS=list(range(25,251,25))


def verify():
    import dnn10_optimizer_parallel as optimizer
    optimizer.verify()
    protocol=load(PROTOCOL)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert load(POLICY/'status.json')['phase']=='complete'
    frozen=load(POLICY/'search/frozen_optimizer.json')
    assert sha(POLICY/'search/frozen_optimizer.json')==protocol['frozen_optimizer_record_sha256']
    assert frozen['architecture']=='ten_layers_width100'
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']==protocol['optimizer_sha256']
    assert sha(ROOT/'data/independent_test.npz')==protocol['test_sha256']
    return protocol,frozen


def cfg_for(length):
    cfg=copy.deepcopy(load(PREVIOUS/'deep/training_config.json'))
    cfg['paths']['checkpoint_cache_dir']=str(OUT/f'l{length}'/'checkpoints')
    assert cfg['models']['dnn'][0]['hidden_dims']==[100]*10
    assert cfg['training']['seed']==2034 and cfg['training']['lr']==.001
    assert cfg['training']['batch_size']==cfg['training']['long_window_batch_size']==256
    assert cfg['training']['loss']=='mse_physical_um2' and cfg['training']['deterministic'] is False
    return cfg


def progress(length,phase,**kw):
    rec=dict(phase=phase,pid=os.getpid(),at_utc=utc(),length=length,**kw)
    save(OUT/'progress'/f'l{length}.json',rec);print(json.dumps(rec),flush=True)


def reuse_l250(frozen):
    folder=OUT/'l250';target=folder/'final_test.csv'
    if target.exists():
        assert read_csv(target)[0]['optimizer_sha256']==frozen['optimizer_sha256']
        save(OUT/'done/l250.json',dict(at_utc=utc(),result=str(target),reused=True))
        return
    old=load(PREVIOUS/'deep/result.json');cfg=cfg_for(250)
    assert sha(old['checkpoint'])==old['checkpoint_sha256']
    assert sha(PREVIOUS/'deep/curve.csv')==old['curve_sha256']
    assert old['seed']==2034 and old['epochs']==60 and old['start']==int(frozen['starts']['250'])==1
    assert old['early_stopping'] is False
    source=next(r for r in read_csv(PREVIOUS/'comparison.csv') if r['architecture']=='ten_layers_width100')
    row=dict(source,model='dnn',condition='optimized',optimizer_sha256=frozen['optimizer_sha256'],
        checkpoint_source=old['checkpoint'],reused_existing_l250=True,
        reuse_reason='Unique feasible detection start at l250; identical input, model and training protocol',
        training_loss='mse_physical_um2',early_stopping=False)
    save(folder/'training_config.json',cfg)
    curve=folder/'curve.csv';curve.write_bytes((PREVIOUS/'deep/curve.csv').read_bytes())
    prediction=folder/'predictions.npz';prediction.write_bytes((PREVIOUS/'ten_layers_predictions.npz').read_bytes())
    row['predictions_sha256']=sha(prediction);write_csv(target,[row])
    save(folder/'result.json',dict(old,reused_existing_l250=True,optimizer_sha256=frozen['optimizer_sha256']))
    save(OUT/'done/l250.json',dict(at_utc=utc(),result=str(target),reused=True))


def worker(length):
    import numpy as np
    import torch
    import experiment_workers as common
    import dnn10_width100_cnn50 as architecture
    protocol,frozen=verify();common.configure();base=common.base;architecture.install_model(base)
    assert length in LENGTHS and length!=250
    start=int(frozen['starts'][str(length)]);assert 1<=start<=251-length
    cfg=cfg_for(length);folder=OUT/f'l{length}';save(folder/'training_config.json',cfg)
    progress(length,'loading_data',start=start,seed=2034,batch_size=256)
    tr,va=common.full_data(cfg);scaler=base.TargetScaler(tr)
    assert len(tr['x'])==410640 and len(va['x'])==25500
    td,vd=[base.make_dataset(d,scaler,length,start,cfg) for d in [tr,va]]
    rows=[];curve=folder/'curve.csv'
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,length=length,start=start,seed=2034,batch_size=256,
            train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows)
        progress(length,'training',epoch=epoch,start=start,train_mse_um2=train_mse,val_mse_um2=val_mse)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,'dnn10_optimized_full60','dnn',length,start,
        2034,td,vd,torch.device('cuda'),60,61)
    if cached:rows=read_csv(curve)
    assert [int(r['epoch']) for r in rows]==list(range(1,61))
    checkpoint=base.checkpoint_path(cfg,'dnn10_optimized_full60','dnn',length,start,2034,60,61)
    record=dict(length=length,start=start,seed=2034,batch_size=256,lr=.001,epochs=60,early_stopping=False,
        best_validation_mse_um2=loss,best_validation_epoch=epoch,training_seconds=elapsed,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),curve_sha256=sha(curve),
        reused_checkpoint=cached,optimizer_sha256=frozen['optimizer_sha256'])
    save(folder/'result.json',record);progress(length,'evaluating',start=start)
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    metrics,pred,truth,errors=common.infer(cfg,state,'dnn',length,start,test,scaler)
    prediction=folder/'predictions.npz';np.savez_compressed(prediction,predictions=pred,truth=truth,errors_um=errors)
    net=base.build_localizer(cfg,'dnn',length);parameters=sum(w.numel() for w in net.parameters() if w.requires_grad);del net
    row=dict(architecture='ten_layers_width100',model='dnn',condition='optimized',length=length,start=start,
        parameters=parameters,seed=2034,batch_size=256,lr=.001,epochs=60,early_stopping=False,n_test=len(errors),
        training_loss='mse_physical_um2',final_train_mse_um2=float(rows[-1]['train_mse_um2']),
        final_validation_mse_um2=float(rows[-1]['val_mse_um2']),best_validation_mse_um2=loss,best_validation_epoch=epoch,
        optimizer_sha256=frozen['optimizer_sha256'],test_sha256=protocol['test_sha256'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(prediction),
        reused_existing_l250=False,**metrics)
    write_csv(folder/'final_test.csv',[row]);save(OUT/'done'/f'l{length}.json',dict(at_utc=utc(),result=str(folder/'final_test.csv')))
    progress(length,'complete',result=row)


def collect():
    rows=[]
    for length in LENGTHS:
        p=OUT/f'l{length}/final_test.csv'
        if p.exists():rows+=read_csv(p)
    if rows:write_csv(OUT/'all_test_results.csv',rows)
    return rows


def external_fits():
    from dnn10_optimizer_parallel import external_fits as basic
    found={v['pid']:v for v in basic()}
    path=ROOT/'CURRENT_DNN10_START_OPTIMIZER.json'
    if path.exists():
        d=Path(load(path)['output_directory']);p=d/'active.json'
        if p.exists():
            for v in load(p).values():
                if v['kind']=='candidate' and alive(v['identity']):found[v['identity']['pid']]=v['identity']
    return list(found.values())


def run():
    import fcntl
    from dnn10_optimizer_parallel import gpu_free
    _,frozen=verify()
    for folder in ['logs','done','failed','progress']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reuse_l250(frozen)
        active=load(OUT/'active.json') if (OUT/'active.json').exists() else {}
        attempts=load(OUT/'attempts.json') if (OUT/'attempts.json').exists() else {}
        while True:
            for key,item in list(active.items()):
                if alive(item['identity']):continue
                del active[key]
                if not (OUT/'done'/f'{key}.json').exists() and attempts.get(key,1)>=2:
                    save(OUT/'failed'/f'{key}.json',dict(at_utc=utc(),log=str(OUT/'logs'/f'{key}.log')))
            completed={p.stem for p in (OUT/'done').glob('*.json')}
            failed={p.stem:load(p) for p in (OUT/'failed').glob('*.json')}
            external=external_fits()
            for length in LENGTHS:
                key=f'l{length}'
                if key in completed or key in active or key in failed:continue
                if len(active)+len(external)>=6:break
                if shutil.disk_usage(OUT).free/2**30<3 or gpu_free()<3.5:break
                anon=int(dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())['anon'])/2**30
                if anon+7>80 or len(active)*7+len(external)*9+7>60:break
                with (OUT/'logs'/f'{key}.log').open('a',buffering=1) as log:
                    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--worker',str(length)],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                active[key]=dict(length=length,identity=identity(child.pid));attempts[key]=attempts.get(key,0)+1
                save(OUT/'active.json',active);save(OUT/'attempts.json',attempts);time.sleep(.5)
            rows=collect()
            state=dict(phase='running',pid=os.getpid(),at_utc=utc(),completed_lengths=[int(r['length']) for r in rows],
                active=active,external_training_processes=external,failed=failed,lengths=LENGTHS,max_shared_parallel_fits=6,
                optimizer_sha256=frozen['optimizer_sha256'],disk_free_gib=shutil.disk_usage(OUT).free/2**30)
            save(OUT/'active.json',active)
            if len(rows)==10 and not active:
                save(OUT/'status.json',dict(state,phase='complete',completed_at_utc=utc()));return
            if failed and not active:
                save(OUT/'status.json',dict(state,phase='failed'));raise RuntimeError(failed)
            save(OUT/'status.json',state);time.sleep(3)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True);target=OUT/'launch.json'
    if target.exists() and alive(load(target)['identity']):print(json.dumps(load(target)));return
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    rec=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL))
    save(target,rec);save(ROOT/'CURRENT_DNN10_FORMAL_TESTS.json',rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker',type=int);args=p.parse_args()
    if args.launch:launch()
    elif args.run:run()
    elif args.worker:
        try:worker(args.worker)
        except Exception:progress(args.worker,'error',error=traceback.format_exc());raise
