"""Matched new-seed CNN fixed/optimized start retest; keep published data intact."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv
OUT=ROOT/'cnn_l225_seed2035_fixed_vs_optimized_20261029'
PROTOCOL=ROOT/'cnn_l225_seed2035_pair_protocol.json'


def verify():
    import run_with_convergence_seed2034 as controller
    controller.check_sources(load(ROOT/'deployment_protocol.json'))
    p=load(PROTOCOL)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    frozen=load(ROOT/'search/cnn/frozen_optimizer.json')
    assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']==p['optimizer_sha256']
    assert int(frozen['starts']['225'])==21
    return frozen


def progress(role,phase,**kw):
    save(OUT/role/'progress.json',dict(pid=os.getpid(),at_utc=utc(),phase=phase,**kw))


def worker(role):
    import torch,numpy as np
    import experiment_workers as common
    frozen=verify();common.configure();base=common.base
    cfg=copy.deepcopy(common.cfg_for('cnn'))
    cfg['training'].update(seed=2035,batch_size=256,long_window_batch_size=256)
    cfg['paths']['checkpoint_cache_dir']=str(OUT/role/'checkpoints')
    assert cfg['training']['lr']==.001 and cfg['training']['loss']=='mse_physical_um2'
    start=1 if role=='fixed' else int(frozen['starts']['225'])
    save(OUT/role/'training_config.json',cfg);progress(role,'loading_data',length=225,start=start)
    tr,va=common.full_data(cfg);scaler=base.TargetScaler(tr)
    td,vd=[base.make_dataset(d,scaler,225,start,cfg) for d in [tr,va]]
    rows=[];curve=OUT/role/'curve.csv'
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,start=start,train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);progress(role,'training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,seconds,cached=base.train_or_load(cfg,'cnn225_pair_seed2035','cnn',225,start,
        2035,td,vd,torch.device('cuda'),60,10)
    checkpoint=base.checkpoint_path(cfg,'cnn225_pair_seed2035','cnn',225,start,2035,60,10)
    progress(role,'evaluating',length=225,start=start)
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    metrics,pred,truth,errors=common.infer(cfg,state,'cnn',225,start,test,scaler)
    target=OUT/role/'predictions.npz';np.savez_compressed(target,predictions=pred,truth=truth,errors_um=errors)
    row=dict(model='cnn',condition=role,length=225,start=start,seed=2035,batch_size=256,lr=.001,
        max_epochs=60,patience=10,best_validation_mse_um2=loss,best_epoch=epoch,training_seconds=seconds,
        optimizer_retrained=False,optimizer_sha256=frozen['optimizer_sha256'],n_test=len(errors),
        test_sha256=sha(ROOT/'data/independent_test.npz'),checkpoint_source=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(target),**metrics)
    write_csv(OUT/role/'final_test.csv',[row]);save(OUT/role/'result.json',row);progress(role,'complete',result=row)


def run():
    import fcntl
    verify();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};attempts={};failed={}
        while True:
            for role,item in list(active.items()):
                if alive(item['identity']):continue
                del active[role]
                if not (OUT/role/'final_test.csv').exists() and attempts.get(role,0)>=2:
                    failed[role]=dict(log=str(OUT/role/'worker.log'))
            for role in ['fixed','optimized']:
                if role in active or role in failed or (OUT/role/'final_test.csv').exists():continue
                (OUT/role).mkdir(exist_ok=True)
                with (OUT/role/'worker.log').open('a',buffering=1) as log:
                    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--worker',role],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                active[role]=dict(identity=identity(child.pid));attempts[role]=attempts.get(role,0)+1
            completed=[role for role in ['fixed','optimized'] if (OUT/role/'final_test.csv').exists()]
            state=dict(phase='running',pid=os.getpid(),at_utc=utc(),active=active,completed=completed,failed=failed,length=225,seed=2035)
            if len(completed)==2:
                rows=[read_csv(OUT/role/'final_test.csv')[0] for role in ['fixed','optimized']]
                write_csv(OUT/'comparison.csv',rows)
                reduction=100*(float(rows[0]['mean_error_um'])-float(rows[1]['mean_error_um']))/float(rows[0]['mean_error_um'])
                save(OUT/'status.json',dict(state,phase='complete',relative_error_reduction_percent=reduction,comparison=rows));return
            if failed and not active:save(OUT/'status.json',dict(state,phase='failed'));return
            save(OUT/'status.json',state);time.sleep(5)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True);p=OUT/'launch.json'
    if p.exists():assert not alive(load(p)['identity']),'Already running'
    assert not (OUT/'status.json').exists(),'Inspect existing results before restarting'
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    rec=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL))
    save(p,rec);save(ROOT/'CURRENT_CNN_L225_RETEST.json',rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--worker',choices=['fixed','optimized']);a=p.parse_args()
    if a.launch:launch()
    elif a.run:run()
    elif a.worker:
        try:worker(a.worker)
        except Exception:progress(a.worker,'failed',error=traceback.format_exc());raise
