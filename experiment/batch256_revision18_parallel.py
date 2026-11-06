"""Batch256 localizer refits and validation-selected coarse baselines, isolated outputs."""
from pathlib import Path
import argparse,copy,fcntl,gc,json,os,subprocess,sys,time,traceback
from collections import OrderedDict
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

PROTOCOL=ROOT/'batch256_revision18_protocol_20261029.json'
OUT=ROOT/'batch256_revision18_retests_20261029'

def verify():
    p=load(PROTOCOL)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for path,digest in p['frozen_files_sha256'].items():assert sha(Path(path))==digest,path
    return p

def state():
    path=OUT/'queue_state.json'
    return load(path) if path.exists() else dict(running={},completed=[],failed={},attempts={})

def claim(worker):
    with (OUT/'queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        s=state();p=load(PROTOCOL)
        for key,entry in list(s['running'].items()):
            if not alive(entry['identity']):
                del s['running'][key]
                if s['attempts'].get(key,0)>=2:s['failed'][key]='Worker exited twice; inspect worker logs'
        for job in sorted(p['jobs'],key=lambda j:(j['priority'],-j['length'],j['key'])):
            key=job['key']
            if key in s['completed'] or key in s['running'] or key in s['failed']:continue
            if not all(k in s['completed'] for k in job.get('deps',[])):continue
            s['attempts'][key]=s['attempts'].get(key,0)+1
            s['running'][key]=dict(worker=worker,identity=identity(os.getpid()),started_at_utc=utc())
            save(OUT/'queue_state.json',s);return job
        save(OUT/'queue_state.json',s);return None

def finish_job(job,error=None):
    with (OUT/'queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX);s=state();key=job['key']
        s['running'].pop(key,None)
        if error:
            save(OUT/'attempt_errors'/f"{key}_attempt{s['attempts'][key]}.json",dict(at_utc=utc(),error=error))
            if s['attempts'][key]>=2:s['failed'][key]=error
        elif key not in s['completed']:s['completed'].append(key)
        save(OUT/'queue_state.json',s)

def progress(worker,phase,**kw):
    save(OUT/'progress'/f'worker_{worker}.json',dict(pid=os.getpid(),at_utc=utc(),phase=phase,**kw))

def cfg_for(job):
    if job['case_id']!='base':
        import paper_suite_workers as param
        cfg=param.cfg_for(job['case_id'])
    elif job['model'] in ['gru','transformer']:
        cfg=copy.deepcopy(load(ROOT/(job['model']+'_protocol.json'))['case']['final_cfg'])
    else:cfg=copy.deepcopy(load(ROOT/'deployment_protocol.json')['cases']['cnn']['final_cfg'])
    cfg['training'].update(seed=2025,batch_size=256,long_window_batch_size=256)
    cfg['paths']['checkpoint_cache_dir']=str(OUT/'checkpoints'/job['case_id']/job['model'])
    cfg['evaluation']['success_threshold_um']=.8
    assert cfg['training']['lr']==.001 and cfg['training']['loss']=='mse_physical_um2'
    assert cfg['training']['deterministic'] is False and cfg['training']['num_workers']==0
    return cfg

def metric_threshold(metrics,errors):
    metrics=dict(metrics)
    metrics['success_rate_threshold04']=float(metrics['success_rate'])
    metrics['success_rate']=float((errors<.8).mean());metrics['accuracy08']=metrics['success_rate']
    metrics['success_threshold_um']=.8
    return metrics

def dataset(job,cfg,cache):
    import numpy as np
    import experiment_workers as common
    case=job['case_id']
    if case not in cache:
        if case=='base':
            tr,va=common.full_data(cfg);test_path=ROOT/'data/independent_test.npz'
        else:
            import paper_suite_workers as param
            tr,va=param.full(case,cfg);test_path=param.data_root(case)/'independent_test.npz'
        with np.load(test_path,allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
        cache[case]=(tr,va,test,common.base.TargetScaler(tr),sha(test_path))
        while len(cache)>2:cache.popitem(last=False)
        gc.collect()
    cache.move_to_end(case)
    return cache[case]

def evaluate_alias(alias,job,cfg,state_dict,test,scaler,test_sha,checkpoint,fit):
    import numpy as np
    import experiment_workers as common
    output=OUT/'results'/alias['tag'];output.mkdir(parents=True,exist_ok=True)
    metrics,pred,truth,errors=common.infer(cfg,state_dict,job['model'],job['length'],job['start'],test,scaler)
    metrics=metric_threshold(metrics,errors)
    target=output/'predictions.npz';np.savez_compressed(target,predictions=pred,truth=truth,errors_um=errors)
    record=dict(model=job['model'],condition=alias['condition'],case_id=job['case_id'],
        length=job['length'],start=job['start'],seed=2025,batch_size=256,lr=.001,
        training_loss='mse_physical_um2',deterministic=False,amp=cfg['training']['amp'],
        max_epochs=60,patience=10,optimizer_retrained=False,
        optimizer_sha256=alias.get('optimizer_sha256'),optimizer_source=alias.get('optimizer_source'),
        best_epoch=fit['best_epoch'],best_validation_mse_um2=fit['val_mse_um2'],
        validation_mean_error_um=fit['validation_mean_error_um'],
        n_test=len(errors),test_sha256=test_sha,checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        predictions_sha256=sha(target),previous_source_csv=alias['previous_source_csv'],
        source_training_job=job['key'],**metrics)
    write_csv(output/'final_test.csv',[record]);save(output/'result.json',record)

def fit_job(worker,job,cache):
    import torch
    import experiment_workers as common
    cfg=cfg_for(job);folder=OUT/'fits'/job['key'];folder.mkdir(parents=True,exist_ok=True)
    save(folder/'training_config.json',cfg);progress(worker,'loading_data',job=job['key'],model=job['model'],length=job['length'],case_id=job['case_id'])
    tr,va,test,scaler,test_sha=dataset(job,cfg,cache);base=common.base
    td,vd=[base.make_dataset(d,scaler,job['length'],job['start'],cfg) for d in [tr,va]]
    rows=[];curve=folder/'curve.csv'
    def observe(epoch,training,validation):
        row=dict(epoch=epoch,train_mse_um2=training,val_mse_um2=validation)
        rows.append(row);write_csv(curve,rows)
        progress(worker,'training',job=job['key'],model=job['model'],length=job['length'],start=job['start'],case_id=job['case_id'],**row)
    base.TRAINING_OBSERVER=observe;torch.cuda.reset_peak_memory_stats()
    state_dict,loss,epoch,seconds,cached=base.train_or_load(cfg,'revision18_batch256',job['model'],job['length'],job['start'],2025,
        td,vd,torch.device('cuda'),60,10)
    checkpoint=base.checkpoint_path(cfg,'revision18_batch256',job['model'],job['length'],job['start'],2025,60,10)
    if cached and not rows and curve.exists():rows=read_csv(curve)
    val_metrics,_,_,_=common.infer(cfg,state_dict,job['model'],job['length'],job['start'],va,scaler)
    fit=dict(job=job['key'],model=job['model'],length=job['length'],start=job['start'],case_id=job['case_id'],
        seed=2025,batch_size=256,best_epoch=epoch,val_mse_um2=loss,
        validation_mean_error_um=val_metrics['mean_error_um'],training_seconds=seconds,
        checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),cached=cached,
        epochs_recorded=len(rows),gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    for alias in job['aliases']:
        progress(worker,'testing_direct_result',job=job['key'],tag=alias['tag'])
        evaluate_alias(alias,job,cfg,state_dict,test,scaler,test_sha,checkpoint,fit)
    save(folder/'fit.json',fit)
    del td,vd,state_dict
    base.TRAINING_OBSERVER=None;gc.collect();torch.cuda.empty_cache()

def coarse_job(worker,job,cache):
    import torch
    candidates=[load(OUT/'fits'/key/'fit.json') for key in job['deps']]
    selected=min(candidates,key=lambda r:(r['validation_mean_error_um'],r['start']))
    folder=OUT/'coarse'/f"l{job['length']}";write_csv(folder/'candidate_validation.csv',candidates)
    save(folder/'selected_start.json',dict(selected,candidates=len(candidates),selection_metric='minimum validation mean Euclidean error'))
    parent=next(r for r in load(PROTOCOL)['jobs'] if r['key']==selected['job'])
    cfg=cfg_for(parent);_,_,test,scaler,test_sha=dataset(parent,cfg,cache)
    checkpoint=Path(selected['checkpoint_source']);assert sha(checkpoint)==selected['checkpoint_sha256']
    state_dict=torch.load(checkpoint,map_location='cpu',weights_only=False)['state']
    progress(worker,'testing_validation_selected_coarse',job=job['key'],length=job['length'],start=selected['start'])
    evaluate_alias(job['alias'],parent,cfg,state_dict,test,scaler,test_sha,checkpoint,selected)
    del state_dict;gc.collect();torch.cuda.empty_cache()

def worker_loop(worker):
    import torch
    import experiment_workers as common
    verify();common.configure();cache=OrderedDict()
    progress(worker,'ready')
    while True:
        job=claim(worker)
        if job is None:
            s=state()
            if len(s['completed'])+len(s['failed'])==len(load(PROTOCOL)['jobs']):
                progress(worker,'idle_exit');return
            progress(worker,'waiting_for_dependencies');time.sleep(3);continue
        try:
            if job['kind']=='fit':fit_job(worker,job,cache)
            else:coarse_job(worker,job,cache)
            finish_job(job);progress(worker,'job_complete',job=job['key'])
        except Exception:
            error=traceback.format_exc();finish_job(job,error)
            progress(worker,'failed_attempt',job=job['key'],error=error)
            print(error,flush=True)
            if 'out of memory' in error.lower():return
            gc.collect();torch.cuda.empty_cache()

def resources():
    values=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    free,util=[float(v.strip()) for v in values.stdout.splitlines()[0].split(',')]
    return dict(gpu_free_mib=free,gpu_utilization_percent=util,
        memory_current_gib=int(Path('/sys/fs/cgroup/memory.current').read_text())/2**30,
        memory_limit_gib=int(Path('/sys/fs/cgroup/memory.max').read_text())/2**30)

def controller():
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        workers={};started=time.monotonic();last_status=0
        while True:
            for slot,who in list(workers.items()):
                if not alive(who):del workers[slot]
            with (OUT/'queue.lock').open('a+b') as queue_lock:
                fcntl.flock(queue_lock.fileno(),fcntl.LOCK_EX)
                s=state()
                for key,entry in list(s['running'].items()):
                    if not alive(entry['identity']):
                        del s['running'][key]
                        if s['attempts'].get(key,0)>=2:s['failed'][key]='Worker exited twice; inspect worker logs'
                for job in p['jobs']:
                    if any(k in s['failed'] for k in job.get('deps',[])):
                        s['failed'][job['key']]='Required fit failed'
                save(OUT/'queue_state.json',s)
            todo=len(p['jobs'])-len(s['completed'])-len(s['failed'])
            if not todo:
                rows=[]
                for q in (OUT/'results').glob('*/final_test.csv'):rows+=read_csv(q)
                write_csv(OUT/'all_retest_results.csv',rows)
                phase='complete' if not s['failed'] and len(rows)==20 else 'failed'
                save(OUT/'status.json',dict(phase=phase,at_utc=utc(),completed_jobs=len(s['completed']),total_jobs=len(p['jobs']),
                    final_result_count=len(rows),failed=s['failed'],active_workers=workers))
                return
            if time.monotonic()-last_status>=10:
                usage=resources();control=load(OUT/'parallel_control.json')
                cap=min(int(control['max_workers']),8)
                desired=min(cap,6 if time.monotonic()-started<45 else 8)
                if usage['gpu_free_mib']<3500 or usage['memory_current_gib']>72:desired=min(desired,len(workers))
                pending=len(p['jobs'])-len(s['completed'])-len(s['failed'])-len(s['running'])
                for slot in range(desired):
                    if slot in workers or pending<=0:continue
                    if len(workers)>=6 and usage['gpu_free_mib']<6500:break
                    (OUT/'logs').mkdir(exist_ok=True)
                    with (OUT/'logs'/f'worker_{slot}.log').open('a',buffering=1) as log:
                        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--worker',str(slot)],cwd=ROOT,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                    workers[slot]=identity(child.pid);pending-=1
                rows=[]
                for q in (OUT/'results').glob('*/final_test.csv'):rows+=read_csv(q)
                if rows:write_csv(OUT/'all_retest_results.csv',rows)
                save(OUT/'status.json',dict(phase='running',at_utc=utc(),pid=os.getpid(),completed_jobs=len(s['completed']),
                    training_fits_total=p['fit_count'],total_jobs=len(p['jobs']),final_result_count=len(rows),
                    failed=s['failed'],running=s['running'],active_workers=workers,resources=usage))
                last_status=time.monotonic()
            time.sleep(2)

def launch():
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    f=OUT/'launch.json'
    if f.exists():
        old=load(f)
        if alive(old['identity']):print(json.dumps(dict(already_running=True,**old)));return
    if not (OUT/'parallel_control.json').exists():save(OUT/'parallel_control.json',dict(max_workers=8))
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    info=dict(identity=identity(child.pid),started_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL),
        training_fits=p['fit_count'],total_jobs=len(p['jobs']),target_final_results=20,seed=2025,batch_size=256,
        success_threshold_um=.8,initial_parallel_workers=6,max_parallel_workers=8,optimizer_retrained=False)
    save(f,info);save(ROOT/'CURRENT_BATCH256_REVISION18_RETEST.json',info);print(json.dumps(info),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.launch:launch()
    elif args.run:controller()
    elif args.worker is not None:worker_loop(args.worker)
