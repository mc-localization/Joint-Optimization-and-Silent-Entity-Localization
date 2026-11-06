"""Isolated asynchronous start optimizer for the ten-layer width100 DNN."""
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

OUT=ROOT/'start_optimizer_dnn10_width100_20261029'
PROTOCOL=ROOT/'dnn10_optimizer_protocol.json'


def verify():
    import dnn10_width100_cnn50 as architecture
    architecture.verify()
    p=load(PROTOCOL)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return p


def progress(key,phase,**fields):
    rec=dict(phase=phase,pid=os.getpid(),at_utc=utc(),**fields)
    save(OUT/'progress'/f'{key}.json',rec);print(json.dumps(rec),flush=True)


def done(key,**fields):save(OUT/'done'/f'{key}.json',dict(at_utc=utc(),**fields))


def setup():
    import experiment_workers as common
    import dnn10_width100_cnn50 as architecture
    common.configure();architecture.install_model(common.base)
    return common


def cfg_for(common):
    import dnn10_width100_cnn50 as architecture
    cfg=copy.deepcopy(common.cfg_for('dnn','search'))
    cfg['models']['dnn']=[dict(id=architecture.CONFIG_ID,hidden_dims=[100]*10,dropout=.05)]
    cfg['training'].update(seed=2026,batch_size=256,long_window_batch_size=256)
    cfg['paths'].update(checkpoint_cache_dir=str(OUT/'checkpoints/search'),output_dir=str(OUT/'search'))
    return cfg


def request(length,start):
    key=f'candidate_l{length}_s{start}'
    save(OUT/'requests'/f'{key}.json',dict(key=key,kind='candidate',length=int(length),start=int(start)))
    return key


def candidate(job,common):
    import torch
    base=common.base;cfg=cfg_for(common);length=job['length'];start=job['start'];key=job['key']
    tr,va=common.micro_data(cfg);scaler=base.TargetScaler(tr)
    td,vd=[base.make_dataset(data,scaler,length,start,cfg) for data in [tr,va]]
    checkpoint=base.checkpoint_path(cfg,'dnn10_policy_search','dnn',length,start,2026,60,10)
    curve=checkpoint.with_suffix('.curve.csv');rows=[]
    def observe(epoch,train_mse,val_mse):
        row=dict(epoch=epoch,length=length,start=start,train_mse_um2=train_mse,val_mse_um2=val_mse)
        rows.append(row);write_csv(curve,rows);progress(key,'training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,elapsed,cached=base.train_or_load(cfg,'dnn10_policy_search','dnn',length,start,
        2026,td,vd,torch.device('cuda'),60,10)
    metrics,_,_,_=common.infer(cfg,state,'dnn',length,start,va,scaler)
    row=dict(model='dnn',architecture='ten_layers_width100',length=length,start=start,seed=2026,
        batch_size=256,best_epoch=epoch,val_mse_um2=loss,val_mean_error_um=metrics['mean_error_um'],
        checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),training_seconds=elapsed,
        reused_checkpoint=cached,gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        generation_signature=cfg['data']['dataset_protocol_sha256'],**metrics)
    target=OUT/'search/score_cache'/f'l{length}_s{start}.json'
    save(target,row);done(key,result=str(target));progress(key,'complete',result=row)


def policy(common):
    import torch
    import policy_search as search
    base=common.base;cfg=cfg_for(common);folder=OUT/'search';folder.mkdir(parents=True,exist_ok=True)
    cfg.update(_localizer='dnn',_variant='full',_signature=sha(PROTOCOL)+':dnn10_width100')
    save(folder/'server_config.json',cfg);tr,va=common.micro_data(cfg)
    base.load_splits=lambda local,final,include_test=False:(tr,va,None,tr,va)
    native=search.candidates;calls={};budgetfile=folder/'candidate_budgets.csv'
    budgets={(int(r['round']),int(r['slot'])):r for r in read_csv(budgetfile)} if budgetfile.exists() else {}
    def candidates(local,method,rng,center,length,round_idx,visited):
        slot=calls.get(round_idx,0);calls[round_idx]=slot+1
        picks=native(local,method,rng,center,length,round_idx,visited)
        budgets[round_idx+1,slot]=dict(round=round_idx+1,slot=slot,length=length,center=center,
            evaluations=len(picks),candidate_start_steps=json.dumps(picks))
        write_csv(budgetfile,[budgets[k] for k in sorted(budgets)]);return picks
    search.candidates=candidates
    search.prepare_candidates=lambda local,length,picks:[request(length,start) for start in picks]
    def score(local,length,start,*args):
        target=folder/'score_cache'/f'l{length}_s{start}.json';key=request(length,start)
        while not target.exists():
            failed=OUT/'failed'/f'{key}.json'
            if failed.exists():raise RuntimeError(failed.read_text())
            progress('policy','waiting_candidate',length=length,start=start);time.sleep(5)
        row=load(target);assert sha(row['checkpoint_source'])==row['checkpoint_sha256']
        return row,0.,True
    search.score=score;search.refit=lambda *args,**kw:None
    search.search(cfg,'guided')
    net=base.PolicyNet(cfg['policy']['hidden_dims'])
    net.load_state_dict(torch.load(folder/'policy_raw.pt',map_location='cpu',weights_only=True))
    starts={str(length):search.predict(net,length,250) for length in range(1,251)}
    assert all(1<=start<=251-int(length) for length,start in starts.items())
    record=dict(starts=starts,optimizer_path=str(folder/'policy_raw.pt'),optimizer_sha256=sha(folder/'policy_raw.pt'),
        seed=2026,architecture='ten_layers_width100',candidate_batch_size=256,
        generation_signature=cfg['data']['dataset_protocol_sha256'],
        candidate_position_plan_sha256=sha(ROOT/'candidate_position_plan.json'),protocol_sha256=sha(PROTOCOL))
    save(folder/'frozen_optimizer.json',record);done('policy',**record);progress('policy','complete')


def worker(key):
    verify();common=setup()
    if key=='policy':policy(common)
    else:candidate(load(OUT/'requests'/f'{key}.json'),common)


def external_fits():
    found={}
    for item in load(ROOT/'active.json').values():
        if item['job'].get('resource')=='gpu' and alive(item['identity']):found[item['identity']['pid']]=item['identity']
    path=ROOT/'CURRENT_DNN_DEPTH_ABLATION.json'
    if path.exists():
        rec=load(path)
        if alive(rec['identity']):found[rec['identity']['pid']]=rec['identity']
    return list(found.values())


def gpu_free():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    return int(p.stdout.splitlines()[0])/1024 if p.returncode==0 else 0


def run():
    import fcntl
    verify()
    for folder in ['requests','progress','done','failed','logs']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=load(OUT/'active.json') if (OUT/'active.json').exists() else {}
        attempts=load(OUT/'attempts.json') if (OUT/'attempts.json').exists() else {}
        def spawn(key,kind):
            with (OUT/'logs'/f'{key}.log').open('a',buffering=1) as log:
                child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--worker',key],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
            active[key]=dict(kind=kind,identity=identity(child.pid));attempts[key]=attempts.get(key,0)+1
            save(OUT/'attempts.json',attempts);save(OUT/'active.json',active)
        while True:
            for key,item in list(active.items()):
                if alive(item['identity']):continue
                del active[key]
                if not (OUT/'done'/f'{key}.json').exists() and attempts.get(key,1)>=2:
                    save(OUT/'failed'/f'{key}.json',dict(at_utc=utc(),attempts=attempts[key],log=str(OUT/'logs'/f'{key}.log')))
            failed={p.stem:load(p) for p in (OUT/'failed').glob('*.json')}
            completed={p.stem for p in (OUT/'done').glob('*.json')}
            external=external_fits();free_disk=shutil.disk_usage(OUT).free/2**30
            if 'policy' not in active and 'policy' not in completed and 'policy' not in failed:spawn('policy','policy')
            requests={p.stem:load(p) for p in (OUT/'requests').glob('*.json')}
            for key in sorted(requests):
                if key in active or key in completed or key in failed:continue
                if sum(v['kind']=='candidate' for v in active.values())+len(external)>=6:break
                if free_disk<3 or gpu_free()<3:break
                anon=int(dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())['anon'])/2**30
                if anon+3>80:break
                spawn(key,'candidate');time.sleep(.4)
            trace=read_csv(OUT/'search/guided_trace.csv') if (OUT/'search/guided_trace.csv').exists() else []
            state=dict(phase='running',pid=os.getpid(),at_utc=utc(),active=active,
                external_training_processes=external,max_total_parallel_fits=6,
                completed_candidate_fits=sum(k!='policy' for k in completed),requested_candidate_fits=len(requests),
                completed_rounds=len(trace),total_rounds=10,failed=failed,disk_free_gib=free_disk)
            save(OUT/'active.json',active)
            if 'policy' in completed and not active:
                save(OUT/'status.json',dict(state,phase='complete',completed_at_utc=utc()));return
            if failed and not active:
                save(OUT/'status.json',dict(state,phase='failed'));raise RuntimeError(failed)
            save(OUT/'status.json',state);time.sleep(3)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'launch.json'
    if p.exists() and alive(load(p)['identity']):print(json.dumps(load(p)));return
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    rec=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL))
    save(p,rec);save(ROOT/'CURRENT_DNN10_START_OPTIMIZER.json',rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker')
    args=parser.parse_args()
    if args.launch:launch()
    elif args.run:run()
    elif args.worker:
        try:worker(args.worker)
        except Exception:progress(args.worker,'error',error=traceback.format_exc());raise
