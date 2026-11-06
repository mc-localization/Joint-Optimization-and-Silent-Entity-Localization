"""Run only the authorized batch-256 CNN/Transformer localizer retest."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

BASE=ROOT.parent
MODELS=['cnn','transformer']
LENGTHS=list(range(25,251,25))
sys.dont_write_bytecode=True

def check_sources():
    p=load(ROOT/'protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for manifest in ['deployment_protocol.json','transformer_protocol.json','gru_protocol.json','paper_suite_protocol.json']:
        for name,digest in load(BASE/manifest)['source_sha256'].items():assert sha(BASE/name)==digest,name
    for model in MODELS:
        cfg=p['cases'][model]['final_cfg'];tr=cfg['training']
        assert tr['batch_size']==tr['long_window_batch_size']==256
        assert tr['seed']==2033 and tr['deterministic'] is False
        assert tr['lr']==.001 and tr['loss']=='mse_physical_um2'
        record=p['reused_optimizers'][model]
        path=ROOT/'search'/model/'frozen_optimizer.json';frozen=load(path)
        assert sha(path)==record['source_frozen_sha256']==sha(BASE/record['source_frozen_file'])
        assert sha(frozen['optimizer_path'])==record['optimizer_sha256']==frozen['optimizer_sha256']
    return p

def jobs_for():
    jobs={}
    for length in LENGTHS:
        for model in MODELS:
            start=int(load(ROOT/'search'/model/'frozen_optimizer.json')['starts'][str(length)])
            fit=f'b256_s2033_fit_{model}_l{length}_s{start}'
            jobs[fit]=dict(key=fit,kind='formal_fit',model=model,length=length,start=start,deps=[],
                gpu_gib=2.5 if model=='cnn' else 3.,memory_gib=8.)
            key=f'b256_s2033_eval_{model}_l{length}'
            jobs[key]=dict(key=key,kind='evaluate',model=model,length=length,start=start,role='optimized',
                deps=[fit],gpu_gib=1.,memory_gib=2.)
    return jobs

def gpu_free():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    assert r.returncode==0,r.stderr
    return float(r.stdout.strip().splitlines()[0])/1024

def collect():
    rows=[]
    for model in MODELS:
        for length in LENGTHS:
            p=ROOT/'results'/f'{model}_optimized'/f'l{length}'/'final_test.csv'
            if p.exists():rows+=read_csv(p)
    write_csv(ROOT/'all_test_results.csv',rows)
    return len(rows)

def run():
    check_sources()
    with (ROOT/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=jobs_for();active={};failures={}
        attempts=load(ROOT/'attempts.json') if (ROOT/'attempts.json').exists() else {}
        if (ROOT/'active.json').exists():
            for key,v in load(ROOT/'active.json').items():
                if alive(v['identity']):active[key]=dict(**v,process=None,log=None)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        save(ROOT/'queue.json',list(jobs.values()))
        while True:
            for key,v in list(active.items()):
                if v['process'] is not None:
                    rc=v['process'].poll()
                    if rc is None:continue
                    v['log'].close()
                elif alive(v['identity']):continue
                else:rc=None
                del active[key]
                if not (ROOT/'done'/f'{key}.json').exists():
                    error=dict(exit_code=rc,attempt=attempts[key],at_utc=utc(),log=str(ROOT/'logs'/f'{key}.log'))
                    if attempts[key]>=2:save(ROOT/'failed'/f'{key}.json',error)
                    else:save(ROOT/'retry'/f'{key}.json',error)
            completed={p.stem for p in (ROOT/'done').glob('*.json')}
            failures={p.stem:load(p) for p in (ROOT/'failed').glob('*.json')}
            ready=sorted([j for key,j in jobs.items() if key not in completed and key not in active and key not in failures and set(j['deps'])<=completed],
                key=lambda j:(0 if j['kind']=='evaluate' else 1,j['length'],j['model']))
            for job in ready:
                current=[v['job'] for v in active.values()]
                if len(current)>=6:break
                if shutil.disk_usage(ROOT).free/2**30<3:break
                if sum(j['gpu_gib'] for j in current)+job['gpu_gib']>21.5:continue
                if gpu_free()<job['gpu_gib']+1:continue
                if sum(j['memory_gib'] for j in current)+job['memory_gib']>60:continue
                anon=int(dict(x.split() for x in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())['anon'])/2**30
                if anon+job['memory_gib']>80:continue
                key=job['key'];save(ROOT/'jobs'/f'{key}.json',job)
                (ROOT/'logs').mkdir(exist_ok=True);log=(ROOT/'logs'/f'{key}.log').open('a',buffering=1)
                child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'workers.py'),'--job',str(ROOT/'jobs'/f'{key}.json')],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                active[key]=dict(job=job,identity=identity(child.pid),process=child,log=log)
                attempts[key]=attempts.get(key,0)+1;save(ROOT/'attempts.json',attempts)
                save(ROOT/'active.json',{k:dict(job=v['job'],identity=v['identity']) for k,v in active.items()})
                time.sleep(.5)
            n_results=collect()
            state=dict(phase='running' if active else 'waiting',at_utc=utc(),pid=os.getpid(),seed=2033,batch_size=256,
                models=MODELS,lengths=LENGTHS,optimizer_reused=True,optimizer_training=False,max_parallel_gpu_workers=6,
                completed_jobs=len(set(jobs)&completed),total_jobs=40,completed_tests=n_results,total_tests=20,
                jobs_pending=len(set(jobs)-completed-set(active)),failed=failures,
                active={k:dict(identity=v['identity'],kind=v['job']['kind'],model=v['job']['model'],length=v['job']['length']) for k,v in active.items()})
            save(ROOT/'status.json',state)
            save(ROOT/'active.json',{k:dict(job=v['job'],identity=v['identity']) for k,v in active.items()})
            if not active and set(jobs)<=completed:
                save(ROOT/'status.json',dict(state,phase='complete',completed_at_utc=utc()));break
            if not active and failures and not ready:
                save(ROOT/'status.json',dict(state,phase='needs_attention'));raise RuntimeError('Inspect failed/ and logs/.')
            time.sleep(3)

def launch():
    check_sources()
    p=ROOT/'launch.json'
    if p.exists() and alive(load(p)['identity']):print(json.dumps(load(p)));return
    for name in ['launch.json','future_datasets/launch.json']:
        previous=BASE/name
        assert not previous.exists() or not alive(load(previous)['identity']),'Previous experiment is still running'
    with (ROOT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    record=dict(identity=identity(child.pid),at_utc=utc(),models=MODELS,lengths=LENGTHS,seed=2033,batch_size=256,
        optimizer_training=False,max_parallel_gpu_workers=6,other_paper_experiments='stopped')
    save(p,record);print(json.dumps(record))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.launch:launch()
    else:print(json.dumps(dict(launch_required=True)))
