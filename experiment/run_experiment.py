"""Detached, resumable scheduler for CNN/DNN start searches and CNN controls."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys,time,traceback
from experiment_common import ROOT,load,save,sha,utc,identity,alive,mark_done,read_csv,write_csv,cpu_quota
sys.dont_write_bytecode=True
COARSE_LENGTHS=list(range(25,226,25))

def static_jobs(protocol):
    jobs={}
    def add(key,kind,deps=(),resource='gpu',priority=10,gpu_gib=4.,memory_gib=7.,**kw):
        jobs.setdefault(key,dict(key=key,kind=kind,deps=list(deps),resource=resource,
            priority=priority,gpu_gib=gpu_gib,memory_gib=memory_gib,**kw))
    add('data_generate','data',resource='cpu',priority=0,gpu_gib=0,memory_gib=3)
    for model in ['cnn','dnn']:
        add('policy_'+model,'policy',['micro_ready'],resource='control',priority=0,gpu_gib=.5,memory_gib=1,model=model)
    lengths=protocol['cases']['cnn']['final_cfg']['final']['lengths']
    for length in lengths:
        fit=f'fit_cnn_l{length}_s1'
        add(fit,'formal_fit',['data_generate'],priority=8,model='cnn',length=length,start=1)
        add(f'eval_cnn_fixed_l{length}','evaluate',[fit,'policy_cnn'],priority=12,
            gpu_gib=2,memory_gib=2,model='cnn',length=length,role='fixed')
    for length in COARSE_LENGTHS:
        starts=sorted(set(range(1,252-length,5))|{251-length})
        for start in starts:
            add(f'fit_cnn_l{length}_s{start}','formal_fit',['data_generate'],priority=9,model='cnn',length=length,start=start)
    return jobs

def reconcile(protocol,jobs):
    for path in (ROOT/'requests').glob('*.json'):
        job=load(path)
        assert job['kind']=='search_fit' and job['model'] in ['cnn','dnn']
        assert 1<=job['length']<=249 and 1<=job['start']<=251-job['length']
        assert job['key']==f"search_{job['model']}_l{job['length']}_s{job['start']}"
        jobs.setdefault(job['key'],job)
    for model in ['cnn','dnn']:
        if not (ROOT/'done'/f'policy_{model}.json').exists():continue
        frozen=load(ROOT/'search'/model/'frozen_optimizer.json')
        assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
        for length in protocol['cases'][model]['final_cfg']['final']['lengths']:
            start=int(frozen['starts'][str(length)]);fit=f'fit_{model}_l{length}_s{start}'
            jobs.setdefault(fit,dict(key=fit,kind='formal_fit',deps=['data_generate','policy_'+model],
                resource='gpu',priority=7,gpu_gib=4.,memory_gib=7.,model=model,length=length,start=start))
            key=f'eval_{model}_optimized_l{length}'
            jobs.setdefault(key,dict(key=key,kind='evaluate',deps=[fit,'policy_'+model],
                resource='gpu',priority=12,gpu_gib=2.,memory_gib=2.,model=model,length=length,role='optimized'))
        if model=='cnn':
            for length in COARSE_LENGTHS:
                starts=sorted(set(range(1,252-length,5))|{251-length,int(frozen['starts'][str(length)])})
                key=f'coarse_select_l{length}'
                jobs.setdefault(key,dict(key=key,kind='coarse_select',deps=[f'fit_cnn_l{length}_s{s}' for s in starts]+['policy_cnn'],
                    resource='cpu_light',priority=10,gpu_gib=0,memory_gib=.5,length=length))
                key2=f'eval_cnn_coarse_l{length}'
                jobs.setdefault(key2,dict(key=key2,kind='evaluate',deps=[key],resource='gpu',priority=12,
                    gpu_gib=2.,memory_gib=2.,model='cnn',length=length,role='coarse'))

def check_sources(protocol):
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name

def collect():
    rows=[]
    for p in sorted((ROOT/'results').glob('*_*/l*/final_test.csv')):rows+=read_csv(p)
    if rows:write_csv(ROOT/'all_test_results.csv',rows)

def gpu_free():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    return int(result.stdout.strip().splitlines()[0])/1024 if result.returncode==0 else 0

def run():
    import fcntl
    protocol=load(ROOT/'deployment_protocol.json');check_sources(protocol)
    with (ROOT/'.controller.lock').open('a+b') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=static_jobs(protocol);active={};failures={}
        attempts=load(ROOT/'attempts.json') if (ROOT/'attempts.json').exists() else {}
        if (ROOT/'active.json').exists():
            for key,item in load(ROOT/'active.json').items():
                if alive(item['identity']):
                    active[key]=dict(job=item['job'],identity=item['identity'],process=None,log=None)
                    jobs.setdefault(key,item['job'])
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        last_collect=0
        while True:
            reconcile(protocol,jobs)
            for key,item in list(active.items()):
                if item['process'] is not None:
                    rc=item['process'].poll()
                    if rc is None:continue
                    item['log'].close()
                elif alive(item['identity']):continue
                else:rc=None
                del active[key]
                if not (ROOT/'done'/f'{key}.json').exists():
                    error=dict(exit_code=rc,log=str(ROOT/'logs'/f'{key}.log'),attempts=attempts.get(key,1),at_utc=utc())
                    if attempts.get(key,1)>=2:
                        failures[key]=error;save(ROOT/'failed'/f'{key}.json',error)
                    else:save(ROOT/'retry'/f'{key}.json',error)
            completed={p.stem for p in (ROOT/'done').glob('*.json')}
            for p in (ROOT/'failed').glob('*.json'):failures[p.stem]=load(p)
            ready=sorted([j for k,j in jobs.items() if k not in completed and k not in active and k not in failures and set(j['deps'])<=completed],key=lambda j:(j['priority'],j['key']))
            disk_free=shutil.disk_usage(ROOT).free/2**30
            for job in ready:
                current=[a['job'] for a in active.values()]
                kind=job['resource']
                if disk_free<3:break
                if kind=='cpu' and any(a['resource']=='cpu' for a in current):continue
                if kind=='control' and sum(a['resource']=='control' for a in current)>=2:continue
                if kind=='cpu_light' and sum(a['resource']=='cpu_light' for a in current)>=1:continue
                if kind=='gpu':
                    if sum(a['resource']=='gpu' for a in current)>=6:continue
                    if job['kind']=='formal_fit' and sum(a['kind']=='formal_fit' for a in current)>=3:continue
                    if (ROOT/'done/data_generate.json').exists() and job['kind']=='search_fit' and sum(a['kind']=='search_fit' for a in current)>=4:continue
                    if sum(a['gpu_gib'] for a in current)+job['gpu_gib']>21.5:continue
                    if gpu_free()<job['gpu_gib']+1:continue
                if sum(a['memory_gib'] for a in current)+job['memory_gib']>60:continue
                anonymous=int(dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())['anon'])/2**30
                if anonymous+job['memory_gib']>80:continue
                key=job['key'];save(ROOT/'jobs'/f'{key}.json',job)
                (ROOT/'logs').mkdir(exist_ok=True)
                log=(ROOT/'logs'/f'{key}.log').open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'run_experiment.py'),'--worker',key],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                active[key]=dict(job=job,identity=identity(process.pid),process=process,log=log)
                attempts[key]=attempts.get(key,0)+1;save(ROOT/'attempts.json',attempts)
                save(ROOT/'active.json',{k:dict(job=v['job'],identity=v['identity']) for k,v in active.items()})
                time.sleep(.5)
            state=dict(phase='running' if active else ('needs_attention' if failures else 'waiting'),
                pid=os.getpid(),at_utc=utc(),known_jobs=len(jobs),completed_jobs=len(set(jobs)&completed),
                candidate_requests_discovered=sum(j['kind']=='search_fit' for j in jobs.values()),
                jobs_pending=len(set(jobs)-completed-set(active)),failed=failures,
                active={k:dict(identity=v['identity'],kind=v['job']['kind'],gpu_reserve_gib=v['job']['gpu_gib']) for k,v in active.items()},
                disk_free_gib=disk_free,micro_ready='micro_ready' in completed,full_data_ready='data_generate' in completed,
                max_parallel_gpu_fits=6,max_parallel_full_fits=3,scope=protocol['scope'])
            save(ROOT/'status.json',state)
            save(ROOT/'active.json',{k:dict(job=v['job'],identity=v['identity']) for k,v in active.items()})
            save(ROOT/'queue.json',list(jobs.values()))
            if time.monotonic()-last_collect>30:collect();last_collect=time.monotonic()
            if not active and set(jobs)<=completed:
                collect();save(ROOT/'status.json',dict(state,phase='complete',completed_at_utc=utc()));break
            if not active and failures and not ready:
                raise RuntimeError('Failed jobs and blocked dependencies; inspect status.json and logs.')
            time.sleep(3)

def launch():
    protocol=load(ROOT/'deployment_protocol.json');check_sources(protocol)
    target=ROOT/'launch.json'
    if target.exists() and alive(load(target)['identity']):print(json.dumps(load(target)));return
    with (ROOT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'run_experiment.py'),'--run'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),launched_at_utc=utc(),scope=protocol['scope'],resources=protocol['resources'])
    save(target,result);print(json.dumps(result,ensure_ascii=False),flush=True)

def worker(key):
    job=load(ROOT/'jobs'/f'{key}.json');check_sources(load(ROOT/'deployment_protocol.json'))
    if job['kind']=='data':
        import priority_data;priority_data.run()
    else:
        import experiment_workers;experiment_workers.run(job)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.run:run()
    elif args.launch:launch()
    else:print(json.dumps(dict(protocol=str(ROOT/'deployment_protocol.json'),launch_required=True)))
