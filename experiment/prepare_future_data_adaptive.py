"""Adapt CPU simulation concurrency to active GPU jobs; preserve frozen physics."""
from pathlib import Path
import argparse,os,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import prepare_future_data as original
from experiment_common import ROOT,load,save,sha,identity,alive,utc,cpu_quota

def target_workers():
    statuses=[load(ROOT/'status.json')]
    current=ROOT/'CURRENT_LOCALIZER_RETEST.json'
    if current.exists():
        directory=(ROOT/load(current)['directory']).resolve()
        assert directory.is_relative_to(ROOT.resolve())
        if (directory/'status.json').exists():statuses.append(load(directory/'status.json'))
    identities=set()
    for status in statuses:
        for item in status.get('active',{}).values():
            if item['kind'] in ['formal_fit','search_fit','convergence','parameter_fit','parameter_search_fit','evaluate','parameter_evaluate'] and alive(item['identity']):
                identities.add((item['identity']['pid'],item['identity']['start_ticks']))
    gpu=len(identities)
    limit=int(load(ROOT/'parallel_resources.json').get('auxiliary_cpu_workers',14))
    return max(1,min(14,limit,cpu_quota()-gpu))

def generate(case,spec,positions,selected):
    tasks=[];existing=0
    for split in ['valid','train','test']:
        indices=list(range(len(positions[split])))
        if split=='train':
            chosen=set(selected.tolist());indices=selected.tolist()+[i for i in indices if i not in chosen]
        for index in indices:
            if original.checked_position(original.position_path(case,split,index),spec):existing+=1
            else:tasks.append((case,split,index,*map(float,positions[split][index])))
    total=sum(map(len,positions.values()));completed=existing;began=time.monotonic()
    with ProcessPoolExecutor(max_workers=14) as pool:
        iterator=iter(tasks);active={}
        def fill():
            workers=target_workers()
            while len(active)<workers:
                original.disk_guard()
                try:task=next(iterator)
                except StopIteration:break
                active[pool.submit(original.simulate_position,task)]=task[1:3]
        fill()
        while active:
            ready,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in ready:
                split,index,x,y,data=future.result();original.disk_guard()
                path=original.position_path(case,split,index)
                original.npz_write(path,signature=original.np.asarray(spec['signature']),x=x,y=y,
                    repetitions=original.np.arange(original.REPEATS[split]),**data)
                save(path.with_suffix('.json'),dict(signature=spec['signature'],sha256=sha(path)))
                del active[future];completed+=1
            fill()
            original.emit(phase='generating',case=case['name'],positions_completed=completed,
                positions_total=total,cpu_workers=target_workers(),active_position_tasks=len(active),
                adaptive_cpu_concurrency=True,elapsed_seconds=time.monotonic()-began)

def run():
    original.generate=generate;original.run()

def launch():
    out=original.OUT;record=out/'launch.json'
    if record.exists() and alive(load(record)['identity']):print(__import__('json').dumps(load(record)));return
    assert load(out/'preflight.json')['passed']
    assert load(out/'plan.json')['generator_sha256']==sha(ROOT/'prepare_future_data.py')
    with (out/'generator.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),at_utc=utc(),scope='CPU dataset preparation',
        scheduling_adapter=str(Path(__file__)),scheduling_adapter_sha256=sha(Path(__file__)),
        maximum_cpu_workers=14,adaptive_cpu_concurrency=True,original_generator_unchanged=True)
    save(record,result);print(__import__('json').dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.launch:launch()
    else:parser.print_help()
