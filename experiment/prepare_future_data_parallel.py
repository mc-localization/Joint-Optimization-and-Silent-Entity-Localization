"""Use the remaining CPU core while preserving the frozen data generator.

Only process scheduling changes. Simulation, seeds, signatures, committed
position files and dataset materialization use prepare_future_data unchanged.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import prepare_future_data as original
from experiment_common import ROOT,load,save,sha,identity,alive,utc,cpu_quota

def generate(case,spec,positions,selected):
    tasks=[];existing=0
    for split in ['valid','train','test']:
        indices=list(range(len(positions[split])))
        if split=='train':
            chosen=set(selected.tolist());indices=selected.tolist()+[i for i in indices if i not in chosen]
        for index in indices:
            if original.checked_position(original.position_path(case,split,index),spec):existing+=1
            else:tasks.append((case,split,index,*map(float,positions[split][index])))
    state=load(ROOT/'status.json')
    train_workers=sum(v['kind']=='formal_fit' for v in state.get('active',{}).values())
    limit=load(ROOT/'parallel_resources.json')['auxiliary_cpu_workers']
    # Parent/controller CPU use is small; no full core needs to be reserved.
    workers=max(1,min(limit,cpu_quota()-train_workers))
    total=sum(map(len,positions.values()));completed=existing;began=time.monotonic()
    original.emit(phase='generating',case=case['name'],positions_completed=completed,
        positions_total=total,cpu_workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator=iter(tasks);active={}
        def fill():
            while len(active)<2*workers:
                original.disk_guard()
                try:task=next(iterator)
                except StopIteration:break
                active[pool.submit(original.simulate_position,task)]=task[1:3]
        fill()
        while active:
            ready,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for f in ready:
                split,index,x,y,data=f.result();original.disk_guard()
                path=original.position_path(case,split,index)
                original.npz_write(path,signature=original.np.asarray(spec['signature']),x=x,y=y,
                    repetitions=original.np.arange(original.REPEATS[split]),**data)
                save(path.with_suffix('.json'),dict(signature=spec['signature'],sha256=sha(path)))
                del active[f];completed+=1
            original.emit(phase='generating',case=case['name'],positions_completed=completed,
                positions_total=total,cpu_workers=workers,elapsed_seconds=time.monotonic()-began)
            fill()

def run():
    original.generate=generate
    original.run()

def launch():
    out=original.OUT;record=out/'launch.json'
    if record.exists() and alive(load(record)['identity']):print(json.dumps(load(record)));return
    assert load(out/'preflight.json')['passed']
    assert load(out/'plan.json')['generator_sha256']==sha(ROOT/'prepare_future_data.py')
    with (out/'generator.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),at_utc=utc(),scope='CPU dataset preparation only',
        scheduling_adapter=str(Path(__file__)),scheduling_adapter_sha256=sha(Path(__file__)),
        maximum_cpu_workers=load(ROOT/'parallel_resources.json')['auxiliary_cpu_workers'],
        original_generator_unchanged=True)
    save(record,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.launch:launch()
    else:parser.print_help()
