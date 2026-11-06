"""CPU-only, resumable datasets for the existing paper's parameter studies.

All coordinates, realizations and random seeds follow the new Rx20 baseline.
The physical two-receiver experiment removes two absorbing receivers; it does
not select two channels from a four-receiver simulation.
"""
from pathlib import Path
import sys,os,json,time,hashlib,shutil,argparse,subprocess,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import generate_4rx_dataset as physics
from experiment_common import load,save,sha,utc,identity,alive,cpu_quota
OUT=ROOT/'future_datasets'
REPEATS={'train':60,'valid':15,'test':1}
SEEDS={'train':2025,'valid':502025,'test':1002025}
CASES=[
    dict(name='molecules_10000',molecules=10000,n_rx=4,features=[250]),
    dict(name='molecules_30000',molecules=30000,n_rx=4,features=[250]),
    dict(name='sampling_reference',molecules=20000,n_rx=4,features=[125,375,500]),
    dict(name='physical_2rx',molecules=20000,n_rx=2,features=[125,250]),
]

def sample_indices(m):return np.rint(np.linspace(0,999,m)).astype(np.int64)

def emit(**kw):
    value=dict(at_utc=utc(),pid=os.getpid(),**kw)
    save(OUT/'status.json',value);print(json.dumps(value),flush=True)

def disk_guard():
    while shutil.disk_usage(OUT).free<4*2**30:
        emit(phase='waiting_for_disk',free_gib=shutil.disk_usage(OUT).free/2**30)
        time.sleep(15)

def npz_write(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)

def position_path(case,split,index):
    return OUT/case['name']/'positions'/split/f'p{index:05d}.npz'

def checked_position(path,spec):
    if not path.exists():return False
    record_path=path.with_suffix('.json')
    if record_path.exists():
        record=load(record_path)
        assert record['signature']==spec['signature'] and record['sha256']==sha(path),str(path)
    else:
        # A crash between the NPZ and checksum commits can be recovered safely.
        with np.load(path,allow_pickle=False) as z:
            assert str(z['signature'])==spec['signature']
            assert all(f'X{m}' in z for m in spec['features'])
        save(record_path,dict(signature=spec['signature'],sha256=sha(path)))
    return True

def simulate_position(task):
    case,split,index,x,y=task
    original=physics.RX_LIST
    physics.RX_LIST=original[:case['n_rx']]
    try:
        full_features=1000 if len(case['features'])>1 else case['features'][0]
        values=np.stack([physics.simulate(x,y,SEEDS[split]+index*100003+rep,
            case['molecules'],1000,.005,79.4,4.,full_features)
            for rep in range(REPEATS[split])]).reshape(REPEATS[split],case['n_rx'],full_features)
    finally:physics.RX_LIST=original
    assert np.all(np.diff(values.astype(np.int32),axis=2)>=0)
    assert np.all(values[:,:,-1].sum(axis=1)<=case['molecules'])
    result={f'X{m}':(values[:,:,sample_indices(m)] if full_features==1000 else values).reshape(len(values),-1)
        for m in case['features']}
    if case['name']=='sampling_reference':
        baseline=ROOT/'position_signals'/split/f'p{index:05d}.npz'
        with np.load(baseline,allow_pickle=False) as z:
            assert np.array_equal(values[:,:,sample_indices(250)].reshape(len(values),-1),z['X']),str(baseline)
            assert float(z['x'])==x and float(z['y'])==y
    return split,index,x,y,result

def make_spec(case,protocol):
    spec=dict(case,receivers=physics.RX_LIST[:case['n_rx']],tx=[0,0],eve_radius=4,
        dt=.005,diffusion=79.4,steps=1000,grid_step=.4,grid_origin=[0,0],
        coordinate_design='Exactly the Rx20 four-receiver baseline positions, spatial split and row order',
        position_plan_sha256=sha(ROOT/'position_plan.npz'),repetitions=REPEATS,seeds=SEEDS,
        row_seed='split_seed + position_index*100003 + repetition',
        sampling_rule='rint(linspace(0,999,M)); actual time=(index+1)*0.005 seconds',
        source_indices={str(m):sample_indices(m).tolist() for m in case['features']},
        simulator_sha256=sha(ROOT/'runtime/generate_4rx_dataset.py'),
        generator_sha256=sha(Path(__file__)),baseline_generation_signature=protocol['dataset_spec']['generation_signature'],
        candidate_position_plan_sha256=sha(ROOT/'candidate_position_plan.json'),
        physical_receiver_count=case['n_rx'],training_started=False)
    spec['signature']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return spec

def generate(case,spec,positions,selected):
    tasks=[];existing=0
    for split in ['valid','train','test']:
        indices=list(range(len(positions[split])))
        if split=='train':
            chosen=set(selected.tolist());indices=selected.tolist()+[i for i in indices if i not in chosen]
        for index in indices:
            if checked_position(position_path(case,split,index),spec):existing+=1
            else:tasks.append((case,split,index,*map(float,positions[split][index])))
    state=load(ROOT/'status.json')
    train_workers=sum(v['kind']=='formal_fit' for v in state.get('active',{}).values())
    limit=load(ROOT/'parallel_resources.json')['auxiliary_cpu_workers']
    workers=max(1,min(limit,cpu_quota()-train_workers-1))
    total=sum(map(len,positions.values()));completed=existing;began=time.monotonic()
    emit(phase='generating',case=case['name'],positions_completed=completed,positions_total=total,cpu_workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator=iter(tasks);active={}
        def fill():
            while len(active)<2*workers:
                disk_guard()
                try:task=next(iterator)
                except StopIteration:break
                active[pool.submit(simulate_position,task)]=task[1:3]
        fill()
        while active:
            ready,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for f in ready:
                split,index,x,y,data=f.result();disk_guard()
                path=position_path(case,split,index)
                npz_write(path,signature=np.asarray(spec['signature']),x=x,y=y,
                    repetitions=np.arange(REPEATS[split]),**data)
                save(path.with_suffix('.json'),dict(signature=spec['signature'],sha256=sha(path)))
                del active[f];completed+=1
            emit(phase='generating',case=case['name'],positions_completed=completed,
                positions_total=total,cpu_workers=workers,elapsed_seconds=time.monotonic()-began)
            fill()

def materialize(case,spec,positions):
    outputs={}
    for m in case['features']:
        directory=OUT/case['name']/f'M{m}';checks={};counts={}
        for split,n in REPEATS.items():
            per=max(1,1000//n);count=0
            for shard,left in enumerate(range(0,len(positions[split]),per)):
                disk_guard();target=directory/f'data_{split}'/f'data_{split}_{shard:03d}.npz'
                rec=target.with_suffix('.json')
                if target.exists() and rec.exists():
                    v=load(rec);assert v['signature']==spec['signature'] and sha(target)==v['sha256']
                    checks[str(target.relative_to(directory))]=v['sha256'];count+=v['rows'];continue
                values=[];xy=[]
                for index in range(left,min(left+per,len(positions[split]))):
                    path=position_path(case,split,index);assert checked_position(path,spec)
                    with np.load(path,allow_pickle=False) as z:
                        assert np.array_equal(z['repetitions'],np.arange(n))
                        assert np.allclose([float(z['x']),float(z['y'])],positions[split][index],rtol=0,atol=0)
                        values.append(z[f'X{m}']);xy.extend([positions[split][index]]*n)
                xy=np.asarray(xy);npz_write(target,X=np.concatenate(values),x=xy[:,0],y=xy[:,1])
                digest=sha(target);save(rec,dict(signature=spec['signature'],sha256=digest,rows=len(xy)))
                checks[str(target.relative_to(directory))]=digest;count+=len(xy)
                if shard%50==0:emit(phase='assembling',case=case['name'],M=m,split=split,shard=shard)
            counts[split]=count
        assert counts==dict(train=410640,valid=25500,test=2000)
        test_parts=[]
        for p in sorted((directory/'data_test').glob('*.npz')):
            with np.load(p,allow_pickle=False) as z:test_parts.append({k:z[k].copy() for k in ['X','x','y']})
        npz_write(directory/'independent_test.npz',**{k:np.concatenate([v[k] for v in test_parts]) for k in ['X','x','y']})
        # Shared coordinate/repetition row plan, with disjoint microtrain/microvalidation.
        rowplan=load(ROOT/'candidate_position_plan.json');micro_hashes={}
        for label,key in [('train','training_row_indices'),('valid','validation_row_indices')]:
            ids=np.asarray(rowplan[key]);xs=[];xy=[]
            for index in np.unique(ids//60):
                reps=ids[ids//60==index]%60
                with np.load(position_path(case,'train',int(index)),allow_pickle=False) as z:xs.append(z[f'X{m}'][reps])
                xy.extend([positions['train'][index]]*len(reps))
            xy=np.asarray(xy);p=directory/'micro_dataset'/f'micro_{label}.npz'
            npz_write(p,X=np.concatenate(xs),x=xy[:,0],y=xy[:,1]);micro_hashes[p.name]=sha(p)
        meta=dict(spec,features=m,n_rx=case['n_rx'],source_sample_indices=sample_indices(m).tolist(),
            train_rows=counts['train'],valid_rows=counts['valid'],test_rows=counts['test'],
            test_sha256=sha(directory/'independent_test.npz'),shards_sha256=checks,micro_sha256=micro_hashes,
            generation_status='complete',ready_at_utc=utc())
        save(directory/'dataset_metadata.json',meta)
        save(directory/'readiness.json',dict(passed=True,signature=spec['signature'],features=m,counts=counts,
            metadata_sha256=sha(directory/'dataset_metadata.json')))
        outputs[str(m)]=str(directory)
    save(OUT/case['name']/'complete.json',dict(at_utc=utc(),signature=spec['signature'],outputs=outputs))

def run():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'.generator.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=load(OUT/'plan.json');assert plan['generator_sha256']==sha(Path(__file__))
            assert plan['position_plan_sha256']==sha(ROOT/'position_plan.npz')
            assert plan['simulator_sha256']==sha(ROOT/'runtime/generate_4rx_dataset.py')
            with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:
                positions={s:z[k].copy() for s,k in [('train','train_positions'),('valid','validation_positions'),('test','test_positions')]}
                selected=z['candidate_training_position_indices'].copy()
            for case in CASES:
                spec=plan['cases'][case['name']];folder=OUT/case['name'];folder.mkdir(exist_ok=True)
                if (folder/'complete.json').exists():
                    assert load(folder/'complete.json')['signature']==spec['signature'];continue
                save(folder/'generation_spec.json',spec)
                generate(case,spec,positions,selected);materialize(case,spec,positions)
            emit(phase='complete',cases=[c['name'] for c in CASES])
        except Exception:
            emit(phase='failed',traceback=traceback.format_exc());raise

def preflight():
    OUT.mkdir(exist_ok=True);protocol=load(ROOT/'deployment_protocol.json')
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    original=physics.RX_LIST.copy();checks=[]
    with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:x,y=z['train_positions'][0]
    full=physics.simulate(x,y,2025,20000,1000,.005,79.4,4.,1000).reshape(4,1000)
    with np.load(ROOT/'position_signals/train/p00000.npz',allow_pickle=False) as z:
        assert np.array_equal(full[:,sample_indices(250)].reshape(-1),z['X'][0])
    for case in CASES:
        physics.RX_LIST=original[:case['n_rx']]
        try:
            raw=physics.simulate(x,y,2025,case['molecules'],1000,.005,79.4,4.,1000).reshape(case['n_rx'],1000)
            for m in case['features']:
                direct=physics.simulate(x,y,2025,case['molecules'],1000,.005,79.4,4.,m)
                assert np.array_equal(raw[:,sample_indices(m)].reshape(-1),direct)
        finally:physics.RX_LIST=original
        checks.append(dict(case=case['name'],sampling_equivalence=True,physical_receivers=case['n_rx']))
    plan=dict(created_at_utc=utc(),generator_sha256=sha(Path(__file__)),
        position_plan_sha256=sha(ROOT/'position_plan.npz'),simulator_sha256=sha(ROOT/'runtime/generate_4rx_dataset.py'),
        baseline_signature=protocol['dataset_spec']['generation_signature'],
        cases={c['name']:make_spec(c,protocol) for c in CASES},scope='Dataset preparation only; no additional model training')
    if (OUT/'plan.json').exists():
        previous=load(OUT/'plan.json');plan['created_at_utc']=previous['created_at_utc'];assert plan==previous
    else:save(OUT/'plan.json',plan)
    save(OUT/'preflight.json',dict(passed=True,at_utc=utc(),baseline_sample_equivalence=True,checks=checks))
    print(json.dumps(dict(passed=True,cases=checks)),flush=True)

def launch():
    OUT.mkdir(exist_ok=True);record=OUT/'launch.json'
    if record.exists() and alive(load(record)['identity']):print(json.dumps(load(record)));return
    assert load(OUT/'preflight.json')['passed']
    assert load(OUT/'plan.json')['generator_sha256']==sha(Path(__file__))
    with (OUT/'generator.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),at_utc=utc(),scope='CPU dataset preparation only')
    save(record,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.launch:launch()
    elif args.run:run()
    else:parser.print_help()
