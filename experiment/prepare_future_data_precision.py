"""Reproduce baseline scalar arithmetic in sampling-reference realizations."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import prepare_future_data as original
import prepare_future_data_adaptive as scheduling
from experiment_common import ROOT,load,save,sha,identity,alive,utc

NATIVE_SIMULATE=original.simulate_position
PRIORITY_REPETITIONS=None

def simulate_reference(task):
    case,split,index,x,y=task
    if case['name']!='sampling_reference':return NATIVE_SIMULATE(task)
    global PRIORITY_REPETITIONS
    if PRIORITY_REPETITIONS is None:
        with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:
            ids=np.r_[z['micro_training_row_indices'],z['micro_validation_row_indices']]
        PRIORITY_REPETITIONS={int(i):set((ids[ids//60==i]%60).tolist()) for i in np.unique(ids//60)}
    priority=PRIORITY_REPETITIONS.get(index,set()) if split=='train' else set()
    values=[]
    for rep in range(original.REPEATS[split]):
        # priority_data passed NumPy scalars for its initial micro realizations,
        # then Python scalars for the remaining rows. NumPy 2 treats these
        # differently when subtracting from the float32 particle coordinates.
        xx,yy=(np.float64(x),np.float64(y)) if rep in priority else (float(x),float(y))
        values.append(original.physics.simulate(xx,yy,original.SEEDS[split]+index*100003+rep,
            case['molecules'],1000,.005,79.4,4.,1000))
    values=np.stack(values).reshape(original.REPEATS[split],case['n_rx'],1000)
    assert np.all(np.diff(values.astype(np.int32),axis=2)>=0)
    assert np.all(values[:,:,-1].sum(axis=1)<=case['molecules'])
    baseline=ROOT/'position_signals'/split/f'p{index:05d}.npz'
    record=load(baseline.with_suffix('.json'));assert sha(baseline)==record['sha256']
    with np.load(baseline,allow_pickle=False) as z:
        assert np.array_equal(values[:,:,original.sample_indices(250)].reshape(len(values),-1),z['X']),str(baseline)
        assert float(z['x'])==x and float(z['y'])==y
        assert np.array_equal(z['repetitions'],np.arange(original.REPEATS[split]))
    result={f'X{M}':values[:,:,original.sample_indices(M)].reshape(len(values),-1) for M in case['features']}
    return split,index,x,y,result

def preflight():
    plan=load(original.OUT/'plan.json')
    assert plan['generator_sha256']==sha(ROOT/'prepare_future_data.py')
    case=next(c for c in original.CASES if c['name']=='sampling_reference')
    spec=plan['cases'][case['name']]
    # Recheck the failed position and all train positions committed before the
    # failure. Validation/test coordinates always followed the Python path.
    indices=sorted({656}|{int(p.stem[1:]) for p in (original.OUT/'sampling_reference/positions/train').glob('p*.npz')})
    with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:positions=z['train_positions'].copy()
    tasks=[(case,'train',i,*map(float,positions[i])) for i in indices]
    audit=[]
    with ProcessPoolExecutor(max_workers=min(4,len(tasks))) as pool:
        for split,index,x,y,data in pool.map(simulate_reference,tasks):
            path=original.position_path(case,split,index);existed=path.exists();same=False
            if existed:
                assert original.checked_position(path,spec)
                with np.load(path,allow_pickle=False) as z:same=all(np.array_equal(z[k],a) for k,a in data.items())
            if existed and not same:
                recovery=original.OUT/'recovery/sampling_precision_20261028'
                recovery.mkdir(parents=True,exist_ok=True)
                for previous in [path,path.with_suffix('.json')]:
                    target=recovery/previous.name;assert not target.exists();previous.replace(target)
            if not same:
                original.npz_write(path,signature=np.asarray(spec['signature']),x=x,y=y,
                    repetitions=np.arange(original.REPEATS[split]),**data)
                save(path.with_suffix('.json'),dict(signature=spec['signature'],sha256=sha(path)))
            audit.append(dict(index=index,existing=existed,previous_all_densities_equal=same,
                baseline_250_exact=True,committed_sha256=sha(path)))
    result=dict(passed=True,at_utc=utc(),adapter_sha256=sha(Path(__file__)),
        original_generator_sha256=sha(ROOT/'prepare_future_data.py'),plan_sha256=sha(original.OUT/'plan.json'),
        diagnostic=load(ROOT/'sampling_precision_full_diagnosis.json'),
        scalar_type_rule='NumPy float64 for original priority micro repetitions; Python float for all other realizations',
        sampling_projection_assertion_preserved=True,audited_train_positions=audit)
    save(original.OUT/'precision_recovery_preflight.json',result);print(json.dumps(result),flush=True)

def run():
    original.simulate_position=simulate_reference
    original.generate=scheduling.generate
    original.run()

def launch():
    out=original.OUT;record=out/'launch.json'
    if record.exists() and alive(load(record)['identity']):print(json.dumps(load(record)));return
    check=load(out/'precision_recovery_preflight.json')
    assert check['passed'] and check['adapter_sha256']==sha(Path(__file__))
    assert check['plan_sha256']==sha(out/'plan.json')
    with (out/'generator.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),at_utc=utc(),scope='CPU dataset preparation',
        scheduling_adapter=str(Path(__file__)),scheduling_adapter_sha256=sha(Path(__file__)),
        maximum_cpu_workers=14,adaptive_cpu_concurrency=True,baseline_scalar_precision_reproduced=True,
        original_generator_unchanged=True)
    save(record,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.run:run()
    elif args.launch:launch()
    else:parser.print_help()
