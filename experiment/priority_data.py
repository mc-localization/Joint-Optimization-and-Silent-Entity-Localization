"""Generate micro realizations first, then complete exactly the same full dataset."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib,json,os,time
import numpy as np
import generate_4rx_dataset as physics
from experiment_common import ROOT,load,save,sha,report,mark_done,cpu_quota

KEY='data_generate'

def position_path(split,index):return ROOT/'position_signals'/split/f'p{index:05d}.npz'

def existing(split,index,signature):
    path=position_path(split,index)
    if not path.exists():return None
    rec=load(path.with_suffix('.json'))
    assert rec['generation_signature']==signature and rec['sha256']==sha(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def simulate_task(task):
    split,index,x,y,repetitions,seed_base,simulation=task
    values=[]
    for repetition in repetitions:
        seed=seed_base+index*100003+repetition
        values.append(physics.simulate(x,y,seed,simulation['molecules'],simulation['steps'],
            simulation['dt'],simulation['diffusion'],physics.R_E,simulation['features']))
    return split,index,x,y,np.asarray(repetitions,dtype=np.int64),np.stack(values)

def commit(result,signature):
    split,index,x,y,ids,signals=result
    old=existing(split,index,signature)
    if old is not None:
        assert not np.intersect1d(ids,old['repetitions']).size
        ids=np.concatenate((old['repetitions'],ids));signals=np.concatenate((old['X'],signals))
    order=np.argsort(ids);ids,signals=ids[order],signals[order]
    assert len(np.unique(ids))==len(ids) and signals.shape==(len(ids),1000)
    count=signals.reshape(-1,4,250).astype(np.int32)
    assert (np.diff(count,axis=2)>=0).all() and (count[:,:,-1].sum(axis=1)<=20000).all()
    path=position_path(split,index);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,X=signals,repetitions=ids,x=float(x),y=float(y))
    tmp.replace(path)
    save(path.with_suffix('.json'),dict(sha256=sha(path),generation_signature=signature,realizations=len(ids)))

def stage(tasks,workers,signature,name):
    total=len(tasks);began=time.monotonic();completed=0
    if not total:return
    iterator=iter(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={}
        def fill():
            while len(active)<2*workers:
                try:task=next(iterator)
                except StopIteration:break
                active[pool.submit(simulate_task,task)]=task[:2]
        fill()
        while active:
            done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                commit(future.result(),signature);del active[future];completed+=1
            report(KEY,name,positions_completed=completed,positions_total=total,
                cpu_workers=workers,elapsed_seconds=time.monotonic()-began)
            fill()

def write_micro(plan,signature):
    rowplan=load(ROOT/'candidate_position_plan.json')
    output=ROOT/'micro_dataset';output.mkdir(exist_ok=True)
    for label,key in [('train','training_row_indices'),('valid','validation_row_indices')]:
        global_ids=np.asarray(rowplan[key]);chunks=[];coords=[]
        for index in np.unique(global_ids//60):
            wanted=global_ids[global_ids//60==index]%60
            data=existing('train',int(index),signature)
            slots=np.searchsorted(data['repetitions'],wanted)
            assert np.array_equal(data['repetitions'][slots],wanted)
            chunks.append(data['X'][slots]);coords.extend([(float(data['x']),float(data['y']))]*len(slots))
        xy=np.asarray(coords);target=output/f'micro_{label}.npz'
        with target.with_name(target.name+'.tmp').open('wb') as f:
            np.savez_compressed(f,X=np.concatenate(chunks),x=xy[:,0],y=xy[:,1])
        target.with_name(target.name+'.tmp').replace(target)
    count=rowplan['candidate_positions']
    save(output/'sampling_manifest.json',dict(generation_signature=signature,
        candidate_position_plan_sha256=sha(ROOT/'candidate_position_plan.json'),positions=count,
        actual_samples=count*30,validation_samples=count*8,train_validation_sample_overlap=0,
        fraction=rowplan['fraction'],fraction_denominator='all feasible grid positions',
        artifacts_sha256={n:sha(output/n) for n in ['micro_train.npz','micro_valid.npz']}))
    mark_done('micro_ready',sampling_manifest_sha256=sha(output/'sampling_manifest.json'))
    report(KEY,'micro_ready_full_generation_continues',micro_train=count*30,micro_validation=count*8)

def write_dataset(meta,positions):
    root=ROOT/'data';root.mkdir(exist_ok=True);signature=meta['generation_signature']
    save(root/'generation_manifest.json',meta)
    for split,n in [('train',60),('valid',15),('test',1)]:
        directory=root/('data_'+split);directory.mkdir(exist_ok=True)
        per_shard=max(1,1000//n)
        for shard,left in enumerate(range(0,len(positions[split]),per_shard)):
            target=directory/f'data_{split}_{shard:03d}.npz'
            lock=target.with_suffix('.npz.json')
            if target.exists() and lock.exists():
                r=load(lock);assert r['generation_signature']==signature and r['sha256']==sha(target);continue
            values=[];xy=[]
            for index in range(left,min(left+per_shard,len(positions[split]))):
                d=existing(split,index,signature);assert np.array_equal(d['repetitions'],np.arange(n))
                values.append(d['X']);xy.extend([(float(d['x']),float(d['y']))]*n)
            xy=np.asarray(xy);temp=target.with_name(target.name+'.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,X=np.concatenate(values),x=xy[:,0],y=xy[:,1])
            temp.replace(target);save(lock,dict(generation_signature=signature,sha256=sha(target),rows=len(xy)))
            if shard%20==0:report(KEY,'assembling_shards',split=split,shard=shard)
    tests=[]
    for p in sorted((root/'data_test').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:tests.append({k:z[k] for k in ['X','x','y']})
    target=root/'independent_test.npz'
    with target.with_name(target.name+'.tmp').open('wb') as f:
        np.savez_compressed(f,**{k:np.concatenate([d[k] for d in tests]) for k in ['X','x','y']})
    target.with_name(target.name+'.tmp').replace(target)
    save(root/'dataset_metadata.json',dict(meta,generation_status='complete'))
    save(root/'test_manifest.json',dict(rows=2000,sha256=sha(target),generation_signature=signature))
    save(ROOT/'preflight/readiness.json',dict(passed=True,generation_signature=signature,
        train_rows=410640,valid_rows=25500,test_rows=2000,test_sha256=sha(target)))

def run():
    import fcntl
    with (ROOT/'.data_generation.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (ROOT/'done/data_generate.json').exists():return
        plan=load(ROOT/'deployment_protocol.json');meta=plan['dataset_spec'];sig=meta['generation_signature']
        with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:
            positions={k:z[n].copy() for k,n in [('train','train_positions'),('valid','validation_positions'),('test','test_positions')]}
            selected=z['candidate_training_position_indices'].copy()
            ids=np.concatenate((z['micro_training_row_indices'],z['micro_validation_row_indices']))
        micro=[]
        for index in selected:
            reps=np.sort(ids[ids//60==index]%60).tolist();old=existing('train',int(index),sig)
            missing=[r for r in reps if old is None or r not in old['repetitions']]
            if missing:micro.append(('train',int(index),*positions['train'][index],missing,2025,meta))
        if not (ROOT/'done/micro_ready.json').exists():
            stage(micro,plan['resources']['micro_generation_workers'],sig,'generating_priority_micro_realizations')
            write_micro(plan,sig)
        else:
            manifest=load(ROOT/'micro_dataset/sampling_manifest.json')
            for name,digest in manifest['artifacts_sha256'].items():assert sha(ROOT/'micro_dataset'/name)==digest
        tasks=[]
        for split,n,seed in [('valid',15,502025),('train',60,2025),('test',1,1002025)]:
            for index,(x,y) in enumerate(positions[split]):
                old=existing(split,index,sig)
                missing=[r for r in range(n) if old is None or r not in old['repetitions']]
                if missing:tasks.append((split,index,float(x),float(y),missing,seed,meta))
        stage(tasks,plan['resources']['full_generation_workers'],sig,'generating_remaining_full_realizations')
        write_dataset(meta,positions)
        mark_done(KEY,generation_signature=sig,test_sha256=sha(ROOT/'data/independent_test.npz'))
        report(KEY,'complete')

if __name__=='__main__':run()
