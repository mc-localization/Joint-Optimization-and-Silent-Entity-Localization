"""Deterministic, resumable shards around the unchanged particle simulator."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import generate_4rx_dataset as physics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def spec(cfg):
    s=dict(cfg['simulation'])
    assert 0<s['molecules']<=np.iinfo(np.uint16).max
    assert s['steps']>=s['features']>1 and abs(s['steps']*s['dt']-5.)<1e-10
    grid,parity=physics.make_grid(s['grid_step'])
    train=grid[parity!=0];valid=grid[parity==0]
    meta=dict(s,tx=physics.TX,receivers=physics.RX_LIST,eve_radius=physics.R_E,
        x_range=physics.X_RANGE,y_range=physics.Y_RANGE,train_positions=len(train),valid_positions=len(valid),
        train_rows=len(train)*s['train_repeats'],valid_rows=len(valid)*s['valid_repeats'],test_rows=s['test_points'],
        split='checkerboard parity=(ix+iy)%5; validation parity=0',
        sample_times_s=((np.rint(np.linspace(0,s['steps']-1,s['features'])).astype(int)+1)*s['dt']).tolist(),
        shard_order='deterministic spatial order; deterministic replica order',coordinate_dtype='float64',
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(physics.__file__)]})
    meta=json.loads(json.dumps(meta))
    meta['generation_signature']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    assert cfg['data']['molecules']==s['molecules'] and cfg['data']['M']==s['features']
    assert cfg['data']['final_train_samples']==meta['train_rows']
    assert cfg['data']['final_valid_samples']==meta['valid_rows']
    return meta,train,valid


def verify(path,meta):
    path=Path(path);lock=path.with_suffix('.npz.json')
    if not path.exists() or not lock.exists():return False
    row=json.loads(lock.read_text())
    assert row['generation_signature']==meta['generation_signature'],'Dataset protocol changed'
    assert row['sha256']==sha(path),'Dataset shard changed: '+str(path)
    return True


def write_split(root,name,positions,repeats,seed_base,meta,pool):
    directory=root/name;directory.mkdir(parents=True,exist_ok=True)
    per_shard=max(1,1000//repeats)
    expected=set()
    for shard,left in enumerate(range(0,len(positions),per_shard)):
        target=directory/f'{name}_{shard:03d}.npz';expected.add(target.name)
        if verify(target,meta):
            print(json.dumps(dict(phase='reuse_data_shard',split=name,shard=shard)),flush=True);continue
        tasks=[(i,float(x),float(y),repeats,seed_base,meta['molecules'],meta['dt'],meta['diffusion'],meta['steps'],meta['features'])
               for i,(x,y) in enumerate(positions[left:left+per_shard],start=left)]
        rows=[r for group in pool.map(physics.simulate_position,tasks) for r in group]
        values=np.stack([r[0] for r in rows]);counts=values.reshape(-1,4,meta['features'])
        assert np.all(np.diff(counts.astype(np.int32),axis=2)>=0)
        assert np.all(counts[:,:,-1].sum(axis=1)<=meta['molecules'])
        temp=target.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,X=values,x=np.array([r[1] for r in rows]),y=np.array([r[2] for r in rows]))
        temp.replace(target)
        save(target.with_suffix('.npz.json'),dict(generation_signature=meta['generation_signature'],rows=len(rows),sha256=sha(target)))
        print(json.dumps(dict(phase='generated_data_shard',split=name,shard=shard,rows=len(rows),positions_done=min(left+per_shard,len(positions)),positions_total=len(positions))),flush=True)
    assert {p.name for p in directory.glob('*.npz')}==expected,'Unexpected data shards'


def generate(cfg,workers):
    root=Path(cfg['paths']['data_dir']);meta,train,valid=spec(cfg)
    lock=root/'generation_manifest.json'
    if lock.exists():assert json.loads(lock.read_text())==meta,'Baseline data inputs changed'
    else:
        assert not any(root.glob('data_*/*.npz')) and not (root/'independent_test.npz').exists(),'Untracked old data exist'
        save(lock,meta)
    test=physics.continuous_points(meta['test_points'],meta['seed']+999)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        write_split(root,'data_train',train,meta['train_repeats'],meta['seed'],meta,pool)
        write_split(root,'data_valid',valid,meta['valid_repeats'],meta['seed']+500000,meta,pool)
        write_split(root,'data_test',test,1,meta['seed']+1000000,meta,pool)
    target=root/'independent_test.npz'
    if not verify(target,meta):
        arrays={k:[] for k in ['X','x','y']}
        for path in sorted((root/'data_test').glob('*.npz')):
            with np.load(path,allow_pickle=False) as data:
                for k in arrays:arrays[k].append(data[k])
        temp=target.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,**{k:np.concatenate(v) for k,v in arrays.items()})
        temp.replace(target)
        save(target.with_suffix('.npz.json'),dict(generation_signature=meta['generation_signature'],rows=meta['test_rows'],sha256=sha(target)))
    save(root/'test_manifest.json',dict(rows=meta['test_rows'],sha256=sha(target),test_used_for_selection=False,generation_signature=meta['generation_signature']))
    save(root/'dataset_metadata.json',dict(meta,generation_status='complete'))
    return meta
