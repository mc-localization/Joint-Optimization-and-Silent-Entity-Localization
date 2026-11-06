"""New Rx20 data: an independently trained DNN start optimizer and MLE localizer."""
from pathlib import Path
import argparse,copy,csv,hashlib,json,os,subprocess,sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import run_experiment_4rx as base
import policy_search as search
import mle_likelihood_original as mle
from experiment_common import load,save,sha,utc,identity,alive,write_csv,read_csv

OUT=ROOT/'mle_dnn_optimizer'
LENGTHS=[50,100,150,200,250]
Q,M,N_RX=20000,250,4
PSEUDOCOUNT=.5
LEGACY_HELPER_SHA='c391a458dee0385d58bf76bcea8c73577410f51c05f1a2c3f35ef73408cdf5d4'

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def report(phase,**kw):
    row=dict(phase=phase,pid=os.getpid(),at_utc=utc(),**kw)
    save(OUT/'status.json',row);print(json.dumps(row),flush=True)

def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def increments(features):
    x=np.asarray(features,dtype=np.float64).reshape(-1,N_RX,M)
    assert np.isfinite(x).all() and (x>=0).all()
    delta=np.diff(x,axis=2,prepend=np.zeros((len(x),N_RX,1)))
    assert (delta>=-1e-8).all()
    return np.maximum(delta,0)

def infer(template_inc,observed_inc,points,repeats,start,length):
    # Same float64, observation batching and first-index argmax as the old
    # MLE-specific DNN optimizer implementation. Only dimensions are dynamic.
    assert 1<=length<=M and 1<=start<=M-length+1
    left=start-1
    counts=template_inc[:,:,left:left+length].reshape(len(points),-1)
    observations=observed_inc[:,:,left:left+length].reshape(len(observed_inc),-1)
    probability=mle.smooth_probabilities(counts,Q,repeats,PSEUDOCOUNT)
    full=np.column_stack([observations,np.maximum(Q-observations.sum(axis=1),0)])
    logp=torch.as_tensor(np.log(probability).T.copy(),dtype=torch.float64)
    result=np.empty((len(full),2),dtype=np.float64)
    with torch.inference_mode():
        for left in range(0,len(full),256):
            chosen=torch.argmax(torch.as_tensor(full[left:left+256])@logp,dim=1).numpy()
            result[left:left+256]=points[chosen]
    return result

def micro_data(protocol):
    manifest=load(ROOT/'micro_dataset/sampling_manifest.json');data=[]
    assert manifest['generation_signature']==protocol['generation_signature']
    for name in ['micro_train.npz','micro_valid.npz']:
        path=ROOT/'micro_dataset'/name;assert sha(path)==manifest['artifacts_sha256'][name]
        with np.load(path,allow_pickle=False) as z:data.append({k:z[k].copy() for k in ['X','x','y']})
    train,valid=data
    assert (len(train['X']),len(valid['X']))==(12840,3424)
    xy=np.column_stack([train['x'],train['y']])
    points,inverse,repeats=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
    assert len(points)==428 and np.all(repeats==30)
    vpoints,vreps=np.unique(np.column_stack([valid['x'],valid['y']]),axis=0,return_counts=True)
    assert np.array_equal(vpoints,points) and np.all(vreps==8)
    sums=np.zeros((len(points),M*N_RX),dtype=np.float64)
    np.add.at(sums,inverse,train['X']);templates=sums/repeats[:,None]
    return train,valid,templates,points,repeats

def check_sources(protocol):
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'micro_dataset/sampling_manifest.json')==protocol['micro_manifest_sha256']
    assert sha(ROOT/'candidate_position_plan.json')==protocol['candidate_plan_sha256']
    assert sha(ROOT/'position_plan.npz')==protocol['position_plan_sha256']
    current=load(ROOT/'deployment_protocol.json')
    assert current['dataset_spec']['generation_signature']==protocol['generation_signature']

def make_protocol():
    current=load(ROOT/'deployment_protocol.json')
    assert sha(ROOT/'mle_likelihood_original.py')==LEGACY_HELPER_SHA
    cfg=copy.deepcopy(current['cases']['cnn']['search_cfg'])
    cfg['paths']['output_dir']=str(OUT/'search')
    cfg['training'].update(seed=2026,device='cpu')
    cfg['_localizer']='mle';cfg['_variant']='full';cfg.pop('_signature',None)
    cfg['comparison']['cache_sources']=[]
    cfg['final']['lengths']=LENGTHS
    files=['run_mle_current.py','mle_likelihood_original.py','experiment_common.py',
        'runtime/policy_search.py','runtime/run_experiment_4rx.py','runtime/retest_models_4rx.py','runtime/npz_io.py']
    protocol=dict(method='MLE localizer with its independently trained DNN detection-start optimizer',
        evaluation_lengths=LENGTHS,configuration=cfg,
        generation_signature=current['dataset_spec']['generation_signature'],
        micro_manifest_sha256=sha(ROOT/'micro_dataset/sampling_manifest.json'),
        candidate_plan_sha256=sha(ROOT/'candidate_position_plan.json'),position_plan_sha256=sha(ROOT/'position_plan.npz'),
        source_sha256={n:sha(ROOT/n) for n in files},legacy_likelihood_helper_sha256=LEGACY_HELPER_SHA,
        micro_template_positions=428,micro_template_repetitions=30,micro_validation_repetitions=8,
        micro_training_samples=12840,micro_validation_samples=3424,
        final_training_positions=6844,final_training_repetitions=60,final_training_samples=410640,
        test_samples=2000,test_sha256=sha(ROOT/'data/independent_test.npz'),
        likelihood='Training-simulation empirical multinomial; receiver/time-interval outcomes plus residual',
        pseudocount=PSEUDOCOUNT,template_domain='Current formal training positions',
        selection_metric='Mean Euclidean micro-validation localization error in micrometers',
        optimizer='DNN with hidden dimensions [64,64], original ten-round search and per-round Adam fitting',
        optimizer_seed=2026,optimizer_loss='MSE of normalized detection start time fractions',
        search_length_pool='All integer lengths 1..249, same as current CNN and DNN',
        final_start_selection='Frozen MLE-specific DNN prediction',
        CPU_threads=1,GPU_used=False,old_geometry_templates_reused=False,other_model_optimizers_reused=False)
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol['protocol_sha256']=digest;cfg['_signature']=digest
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():assert load(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    return protocol

def preflight():
    protocol=make_protocol();check_sources(protocol)
    train,valid,templates,points,repeats=micro_data(protocol)
    ti,vi=increments(templates),increments(valid['X'])
    comparisons=[]
    for start,length in [(1,50),(92,100),(101,150),(1,250)]:
        a=infer(ti[:24],vi[:32],points[:24],repeats[:24],start,length)
        count=mle.interval_counts(templates[:24],start,length,M,N_RX)
        observed=mle.interval_counts(valid['X'][:32],start,length,M,N_RX)
        probability=mle.smooth_probabilities(count,Q,repeats[:24],PSEUDOCOUNT)
        b=mle.predict(observed,probability,points[:24],Q,torch.device('cpu'),16,7)
        assert np.array_equal(a,b)
        comparisons.append(dict(start=start,length=length,identical_to_original_helper=True))
    cfg=protocol['configuration'];schedule=np.random.default_rng(cfg['search']['schedule_seed']).choice(
        list(range(1,M)),size=(cfg['search']['iters'],cfg['search']['q_per_round']),replace=True).tolist()
    assert schedule==load(ROOT/'search/cnn/schedule.json')['schedule']
    assert cfg['policy']['hidden_dims']==[64,64]
    assert cfg['search']['iters']==10 and cfg['search']['q_per_round']==6
    save(OUT/'preflight.json',dict(passed=True,at_utc=utc(),protocol_sha256=protocol['protocol_sha256'],
        legacy_likelihood_equivalence=comparisons,same_length_sampling_schedule_as_cnn=True,
        micro_training_samples=len(train['X']),micro_validation_samples=len(valid['X']),
        template_positions=len(points),cpu_threads=torch.get_num_threads()))
    print(json.dumps(dict(passed=True,protocol_sha256=protocol['protocol_sha256'],comparisons=comparisons)),flush=True)

def train_optimizer(protocol):
    cfg=copy.deepcopy(protocol['configuration']);folder=OUT/'search';folder.mkdir(exist_ok=True)
    save(folder/'server_config.json',cfg)
    train,valid,templates,points,repeats=micro_data(protocol)
    ti,vi=increments(templates),increments(valid['X'])
    truth=np.column_stack([valid['x'],valid['y']]).astype(np.float64)
    cache=folder/'score_cache';cache.mkdir(exist_ok=True)
    def score(local,length,start,*args):
        path=cache/f'l{length}_s{start}.json'
        if path.exists():
            row=load(path);assert row['protocol_sha256']==protocol['protocol_sha256'];return row,0.,True
        began=time.perf_counter()
        report('MLE_candidate_evaluation',length=length,start=start,
            completed_candidates=len(list(cache.glob('*.json'))),current_round=round_now[0])
        prediction=infer(ti,vi,points,repeats,start,length);error=np.linalg.norm(prediction-truth,axis=1)
        row=dict(model='mle',length=int(length),start=int(start),seed=2026,mean_error_um=float(error.mean()),
            val_mean_error_um=float(error.mean()),median_error_um=float(np.median(error)),
            success_rate=float((error<.4).mean()),accuracy025=float((error<.25).mean()),
            n_validation=len(truth),n_template_positions=len(points),training_samples=len(train['X']),
            pseudocount=PSEUDOCOUNT,protocol_sha256=protocol['protocol_sha256'])
        save(path,row);return row,time.perf_counter()-began,False
    native=search.candidates;calls={};budgets={};budgetfile=folder/'candidate_budgets.csv';round_now=[1]
    if budgetfile.exists():budgets={(int(r['round']),int(r['slot'])):r for r in read_csv(budgetfile)}
    def candidates(local,method,rng,center,length,round_idx,visited):
        slot=calls.get(round_idx,0);calls[round_idx]=slot+1;round_now[0]=round_idx+1
        chosen=native(local,method,rng,center,length,round_idx,visited)
        budgets[round_idx+1,slot]=dict(round=round_idx+1,slot=slot,length=length,center=center,
            evaluations=len(chosen),candidate_start_steps=json.dumps(chosen))
        write_csv(budgetfile,[budgets[k] for k in sorted(budgets)]);return chosen
    base.load_splits=lambda local,final,include_test=False:(train,valid,None,train,valid)
    search.score=score;search.candidates=candidates;search.refit=lambda *args,**kw:None
    report('training_DNN_start_optimizer',total_rounds=10,device='cpu')
    search.search(cfg,'guided')
    policy=base.PolicyNet(cfg['policy']['hidden_dims'])
    policy.load_state_dict(torch.load(folder/'policy_raw.pt',map_location='cpu',weights_only=True))
    starts={str(l):search.predict(policy,l,M) for l in range(1,M+1)}
    assert all(1<=s<=M-int(l)+1 for l,s in starts.items())
    assert load(folder/'schedule.json')==load(ROOT/'search/cnn/schedule.json')
    budgets=read_csv(budgetfile);reference=read_csv(ROOT/'search/cnn/candidate_budgets.csv')
    assert len(budgets)==len(reference)==60
    assert [(int(r['length']),int(r['evaluations'])) for r in budgets]==[(int(r['length']),int(r['evaluations'])) for r in reference]
    frozen=dict(starts=starts,evaluation_lengths=LENGTHS,optimizer_path=str(folder/'policy_raw.pt'),
        optimizer_sha256=sha(folder/'policy_raw.pt'),optimizer_seed=2026,
        protocol_sha256=protocol['protocol_sha256'],candidate_evaluations=sum(int(r['evaluations']) for r in budgets),
        at_utc=utc(),test_loaded_after_optimizer_freeze=True)
    save(OUT/'frozen_optimizer.json',frozen)
    report('DNN_optimizer_frozen',candidate_evaluations=frozen['candidate_evaluations'],
        starts={str(l):starts[str(l)] for l in LENGTHS})
    return frozen

def full_templates(protocol):
    target=OUT/'templates.npz';record=OUT/'templates_manifest.json'
    if target.exists() and record.exists():
        rec=load(record);assert rec['protocol_sha256']==protocol['protocol_sha256'] and rec['sha256']==sha(target)
        with np.load(target,allow_pickle=False) as z:return z['templates'].copy(),z['positions'].copy(),z['repeats'].copy()
    report('building_final_MLE_templates',training_samples=410640,training_positions=6844)
    shards={}
    for path in sorted((ROOT/'data/data_train').glob('*.npz')):
        rec=load(path.with_suffix('.npz.json'));assert rec['generation_signature']==protocol['generation_signature']
        digest=sha(path);assert digest==rec['sha256'];shards[str(path)]=digest
    templates,points,repeats=mle.build_training_templates(ROOT,410640,6844,Q,M,N_RX,data_dir=ROOT/'data')
    assert templates.shape==(6844,1000) and np.all(repeats==60)
    with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:
        assert np.allclose(points,z['train_positions'],rtol=0,atol=1e-12)
    save_npz(target,templates=templates,positions=points,repeats=repeats)
    save(record,dict(protocol_sha256=protocol['protocol_sha256'],sha256=sha(target),training_shards_sha256=shards,
        n_templates=len(points),training_samples=int(repeats.sum()),at_utc=utc()))
    return templates,points,repeats

def run():
    import fcntl
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            protocol=load(OUT/'protocol.json');check_sources(protocol)
            assert load(OUT/'preflight.json')['protocol_sha256']==protocol['protocol_sha256']
            if (OUT/'frozen_optimizer.json').exists():
                frozen=load(OUT/'frozen_optimizer.json');assert frozen['protocol_sha256']==protocol['protocol_sha256']
                assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
            else:frozen=train_optimizer(protocol)
            templates,points,repeats=full_templates(protocol);ti=increments(templates)
            assert sha(ROOT/'data/independent_test.npz')==protocol['test_sha256']
            with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:
                vi=increments(z['X']);truth=np.column_stack([z['x'],z['y']]).astype(np.float64)
            assert len(truth)==2000;results=[]
            for length in LENGTHS:
                start=int(frozen['starts'][str(length)]);folder=ROOT/'results/mle_optimized'/f'l{length}'
                path=folder/'result.json'
                if path.exists():
                    row=load(path);assert row['protocol_sha256']==protocol['protocol_sha256'] and row['start']==start
                    assert row['optimizer_sha256']==frozen['optimizer_sha256'];results.append(row);continue
                report('final_MLE_test',length=length,start=start,completed_lengths=len(results),total_lengths=len(LENGTHS))
                began=time.perf_counter();pred=infer(ti,vi,points,repeats,start,length);err=np.linalg.norm(pred-truth,axis=1)
                save_npz(folder/'predictions.npz',predictions=pred,truth=truth,errors_um=err)
                row=dict(model='mle',condition='optimized',length=length,start=start,n_test=len(truth),seed=2025,
                    mean_error_um=float(err.mean()),median_error_um=float(np.median(err)),p90_error_um=float(np.quantile(err,.9)),
                    success_rate=float((err<.4).mean()),accuracy025=float((err<.25).mean()),
                    training_loss='not_applicable_to_MLE',optimizer_loss='normalized_start_fraction_MSE',
                    n_templates=len(points),n_training_samples=int(repeats.sum()),pseudocount=PSEUDOCOUNT,
                    test_sha256=protocol['test_sha256'],templates_sha256=sha(OUT/'templates.npz'),
                    predictions_sha256=sha(folder/'predictions.npz'),optimizer_sha256=frozen['optimizer_sha256'],
                    optimizer_seed=2026,start_selection='MLE-specific frozen DNN optimizer prediction',
                    generation_signature=protocol['generation_signature'],protocol_sha256=protocol['protocol_sha256'],
                    inference_device='cpu',inference_seconds=time.perf_counter()-began)
                save(path,row);write_csv(folder/'final_test.csv',[row]);results.append(row)
                write_csv(OUT/'all_mle_results.csv',results)
            check_sources(protocol)
            write_csv(OUT/'all_mle_results.csv',results)
            save(OUT/'ready.json',dict(ready=True,at_utc=utc(),lengths=LENGTHS,
                optimizer_sha256=frozen['optimizer_sha256'],protocol_sha256=protocol['protocol_sha256']))
            report('complete',completed_lengths=len(results),total_lengths=len(LENGTHS),
                candidate_evaluations=frozen['candidate_evaluations'])
        except Exception:
            report('failed',traceback=traceback.format_exc());raise

def launch():
    OUT.mkdir(exist_ok=True);path=OUT/'launch.json'
    if path.exists() and alive(load(path)['identity']):print(json.dumps(load(path)));return
    protocol=load(OUT/'protocol.json');check_sources(protocol)
    assert load(OUT/'preflight.json')['passed']
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    result=dict(identity=identity(child.pid),at_utc=utc(),evaluation_lengths=LENGTHS,
        localizer='MLE',start_optimizer='DNN',CPU_threads=1,GPU_used=False)
    save(path,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();configure()
    if args.preflight:preflight()
    elif args.launch:launch()
    elif args.run:run()
    else:parser.print_help()
