"""Server-side preflight. No experiment optimizer steps or background launch."""
from pathlib import Path
import copy,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
import numpy as np
import torch
from experiment_common import load,save,sha,utc,cpu_quota
import generate_4rx_dataset as physics
import run_experiment_4rx as base
import priority_data
from run_experiment import check_sources,static_jobs

def main():
    p=load(ROOT/'deployment_protocol.json');check_sources(p)
    assert torch.cuda.is_available() and '3090' in torch.cuda.get_device_name(0)
    assert shutil.disk_usage(ROOT).free>6*2**30
    assert cpu_quota()>=p['resources']['micro_generation_workers']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with np.load(ROOT/'position_plan.npz',allow_pickle=False) as z:
        train=z['train_positions'].copy();valid=z['validation_positions'].copy();test=z['test_positions'].copy()
        selected=z['candidate_training_position_indices'].copy()
        tr=z['micro_training_row_indices'].copy();va=z['micro_validation_row_indices'].copy()
    assert (len(train),len(valid),len(test),len(selected))==(6844,1700,2000,428)
    assert not np.intersect1d(tr,va).size and len(tr)==12840 and len(va)==3424
    assert all(physics.valid_point(*xy) for xy in np.concatenate((train,valid,test)))
    x=torch.tensor([[1.,2.],[0.,0.]],device='cuda',requires_grad=True)
    loss=base.localizer_loss(x,torch.zeros_like(x),torch.tensor([2.,3.],device='cuda'),'mse_physical_um2')
    assert loss.item()==20.;loss.backward()
    assert torch.equal(x.grad.cpu(),torch.tensor([[4.,18.],[0.,0.]]))
    tests=[]
    for model in ['cnn','dnn']:
        cfg=p['cases'][model]['final_cfg'];base.configure_seed(cfg,2025)
        net=base.build_localizer(cfg,model,100).cuda();net.train()
        xb=torch.rand(16,400,device='cuda')*30;target=torch.randn(16,2,device='cuda')
        with torch.amp.autocast(device_type='cuda',enabled=True):
            pred=base.forward_model(net,xb)
            value=base.localizer_loss(pred,target,torch.tensor([12.,12.],device='cuda'),'mse_physical_um2')
        value.backward();assert torch.isfinite(value)
        assert all(torch.isfinite(w.grad).all() for w in net.parameters() if w.grad is not None)
        tests.append(dict(model=model,parameters=sum(w.numel() for w in net.parameters()),amp_gradient_check=True))
        del net,xb,target,pred,value
    print(json.dumps(dict(phase='neural_preflight_passed',models=tests)),flush=True)
    index=int(selected[0]);xy=train[index];repetition=int(tr[tr//60==index][0]%60)
    task=('train',index,float(xy[0]),float(xy[1]),[repetition],2025,p['dataset_spec'])
    result=priority_data.simulate_task(task)
    expected=physics.simulate(*xy,2025+index*100003+repetition,20000,1000,.005,79.4,4.,250)
    assert np.array_equal(result[-1][0],expected)
    arr=expected.reshape(4,250).astype(np.int32)
    assert (np.diff(arr,axis=1)>=0).all() and arr[:,-1].sum()<=20000
    print(json.dumps(dict(phase='physical_preflight_passed',rx_count=4,steps=1000,Q=20000)),flush=True)
    # Exercise the inherited policy algorithm with a deterministic mock score.
    import policy_search as search
    cfg=copy.deepcopy(p['cases']['cnn']['search_cfg'])
    cfg['paths']['output_dir']=str(ROOT/'preflight/policy_smoke')
    cfg['training']['device']='cpu';cfg['search'].update(iters=1,q_per_round=2,policy_epochs_per_round=2)
    cfg['_signature']='preflight_only_not_formal_search'
    mock=dict(X=np.zeros((4,1000),dtype=np.uint16),x=np.array([-1.,-1.,1.,1.]),y=np.array([-1.,1.,-1.,1.]))
    base.load_splits=lambda *args,**kw:(mock,mock,None,mock,mock)
    search.score=lambda local,l,s,*args:(dict(length=l,start=s,mean_error_um=float((s-50)**2+1)),0.,False)
    search.refit=lambda *args,**kw:None
    search.search(cfg,'guided')
    assert (Path(cfg['paths']['output_dir'])/'policy_raw.pt').exists()
    jobs=static_jobs(p)
    assert all(j.get('model') in [None,'cnn','dnn'] for j in jobs.values())
    result=dict(passed=True,at_utc=utc(),generation_signature=p['dataset_spec']['generation_signature'],
        source_fingerprints_checked=True,initial_jobs=len(jobs),gpu=torch.cuda.get_device_name(0),
        cpu_quota=cpu_quota(),disk_free_gib=shutil.disk_usage(ROOT).free/2**30,
        geometry=True,micro_realization_plan=True,physical_simulator_seed_equivalence=True,
        mse_value_and_gradient=True,amp_models=tests,policy_algorithm_smoke=True,
        formal_experiment_optimizer_steps=0)
    save(ROOT/'preflight/startup.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
