"""Bounded checks for graph dependencies, real microdata and dynamic dimensions."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from experiment_common import ROOT,load,save,sha,utc
import run_paper_suite as scheduler
import paper_suite_queue as queue
import paper_suite_workers as workers

def main():
    workers.common.configure();protocol=load(ROOT/'deployment_protocol.json')
    scheduler.check_sources(protocol)
    jobs=scheduler.static_jobs(protocol);scheduler.reconcile(protocol,jobs);queue.reconcile(jobs)
    assert scheduler.COARSE_LENGTHS==scheduler.TRANSFORMER_LENGTHS==scheduler.GRU_LENGTHS==list(range(25,251,25))
    assert jobs['figure_fig6']['length']==100
    assert len([j for j in jobs.values() if j['kind']=='convergence'])==4
    # Directed dependencies must stay acyclic, including GRU search before Fig7.
    visiting=set();visited=set()
    def visit(key):
        if key in visited or key not in jobs:return
        assert key not in visiting,key
        visiting.add(key)
        for dep in jobs[key]['deps']:visit(dep)
        visiting.remove(key);visited.add(key)
    for key in jobs:visit(key)
    assert queue.rank(jobs['convergence60_cnn'])<queue.rank(jobs['fit_cnn_l25_s1'])
    assert queue.rank(jobs['fit_cnn_l25_s1'])<queue.rank(jobs['parameter_policy_q10000'])
    cfg=workers.cfg_for('q10000','search');tr,va=workers.micro('q10000',cfg)
    scaler=workers.base.TargetScaler(tr)
    tests=[]
    for M,nrx,length in [(125,4,100),(250,4,100),(375,4,300),(500,4,400),(250,2,200)]:
        local=json.loads(json.dumps(cfg));local['data'].update(M=M,source_M=M,n_rx=nrx)
        local['data']['precompute_windows']=True
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        # Dimension sanity only: reuse diverse real rows on an index grid. This
        # is never used as a substitute for the independently generated datasets.
        ids=np.linspace(0,len(tr['x'])-1,32,dtype=int)
        indices=np.rint(np.linspace(0,249,M)).astype(int)
        raw=dict(X=tr['X'][ids].reshape(32,4,250)[:,:nrx,indices].reshape(32,nrx*M),
            x=tr['x'][ids],y=tr['y'][ids])
        workers.base.configure_seed(local,2025)
        ds=workers.base.make_dataset(raw,scaler,length,1,local)
        xb,target=next(iter(workers.base.make_loader(ds,local,torch.device('cuda'),length,False)))
        net=workers.base.build_localizer(local,'cnn',length).cuda();net.train()
        with torch.amp.autocast(device_type='cuda',enabled=True):
            prediction=workers.base.forward_model(net,xb.cuda())
            loss=workers.base.localizer_loss(prediction,target.cuda(),torch.tensor(ds.physical_scale,device='cuda'),'mse_physical_um2')
        loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None),(M,nrx,length)
        tests.append(dict(M=M,n_rx=nrx,length=length,finite_amp_loss_and_gradients=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        del net,xb,target,prediction,loss,ds
    # Validate a full configured batch on the already generated Q=10000 data.
    ds=workers.base.make_dataset(tr,scaler,100,1,cfg)
    xb,target=next(iter(workers.base.make_loader(ds,cfg,torch.device('cuda'),100,False)))
    net=workers.base.build_localizer(cfg,'cnn',100).cuda();net.train()
    with torch.amp.autocast(device_type='cuda',enabled=True):
        loss=workers.base.localizer_loss(workers.base.forward_model(net,xb.cuda()),target.cuda(),
            torch.tensor(ds.physical_scale,device='cuda'),'mse_physical_um2')
    loss.backward();assert torch.isfinite(loss)
    result=dict(passed=True,at_utc=utc(),source_hashes_checked=True,acyclic_dependencies=True,
        fig6_length=100,fig7_epochs=60,fig7_patience=61,fig7_early_stopping=False,
        main_lengths=scheduler.TARGET_LENGTHS,queued_jobs=len(jobs),
        real_q10000_micro_rows=[len(tr['x']),len(va['x'])],shared_coordinate_row_order_verified=True,
        dynamic_dimension_tests=tests,experiment_optimizer_steps=0,
        suite_protocol_sha256=sha(ROOT/'paper_suite_protocol.json'))
    save(ROOT/'preflight/paper_suite_startup.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
