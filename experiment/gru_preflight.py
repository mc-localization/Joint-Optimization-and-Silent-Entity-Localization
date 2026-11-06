"""Check the additional GRU without taking experiment optimizer steps."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import torch
from experiment_common import load,save,sha,utc
import gru_workers as workers
import run_with_gru as scheduler

def main():
    workers.configure()
    original=load(ROOT/'deployment_protocol.json');extra=load(ROOT/'gru_protocol.json')
    scheduler.check_sources(original)
    cfg=workers.cfg_for('gru','search')
    assert cfg['training']['loss']=='mse_physical_um2' and cfg['training']['seed']==2026
    assert workers.cfg_for('gru')['training']['seed']==2025
    assert cfg['models']['gru']==original['cases']['cnn']['search_cfg']['models']['gru']
    for key in ['data','training','search','policy','simulation','geometry']:
        assert cfg[key]==original['cases']['cnn']['search_cfg'][key],key
    train,valid=workers.micro_data(cfg)
    assert len(train['x'])==12840 and len(valid['x'])==3424
    scaler=workers.base.TargetScaler(train);tests=[]
    for length in [50,100,149,150,200,249,250]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        dataset=workers.base.make_dataset(train,scaler,length,1,cfg)
        loader=workers.base.make_loader(dataset,cfg,torch.device('cuda'),length,shuffle=False)
        xb,target=next(iter(loader));net=workers.base.build_localizer(cfg,'gru',length).cuda();net.train()
        with torch.amp.autocast(device_type='cuda',enabled=True):
            prediction=workers.base.forward_model(net,xb.cuda(non_blocking=True))
            loss=workers.base.localizer_loss(prediction,target.cuda(non_blocking=True),
                torch.tensor(dataset.physical_scale,device='cuda'),'mse_physical_um2')
        loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        peak=torch.cuda.max_memory_reserved()/2**30
        assert peak<8.,(length,peak)
        tests.append(dict(length=length,batch_size=len(xb),parameters=sum(p.numel() for p in net.parameters()),
            peak_reserved_gib=peak,finite_amp_loss_and_gradients=True))
        del dataset,loader,net,xb,target,prediction,loss
    jobs=scheduler.static_jobs(original);scheduler.reconcile(original,jobs)
    assert jobs['policy_gru']['model']=='gru'
    assert scheduler.GRU_LENGTHS==[50,100,150,200,250]
    result=dict(passed=True,at_utc=utc(),source_fingerprints_checked=True,
        gru_configuration_matches_previous_architecture=True,same_training_and_search_protocol_as_cnn=True,
        micro_training_rows=len(train['x']),micro_validation_rows=len(valid['x']),
        generation_signature=extra['generation_signature'],tests=tests,experiment_optimizer_steps=0,
        gru_protocol_sha256=sha(ROOT/'gru_protocol.json'))
    save(ROOT/'preflight/gru_startup.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
