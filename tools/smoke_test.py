"""CPU-only checks of the exact archived simulation/training primitives."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiment'),str(ROOT/'experiment/runtime')]
import numpy as np
import torch
import run_experiment_4rx as base
import dnn10_width100_cnn50 as ten_layer
import mle_likelihood_original as mle
import policy_search
import generate_4rx_dataset as physics

def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ten_layer.install_model(base)
    with np.load(ROOT/'reference/smoke_sample.npz',allow_pickle=False) as z:
        datasets={split:{key:z[split+'_'+key].copy() for key in ['X','x','y']} for split in ['train','valid','test']}
    scaler=base.TargetScaler(datasets['train']);report={'device':'cpu','training_epochs_per_model':1,'models':{}}
    expected={'cnn':108290,'dnn':113602,'gru':112578,'transformer':113619}
    for name in expected:
        record=json.loads((ROOT/f'configs/main/{name}_l50.json').read_text(encoding='utf-8'))
        cfg=copy.deepcopy(record['configuration']);cfg['training'].update(device='cpu',amp=False,num_workers=0)
        cfg['data']['precompute_windows']=True
        model=base.build_localizer(cfg,name,50)
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count==expected[name],(name,count)
        model.eval()
        with torch.no_grad():prediction=base.forward_model(model,torch.ones(4,200))
        assert prediction.shape==(4,2) and torch.isfinite(prediction).all()
        train,valid=[base.make_dataset(datasets[k],scaler,50,record['start'],cfg) for k in ['train','valid']]
        state,loss,epoch,seconds=base.train_localizer(cfg,name,50,train,valid,torch.device('cpu'),1,1)
        assert math.isfinite(loss) and epoch==1 and all(torch.isfinite(p).all() for p in state.values())
        report['models'][name]={'parameters_l50':count,'one_epoch_validation_mse_um2':loss,'seconds':seconds}
    policy=base.PolicyNet([64,64]);assert sum(p.numel() for p in policy.parameters())==4353
    for length in range(1,251):
        for fraction in [0.,.1,.5,.9,1.]:assert 1<=base.decode_start(fraction,length,250)<=251-length
    cfg=json.loads((ROOT/'configs/main/cnn_l50.json').read_text(encoding='utf-8'))['configuration']
    for length in [1,25,50,125,249,250]:
        picks=policy_search.candidates(cfg,'guided',np.random.default_rng(2025),1,length,0,set())
        assert len(picks)==len(set(picks)) and all(1<=s<=251-length for s in picks)
    template=mle.interval_counts(datasets['train']['X'][:4],1,50,250,4)
    observed=mle.interval_counts(datasets['valid']['X'][:8],1,50,250,4)
    probabilities=mle.smooth_probabilities(template,20000,np.ones(4))
    positions=np.column_stack([datasets['train']['x'][:4],datasets['train']['y'][:4]])
    prediction=mle.predict(observed,probabilities,positions,20000,torch.device('cpu'),4,2)
    assert prediction.shape==(8,2) and np.isfinite(prediction).all()
    simulated=physics.simulate(10.,0.,2025,20,10,.005,79.4,4.,5)
    counts=np.asarray(simulated).reshape(4,5)
    assert np.all(np.diff(counts.astype('int64'),axis=1)>=0) and counts[:,-1].sum()<=20
    report.update(policy_parameters=4353,feasible_start_checks=1250,mle_inference=True,particle_simulator=True)
    return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
