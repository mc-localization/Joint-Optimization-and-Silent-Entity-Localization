"""Current full 1..249 length-pool, ten-round policy-search implementation."""
from pathlib import Path
import copy, csv, hashlib, json, statistics, shutil, time
import numpy as np
import torch
import run_experiment_4rx as base
HERE=Path(__file__).resolve().parent
CONTROL=HERE
prepare_candidates=lambda cfg,length,picks:None

def save_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def input_tensor(lengths,M,device='cpu'):
    # Legacy input scaling, adapted from 1..500 to the current 1..M samples.
    return torch.tensor([[(l-1)/(M-1)] for l in lengths],dtype=torch.float32,device=device)


def predict(policy,l,M):
    if l==M:return 1
    policy.eval()
    with torch.no_grad():fraction=float(policy(input_tensor([l],M,next(policy.parameters()).device)).item())
    # Canonical rounding is retained for all new comparisons, unlike legacy floor.
    return base.decode_start(fraction,l,M)


def initialize(cfg):
    out=base.get_output_dir(cfg);M=cfg['data']['M']
    # Evaluation lengths have the same sampling probability as every other length.
    # M has only one feasible start (1), so it needs no searched training label.
    pool=list(range(1,M))
    rng=np.random.default_rng(cfg['search']['schedule_seed'])
    schedule=rng.choice(pool,size=(cfg['search']['iters'],cfg['search']['q_per_round']),replace=True).tolist()
    data={'schedule':schedule,'heldout_lengths':[],'possible_training_lengths':pool,'unique_scheduled_lengths':sorted(set(sum(schedule,[]))),
          'sampling':'all integer lengths 1..M-1; no evaluation-length exclusions; sampling with replacement',
          'schedule_rng_independent_of_candidate_rng':True,'test_loaded':False}
    path=out/'schedule.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=data:raise ValueError('Sampling schedule changed')
    save_json(path,data);return schedule


def score(*args,**kw):
    raise RuntimeError("Install the isolated asynchronous score adapter first.")


def train_policy(cfg,policy,optimizer,best,epochs):
    M=cfg['data']['M'];lengths=sorted(best)
    x=input_tensor(lengths,M,next(policy.parameters()).device)
    y=torch.tensor([[(best[l]['start']-1)/(M-l)] for l in lengths],dtype=torch.float32,device=x.device)
    policy.train()
    # Tiny cumulative label set: full batches, patience and best-state retention.
    lowest=float('inf');stale=0;state=None;opt_state=None
    for step in range(epochs+1):
        loss=((policy(x)-y)**2).mean();value=float(loss.detach())
        if value < lowest:
            significant=lowest-value>cfg['search']['policy_min_delta']
            lowest=value;state=copy.deepcopy(policy.state_dict());opt_state=copy.deepcopy(optimizer.state_dict())
            stale=0 if significant else stale+1
        else:stale+=1
        if step==epochs or stale>=cfg['search']['policy_patience']:break
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    policy.load_state_dict(state);optimizer.load_state_dict(opt_state)
    print('POLICY_FIT',{'steps':step,'best_label_mse':lowest},flush=True)
    return lowest


def candidates(cfg,method,rng,center,l,round_idx,visited):
    M=cfg['data']['M'];s=cfg['search'];remaining=[v for v in range(1,M-l+2) if v not in visited]
    budget=min(s['k_perturb']+s['k_local']+s['k_random'],len(remaining))
    if method=='guided':
        groups=base.propose_candidate_groups(rng,center,l,M,s['k_perturb'],s['k_local'],s['k_random'],round_idx,s['perturb_scale'],s['decay'],s['local_radius'])
        chosen=[]
        for _,v in groups:
            if v not in visited and v not in chosen:chosen.append(v)
        chosen=chosen[:budget]
        if len(chosen)<budget:
            pool=[v for v in remaining if v not in chosen]
            chosen.extend(int(v) for v in rng.choice(pool,size=budget-len(chosen),replace=False))
        return sorted(chosen)
    return sorted(int(v) for v in rng.choice(remaining,size=budget,replace=False))


def search(cfg,method):
    out=base.get_output_dir(cfg);schedule=initialize(cfg);M=cfg['data']['M'];device=base.choose_device(cfg)
    train,valid,test,_,_=base.load_splits(cfg,final=False,include_test=False);assert test is None
    scaler=base.TargetScaler(train);base.configure_seed(cfg,cfg['training']['seed'])
    policy=base.PolicyNet(cfg['policy']['hidden_dims']).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=cfg['search']['policy_lr'])
    rng=np.random.default_rng(cfg['search'][method+'_candidate_seed'])
    best={};visited={};rows=[];trace=[];completed=0
    state_path=out/f'{method}_state.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location=device,weights_only=False)
        if saved['signature']!=cfg['_signature']:raise ValueError('Search signature changed')
        completed=saved['completed_rounds'];best=saved['best'];visited=saved['visited'];rows=saved['rows'];trace=saved['trace'];rng.bit_generator.state=saved['rng']
        policy.load_state_dict(saved['policy']);optimizer.load_state_dict(saved['optimizer'])
    for round_idx in range(completed,len(schedule)):
        for slot,l in enumerate(schedule[round_idx]):
            seen=visited.setdefault(l,set());center=predict(policy,l,M) if method=='guided' else None
            picks=candidates(cfg,method,rng,center,l,round_idx,seen)
            prepare_candidates(cfg,l,picks)
            for start in picks:
                item,elapsed,cached=score(cfg,l,start,train,valid,scaler,device);seen.add(start)
                rows.append({**item,'method':method,'round':round_idx+1,'slot':slot,'center':center,'current_training_seconds':elapsed,'current_cache_hit':int(cached)})
                if l not in best or item['mean_error_um']<best[l]['mean_error_um']:best[l]=item
                print('SCORE',method,round_idx+1,l,start,item['mean_error_um'],flush=True)
        loss=train_policy(cfg,policy,optimizer,best,cfg['search']['policy_epochs_per_round']) if method=='guided' else None
        trace.append({'round':round_idx+1,'unique_label_lengths':len(best),'cumulative_candidate_evaluations':len(rows),'label_mse':loss})
        temporary=state_path.with_suffix('.tmp')
        torch.save({'signature':cfg['_signature'],'completed_rounds':round_idx+1,'best':best,'visited':visited,'rows':rows,'trace':trace,'rng':rng.bit_generator.state,'policy':policy.state_dict(),'optimizer':optimizer.state_dict()},temporary);temporary.replace(state_path)
        base.write_csv(out/f'{method}_scores.csv',rows);base.write_csv(out/f'{method}_trace.csv',trace)
        print('ROUND_COMPLETE',method,trace[-1],flush=True)
    base.write_csv(out/f'{method}_best.csv',[best[l] for l in sorted(best)])
    if method=='guided':
        torch.save({k:v.detach().cpu() for k,v in policy.state_dict().items()},out/'policy_raw.pt')
        # Separate label-only convergence diagnostic, not silently substituted for legacy training.
        refit(cfg,best,out/'policy_raw.pt')


def refit(cfg,best,initial_path):
    out=base.get_output_dir(cfg);M=cfg['data']['M'];model=base.PolicyNet(cfg['policy']['hidden_dims'])
    model.load_state_dict(torch.load(initial_path,map_location='cpu',weights_only=True));opt=torch.optim.Adam(model.parameters(),lr=cfg['refit']['lr'])
    lengths=sorted(best);x=input_tensor(lengths,M)
    y=torch.tensor([[(best[l]['start']-1)/(M-l)] for l in lengths],dtype=torch.float32)
    history=[];lowest=float('inf');state=None
    for step in range(cfg['refit']['steps']+1):
        loss=((model(x)-y)**2).mean()
        if float(loss.detach())<lowest:lowest=float(loss.detach());state=copy.deepcopy(model.state_dict())
        if step%100==0 or step==cfg['refit']['steps']:
            with torch.no_grad():starts=[base.decode_start(float(v),l,M) for l,v in zip(lengths,model(x).flatten())]
            history.append({'step':step,'label_mse':float(loss.detach()),'start_mae':statistics.mean(abs(s-best[l]['start']) for s,l in zip(starts,lengths))})
        if step==cfg['refit']['steps']:break
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    torch.save(state,out/'policy_refitted.pt');base.write_csv(out/'policy_refit_curve.csv',history)


