"""Single additional GRU l100 training repeat, with isolated results."""
from pathlib import Path
from collections import OrderedDict
import argparse,copy,json,os,subprocess,sys,traceback
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

OUT=ROOT/'gru_l100_seed2026_retest_20261029'
ORIGINAL=ROOT/'batch256_revision18_retests_20261029'
PROTOCOL=ROOT/'gru_l100_seed2026_retest_protocol_20261029.json'
MODEL='gru';LENGTH=100;START=72;SEED=2026

def verify():
    p=load(PROTOCOL)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for path,digest in p['frozen_files_sha256'].items():assert sha(path)==digest,path
    f=load(ROOT/'search/gru/frozen_optimizer.json')
    assert int(f['starts']['100'])==START and f['optimizer_sha256']==p['optimizer_sha256']
    return p

def progress(phase,**kwargs):
    r=dict(phase=phase,pid=os.getpid(),at_utc=utc(),model=MODEL,length=LENGTH,start=START,seed=SEED,batch_size=256,**kwargs)
    save(OUT/'status.json',r);print(json.dumps(r),flush=True)

def run():
    import numpy as np,torch
    import experiment_workers as common
    import batch256_revision18_parallel as original
    p=verify();common.configure();base=common.base
    cfg=copy.deepcopy(load(ORIGINAL/'fits/fit_base_gru_l100_s72/training_config.json'))
    cfg['training']['seed']=SEED
    cfg['paths']['checkpoint_cache_dir']=str(OUT/'checkpoints')
    cfg['paths']['output_dir']=str(OUT/'model_output')
    assert cfg['training']['batch_size']==cfg['training']['long_window_batch_size']==256
    assert cfg['training']['loss']=='mse_physical_um2' and cfg['training']['lr']==.001
    assert cfg['training']['deterministic'] is False and cfg['data']['sample_seed']==2025
    assert cfg['models']['gru'][0]['hidden_dim']==100 and cfg['models']['gru'][0]['num_layers']==2
    save(OUT/'training_config.json',cfg);progress('loading_data',max_epochs=60,patience=10)
    job=dict(case_id='base',model=MODEL,length=LENGTH,start=START)
    tr,va,test,scaler,test_sha=original.dataset(job,cfg,OrderedDict())
    assert len(tr['x'])==410640 and len(va['x'])==25500 and len(test['x'])==2000
    assert test_sha==p['test_sha256']
    td,vd=[base.make_dataset(d,scaler,LENGTH,START,cfg) for d in [tr,va]]
    rows=[];curve=OUT/'curve.csv'
    def observe(epoch,training,validation):
        row=dict(epoch=epoch,train_mse_um2=training,val_mse_um2=validation)
        rows.append(row);write_csv(curve,rows);progress('training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,seconds,cached=base.train_or_load(cfg,'gru_l100_independent_repeat',MODEL,LENGTH,START,SEED,td,vd,torch.device('cuda'),60,10)
    assert not cached,'Additional repeat must train fresh'
    checkpoint=base.checkpoint_path(cfg,'gru_l100_independent_repeat',MODEL,LENGTH,START,SEED,60,10)
    assert torch.load(checkpoint,map_location='cpu',weights_only=False)['seed']==SEED
    progress('evaluating',best_validation_epoch=epoch,epochs_run=len(rows))
    val_metrics,_,_,_=common.infer(cfg,state,MODEL,LENGTH,START,va,scaler)
    metrics,pred,truth,errors=common.infer(cfg,state,MODEL,LENGTH,START,test,scaler)
    metrics=original.metric_threshold(metrics,errors)
    predictions=OUT/'predictions.npz';np.savez_compressed(predictions,predictions=pred,truth=truth,errors_um=errors)
    result=dict(model=MODEL,condition='optimized',case_id='base',length=LENGTH,start=START,seed=SEED,
        batch_size=256,lr=.001,max_epochs=60,patience=10,epochs_run=len(rows),training_loss='mse_physical_um2',
        deterministic=False,amp=cfg['training']['amp'],optimizer_retrained=False,optimizer_sha256=p['optimizer_sha256'],
        best_epoch=epoch,best_validation_mse_um2=loss,validation_mean_error_um=val_metrics['mean_error_um'],
        n_test=len(errors),test_sha256=test_sha,checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        predictions_sha256=sha(predictions),curve_sha256=sha(curve),training_seconds=seconds,reused_checkpoint=cached,
        accuracy08_percent=100*float((errors<.8).mean()),paper_result_replaced=False,**metrics)
    save(OUT/'result.json',result);write_csv(OUT/'final_test.csv',[result])
    previous=read_csv(ORIGINAL/'results/base_gru_optimized_l100/final_test.csv')[0]
    write_csv(OUT/'comparison.csv',[dict(previous,repeat_label='previous_seed2025'),dict(result,repeat_label='new_seed2026')])
    verify();progress('complete',result=result,previous_results_unchanged=True)

def launch():
    p=verify();OUT.mkdir(parents=True,exist_ok=True);target=OUT/'launch.json'
    if target.exists():
        old=load(target)
        if alive(old['identity']):print(json.dumps(dict(already_running=True,**old)));return
        raise RuntimeError('A previous attempt exists; inspect before restarting')
    assert not (OUT/'checkpoints').exists()
    with (OUT/'worker.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    r=dict(identity=identity(child.pid),started_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL),model=MODEL,
        length=LENGTH,start=START,seed=SEED,batch_size=256,max_epochs=60,patience=10,
        optimizer_retrained=False,replace_paper_automatically=False)
    save(target,r);save(ROOT/'CURRENT_GRU_L100_RETEST_20261029.json',r);print(json.dumps(r),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except Exception:
            progress('failed',error=traceback.format_exc());raise
