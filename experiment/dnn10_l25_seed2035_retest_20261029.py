"""One independent l25 DNN repeat; preserve every existing paper result."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,traceback
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

OUT=ROOT/'dnn10_l25_seed2035_earlystop_retest_20261102'
ORIGINAL=ROOT/'dnn10_width100_optimized_tests_seed2034_20261029/l25'
PROTOCOL=ROOT/'dnn10_l25_seed2035_retest_protocol_20261029.json'

def verify():
    p=load(PROTOCOL)
    assert p['early_stopping'] is True and p['patience']==10
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for path,digest in p['frozen_files_sha256'].items():assert sha(path)==digest,path
    f=load(p['frozen_optimizer_record'])
    assert int(f['starts']['25'])==110 and f['optimizer_sha256']==p['optimizer_sha256']
    return p

def progress(phase,**kwargs):
    r=dict(phase=phase,pid=os.getpid(),at_utc=utc(),model='dnn',length=25,start=110,seed=2035,batch_size=256,**kwargs)
    save(OUT/'status.json',r);print(json.dumps(r),flush=True)

def run():
    import numpy as np,torch
    import experiment_workers as common
    import dnn10_width100_cnn50 as architecture
    p=verify();common.configure();base=common.base;architecture.install_model(base)
    cfg=copy.deepcopy(load(ORIGINAL/'training_config.json'))
    cfg['training']['seed']=2035
    cfg['training']['final_patience']=10
    cfg['paths']['data_dir']=str(ROOT/'data')
    cfg['paths']['checkpoint_cache_dir']=str(OUT/'checkpoints')
    cfg['paths']['output_dir']=str(OUT/'model_output')
    assert cfg['models']['dnn'][0]['hidden_dims']==[100]*10
    assert cfg['training']['batch_size']==cfg['training']['long_window_batch_size']==256
    assert cfg['training']['loss']=='mse_physical_um2' and cfg['training']['lr']==.001
    assert cfg['training']['deterministic'] is False and cfg['data']['sample_seed']==2025
    save(OUT/'training_config.json',cfg)
    progress('loading_data',max_epochs=60,patience=10,early_stopping=True)
    tr,va=common.full_data(cfg);scaler=base.TargetScaler(tr)
    assert len(tr['x'])==410640 and len(va['x'])==25500
    td,vd=[base.make_dataset(d,scaler,25,110,cfg) for d in [tr,va]]
    rows=[];curve=OUT/'curve.csv'
    def observe(epoch,training,validation):
        row=dict(epoch=epoch,train_mse_um2=training,val_mse_um2=validation)
        rows.append(row);write_csv(curve,rows);progress('training',**row)
    base.TRAINING_OBSERVER=observe
    state,loss,epoch,seconds,cached=base.train_or_load(cfg,'dnn10_l25_independent_repeat_earlystop','dnn',25,110,2035,td,vd,torch.device('cuda'),60,10)
    assert not cached,'This independent repeat must train fresh'
    assert 1<=len(rows)<=60 and [int(r['epoch']) for r in rows]==list(range(1,len(rows)+1))
    checkpoint=base.checkpoint_path(cfg,'dnn10_l25_independent_repeat_earlystop','dnn',25,110,2035,60,10)
    assert torch.load(checkpoint,map_location='cpu',weights_only=False)['seed']==2035
    progress('evaluating',best_validation_epoch=epoch)
    val_metrics,_,_,_=common.infer(cfg,state,'dnn',25,110,va,scaler)
    with np.load(ROOT/'data/independent_test.npz',allow_pickle=False) as z:test={k:z[k].copy() for k in ['X','x','y']}
    metrics,pred,truth,errors=common.infer(cfg,state,'dnn',25,110,test,scaler)
    prediction=OUT/'predictions.npz';np.savez_compressed(prediction,predictions=pred,truth=truth,errors_um=errors)
    net=base.build_localizer(cfg,'dnn',25);count=sum(w.numel() for w in net.parameters() if w.requires_grad)
    assert count==103402
    result=dict(architecture='ten_layers_width100',model='dnn',condition='optimized',length=25,start=110,seed=2035,
        batch_size=256,lr=.001,epochs=len(rows),max_epochs=60,patience=10,early_stopping=True,
        early_stopped=len(rows)<60,training_loss='mse_physical_um2',deterministic=False,amp=cfg['training']['amp'],
        n_test=len(errors),parameters=count,optimizer_retrained=False,optimizer_sha256=p['optimizer_sha256'],
        test_sha256=sha(ROOT/'data/independent_test.npz'),checkpoint_source=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        predictions_sha256=sha(prediction),curve_sha256=sha(curve),training_seconds=seconds,reused_checkpoint=cached,
        best_validation_epoch=epoch,best_validation_mse_um2=loss,validation_mean_error_um=val_metrics['mean_error_um'],
        final_train_mse_um2=float(rows[-1]['train_mse_um2']),final_validation_mse_um2=float(rows[-1]['val_mse_um2']),
        accuracy08=float((errors<.8).mean()),accuracy08_percent=100*float((errors<.8).mean()),
        original_result_source=str(ORIGINAL/'final_test.csv'),paper_result_replaced=False,**metrics)
    save(OUT/'result.json',result);write_csv(OUT/'final_test.csv',[result])
    previous=dict(read_csv(ORIGINAL/'final_test.csv')[0]);previous['accuracy08_percent']=p['previous_accuracy08_percent']
    write_csv(OUT/'comparison.csv',[dict(previous,repeat_label='original_seed2034'),dict(result,repeat_label='new_seed2035')])
    verify();progress('complete',result=result,original_results_unchanged=True)

def launch():
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'launch.json'
    if target.exists():
        old=load(target)
        if alive(old['identity']):print(json.dumps(dict(already_running=True,**old)));return
        raise RuntimeError('An earlier independent repeat exists; inspect instead of relaunching')
    assert not (OUT/'checkpoints').exists()
    with (OUT/'worker.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    r=dict(identity=identity(child.pid),started_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL),
        model='dnn',length=25,start=110,seed=2035,batch_size=256,max_epochs=60,patience=10,early_stopping=True,
        optimizer_retrained=False,replace_paper_automatically=False)
    save(target,r);save(ROOT/'CURRENT_DNN10_L25_RETEST_20261029.json',r);print(json.dumps(r),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except Exception:
            progress('failed',error=traceback.format_exc());raise
