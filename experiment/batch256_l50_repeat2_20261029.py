"""Fresh CNN/GRU l50 refits; replace the previous supplement points on completion."""
from pathlib import Path
from collections import OrderedDict
import argparse,copy,fcntl,json,os,shutil,subprocess,sys,time,traceback
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

OUT=ROOT/'batch256_l50_cnn_gru_repeat2_20261029'
PREVIOUS=ROOT/'batch256_revision18_retests_20261029'
PROTOCOL=ROOT/'batch256_l50_repeat2_protocol_20261029.json'

def verify():
    import batch256_revision18_parallel as original
    original.verify();p=load(PROTOCOL)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return p

def worker(model):
    import batch256_revision18_parallel as original
    import experiment_workers as common
    p=verify();common.configure();original.OUT=OUT
    job=next(j for j in p['jobs'] if j['model']==model)
    cfg=original.cfg_for(job)
    assert cfg['training']['seed']==2025 and cfg['training']['batch_size']==256
    assert Path(cfg['paths']['checkpoint_cache_dir']).is_relative_to(OUT)
    original.fit_job(model,job,OrderedDict())
    save(OUT/'done'/f'{model}.json',dict(phase='complete',at_utc=utc(),job=job['key']))

def atomic_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+f'.repeat2_{os.getpid()}.tmp')
    shutil.copy2(source,tmp);tmp.replace(target)

def install_results(p):
    # Verify every original point before changing either one.
    for path,digest in p['previous_point_sha256'].items():assert sha(path)==digest,path
    for job in p['jobs']:
        key=job['key'];alias=job['aliases'][0]['tag']
        assert (OUT/'fits'/key/'fit.json').exists() and (OUT/'results'/alias/'final_test.csv').exists()
    backup=OUT/'previous_round_backup'
    for job in p['jobs']:
        key=job['key'];alias=job['aliases'][0]['tag']
        shutil.copytree(PREVIOUS/'fits'/key,backup/'fits'/key)
        shutil.copytree(PREVIOUS/'results'/alias,backup/'results'/alias)
    for job in p['jobs']:
        key=job['key'];alias=job['aliases'][0]['tag']
        result=load(OUT/'results'/alias/'result.json')
        result.update(repeat_index=2,repeat_output_directory=str(OUT),
            replaces_previous_supplement_result=True,old_confirmed_paper_result_overwritten=False)
        save(OUT/'results'/alias/'result.json',result)
        write_csv(OUT/'results'/alias/'final_test.csv',[result])
        for name in ['training_config.json','curve.csv','fit.json']:
            atomic_copy(OUT/'fits'/key/name,PREVIOUS/'fits'/key/name)
        for name in ['predictions.npz','result.json','final_test.csv']:
            atomic_copy(OUT/'results'/alias/name,PREVIOUS/'results'/alias/name)
    # The CNN l50 learned-start fit is also a coarse-search candidate. Its score
    # above is replaced along with the main result; reconcile an already-finished
    # coarse selection, or let the pending selection read the new score naturally.
    coarse_key='coarse_select_and_test_l50'
    while True:
        state=load(PREVIOUS/'queue_state.json')
        if coarse_key not in state['running']:break
        time.sleep(3)
    if (PREVIOUS/'coarse/l50/selected_start.json').exists():
        import batch256_revision18_parallel as original
        import experiment_workers as common
        original.OUT=PREVIOUS;common.configure()
        coarse=next(j for j in load(ROOT/'batch256_revision18_protocol_20261029.json')['jobs'] if j['key']==coarse_key)
        original.coarse_job('repeat2_reconcile',coarse,OrderedDict())
    rows=[]
    for q in (PREVIOUS/'results').glob('*/final_test.csv'):rows+=read_csv(q)
    write_csv(PREVIOUS/'all_retest_results.csv',rows)
    after={}
    for job in p['jobs']:
        alias=job['aliases'][0]['tag'];row=read_csv(PREVIOUS/'results'/alias/'final_test.csv')[0]
        assert row['repeat_index']=='2' and row['batch_size']=='256'
        assert sha(PREVIOUS/'results'/alias/'predictions.npz')==row['predictions_sha256']
        after[job['model']]=row
    return after

def run():
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};attempts={};failed={}
        while True:
            for model,who in list(active.items()):
                if not alive(who):
                    del active[model]
                    if not (OUT/'done'/f'{model}.json').exists() and attempts.get(model,0)>=2:failed[model]='Worker exited twice'
            for model in ['cnn','gru']:
                if model in active or model in failed or (OUT/'done'/f'{model}.json').exists():continue
                (OUT/'logs').mkdir(exist_ok=True)
                with (OUT/'logs'/f'{model}.log').open('a',buffering=1) as log:
                    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--worker',model],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                attempts[model]=attempts.get(model,0)+1;active[model]=identity(child.pid)
            completed=[m for m in ['cnn','gru'] if (OUT/'done'/f'{m}.json').exists()]
            status=dict(phase='running',at_utc=utc(),pid=os.getpid(),completed=completed,active=active,failed=failed,
                length=50,seed=2025,batch_size=256,replace_previous_supplement=True)
            if len(completed)==2:
                save(OUT/'status.json',dict(status,phase='installing_replacements'))
                results=install_results(p)
                save(OUT/'status.json',dict(status,phase='complete',active={},replacement_installed=True,results=results));return
            if failed and not active:save(OUT/'status.json',dict(status,phase='failed'));return
            save(OUT/'status.json',status);time.sleep(3)

def launch():
    p=verify();OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'launch.json'
    if path.exists():
        old=load(path)
        if alive(old['identity']):print(json.dumps(dict(already_running=True,**old)));return
        assert not (OUT/'status.json').exists(),'Inspect existing repeat instead of restarting'
    assert not (OUT/'checkpoints').exists(),'Fresh repeat must use a new checkpoint directory'
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    info=dict(identity=identity(child.pid),started_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL),
        models=['cnn','gru'],length=50,starts=dict(cnn=109,gru=107),seed=2025,batch_size=256,
        replace_targets=[str(PREVIOUS/'results'/j['aliases'][0]['tag']) for j in p['jobs']])
    save(path,info);save(ROOT/'CURRENT_BATCH256_L50_REPEAT2_20261029.json',info);print(json.dumps(info),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker',choices=['cnn','gru']);args=parser.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except Exception:
            save(OUT/'status.json',dict(phase='failed',at_utc=utc(),error=traceback.format_exc()));raise
    elif args.worker:
        try:worker(args.worker)
        except Exception:
            save(OUT/'worker_errors'/f'{args.worker}_{os.getpid()}.json',dict(at_utc=utc(),error=traceback.format_exc()));raise
