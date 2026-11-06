"""Reuse matched 60-epoch records to compare CNN, new DNN, GRU and Transformer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiment_common import ROOT,load,save,sha,utc,identity,alive,read_csv,write_csv

LENGTHS=[50,75,150,200,250]
MODELS=['cnn','dnn','gru','transformer']
OUT=ROOT/'paper_figures/fig7_dnn10_seed2034'
TESTS=ROOT/'dnn10_width100_optimized_tests_seed2034_20261029'
PROTOCOL=ROOT/'fig7_dnn10_four_models_protocol.json'


def sources(length):
    return {m:TESTS/f'l{length}' if m=='dnn' else ROOT/'paper_figures/fig7_seed2034'/f'l{length}'/m for m in MODELS}


def render_length(length,folders,output):
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix',
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8})
    records={};provenance={}
    for model,folder in folders.items():
        result=load(folder/'result.json');rows=read_csv(folder/'curve.csv')
        assert result['seed']==2034 and result['batch_size']==256 and result['epochs']==60
        assert result['length']==length and result['early_stopping'] is False
        assert sha(folder/'curve.csv')==result['curve_sha256']
        assert [int(r['epoch']) for r in rows]==list(range(1,61))
        records[model]=rows
        provenance[model]=dict(curve_source=str(folder/'curve.csv'),curve_sha256=result['curve_sha256'],
            start=result['start'],checkpoint_sha256=result['checkpoint_sha256'])
    output.mkdir(parents=True,exist_ok=True)
    for model,rows in records.items():
        write_csv(output/(model+'_curve.csv'),rows)
    outputs={}
    for end,suffix in [(60,'training_loss'),(10,'training_loss_first10')]:
        fig,ax=plt.subplots(figsize=(4,3));fig.subplots_adjust(left=.18,right=.97,bottom=.20,top=.96)
        largest=0
        for model,color,label in zip(MODELS,['#d62728','#1f77b4','#d8a700','#2ca02c'],['CNN','DNN (10 layers)','GRU','Transformer']):
            ys=[float(r['train_mse_um2']) for r in records[model][:end]]
            assert all(math.isfinite(y) and y>=0 for y in ys);largest=max(largest,max(ys))
            ax.plot(range(1,end+1),ys,label=label,color=color,lw=1.25)
        upper=max(25,5*math.ceil(largest/5))
        ax.set(xlim=(0,end),ylim=(0,upper),xlabel='Epoch',ylabel=r'Training loss ($\mu$m$^2$)',
            xticks=list(range(0,end+1,10 if end==60 else 2)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.set_axisbelow(True);ax.grid(linestyle=':',linewidth=.5,color='#b8b8b8')
        legend=ax.legend(loc='upper right',framealpha=1,edgecolor='black');legend.get_frame().set_linewidth(.6)
        path=output/f'fig7_{suffix}_l{length}_seed2034.png'
        fig.savefig(path,dpi=600,bbox_inches='tight',pad_inches=.04);plt.close(fig)
        outputs[suffix]=dict(path=str(path),sha256=sha(path))
    manifest=dict(length=length,seed=2034,batch_size=256,lr=.001,epochs=60,training_reused=True,
        dnn_architecture='10 hidden layers, width100, dropout0.05, hidden BN',
        starts_are_model_specific=True,smoothing=False,provenance=provenance,outputs=outputs)
    save(output/'manifest.json',manifest)
    return manifest


def verify():
    import dnn10_optimized_tests
    dnn10_optimized_tests.verify()
    protocol=load(PROTOCOL)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for length in LENGTHS:
        for model,folder in sources(length).items():
            if model=='dnn':continue
            assert sha(folder/'curve.csv')==protocol['reference_curve_sha256'][str(length)][model]


def run():
    import fcntl
    verify();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            completed=[]
            for length in LENGTHS:
                target=OUT/f'l{length}'
                if (target/'manifest.json').exists():completed.append(length);continue
                if not (TESTS/f'l{length}/final_test.csv').exists():continue
                manifest=render_length(length,sources(length),target)
                save(target/'source_protocol.json',dict(protocol_sha256=sha(PROTOCOL)))
                completed.append(length)
            state=dict(phase='complete' if len(completed)==len(LENGTHS) else 'waiting_for_dnn_training',
                pid=os.getpid(),at_utc=utc(),completed_lengths=completed,lengths=LENGTHS,
                additional_gpu_training=False)
            save(OUT/'status.json',state)
            if len(completed)==len(LENGTHS):return
            source_state=load(TESTS/'status.json')
            if source_state.get('failed'):
                save(OUT/'status.json',dict(state,phase='blocked_by_training_error',failed=source_state['failed']));return
            time.sleep(30)


def launch():
    verify();OUT.mkdir(parents=True,exist_ok=True);p=OUT/'launch.json'
    if p.exists() and alive(load(p)['identity']):print(json.dumps(load(p)));return
    with (OUT/'controller.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    record=dict(identity=identity(child.pid),launched_at_utc=utc(),output_directory=str(OUT),protocol=str(PROTOCOL))
    save(p,record);save(ROOT/'CURRENT_FIG7_DNN10.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.launch:launch()
    elif args.run:run()
