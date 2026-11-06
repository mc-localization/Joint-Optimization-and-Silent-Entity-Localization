"""Redraw result figures 5-13 from the accepted reference records."""
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
M=json.loads((ROOT/'reference/paper_manifest.json').read_text(encoding='utf-8'))
MODELS=['cnn','dnn','gru','transformer','mle']
LABELS=['CNN','DNN','GRU','Transformer','MLE']
COLORS=['#d62728','#1f77b4','#d6aa00','#2ca02c','#7f7f7f']
LENGTHS=list(range(25,251,25))
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
MAIN={(r['model'],int(r['length'])):r for r in read(ROOT/'reference/main_metrics.csv')}
def mainrow(model,length):return MAIN[model,length]
def rawpath(path):return ROOT/'reference/raw'/path.replace('${EXPERIMENT_ROOT}/','')
def parameter(case,length):return next(r for r in M['parameter_results'] if r['case_id']==case and int(r['length'])==length)
def accuracy(row):
    with np.load(rawpath(row['source']).parent/'predictions.npz',allow_pickle=False) as z:return 100*float((np.linalg.norm(z['predictions']-z['truth'],axis=1)<.8).mean())
def axes(xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(4,3),dpi=600);fig.subplots_adjust(left=.18,bottom=.20,right=.97,top=.97)
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.tick_params(direction='in',top=True,right=True)
    ax.grid(linestyle=':',alpha=.35);ax.set_axisbelow(True)
    return fig,ax
def finish(fig,ax,out,name,legend='upper right'):
    if ax.get_legend_handles_labels()[0]:ax.legend(loc=legend,ncol=1,framealpha=1,edgecolor='black',fontsize=7)
    fig.savefig(out/name,dpi=600);plt.close(fig)
def line(ax,x,y,index):ax.plot(x,y,label=LABELS[index],color=COLORS[index],linewidth=1.2)

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9})
    from plot_fig05 import draw
    draw(out/'fig05.png')
    fig,ax=axes(r'Localization error threshold $\chi$ ($\mu$m)','Localization accuracy (%)');thresholds=np.linspace(0,2,81)
    for i,m in enumerate(MODELS):
        row=next(r for r in M['selected_main_results'] if r['model']==m and int(r['length'])==100)
        with np.load(rawpath(row['source']).parent/'predictions.npz',allow_pickle=False) as z:err=np.linalg.norm(z['predictions']-z['truth'],axis=1)
        line(ax,thresholds,[100*float((err<t).mean()) for t in thresholds],i)
    ax.set(xlim=(0,2),ylim=(0,100));finish(fig,ax,out,'fig06.png')
    histories={r['model']:read(rawpath(r['path'])) for r in M['curve_sources']}
    for number,key,label in [(7,'train_mse_um2','Training loss'),(8,'val_mse_um2','Validation loss')]:
        fig,ax=axes('Epoch',label);values=[]
        for i,m in enumerate(MODELS[:4]):
            rows=histories[m];ys=[float(r[key]) for r in rows]
            if number==8:
                for k in range(1,len(ys)):ys[k]=.2*ys[k]+.8*ys[k-1]
            values.extend(ys);line(ax,[int(r['epoch']) for r in rows],ys,i)
        ax.set(xlim=(0,60),xticks=range(0,61,10))
        if number==7:ax.set_yscale('log');ax.set_ylim(10**math.floor(math.log10(min(values))),10**math.ceil(math.log10(max(values))))
        else:ax.set_ylim(0,math.ceil(max(values)*2)/2)
        finish(fig,ax,out,f'fig{number:02}.png')
    relative=[]
    for length in LENGTHS:
        fixed=float(M['controls']['fixed'][str(length)]['mean_error_um']);optimized=float(mainrow('cnn',length)['mean_error_um'])
        # At the full window, all strategies use the same accepted model at start 1.
        if length==250:fixed=optimized
        if length==225:
            fixed=float(M['fig9_paired_l225']['cnn_fig9_fixed']['mean_error_um']);optimized=float(M['fig9_paired_l225']['cnn_fig9_optimized']['mean_error_um'])
        relative.append(100*(fixed-optimized)/fixed)
    fig,ax=axes('Detection window length $l$','Relative error reduction (%)');ax.bar(LENGTHS,relative,width=15,color='#2879ad')
    ax.set(xlim=(0,250),ylim=(0,100),xticks=range(0,251,25));finish(fig,ax,out,'fig09.png')
    fig,ax=axes('Detection window length $l$',r'Mean localization error ($\mu$m)');ls=[50,100,150,200,250];x=np.arange(len(ls))
    for i,(name,label) in enumerate([('fixed','Fixed detection start time'),('optimized','Optimized detection start time'),('coarse','Coarse grid search')]):
        ys=[float(mainrow('cnn',l)['mean_error_um']) if name=='optimized' or l==250 else float(M['controls'][name][str(l)]['mean_error_um']) for l in ls]
        ax.bar(x+(i-1)*.24,ys,width=.24,label=label)
    ax.set_xticks(x,ls);ax.set_ylim(0,3.5);finish(fig,ax,out,'fig10.png')
    fig,ax=axes('Released molecule count $Q$','Localization accuracy (%)');x=np.arange(3)
    for i,l in enumerate([50,125,200]):
        ys=[accuracy(parameter('q10000',l)),float(mainrow('cnn',l)['accuracy_percent']),accuracy(parameter('q30000',l))]
        ax.bar(x+(i-1)*.24,ys,width=.24,label=f'$l={l}$')
    ax.set_xticks(x,[10000,20000,30000]);ax.set_ylim(0,100);finish(fig,ax,out,'fig11a.png','upper left')
    fig,ax=axes('Sampled sequence length $M$ per receiver',r'Mean localization error ($\mu$m)');x=np.arange(4)
    for i,seconds in enumerate([2,3,4]):
        ys=[float(mainrow('cnn',int(250*seconds/5))['mean_error_um']) if n==250 else float(parameter('m'+str(n),int(n*seconds/5))['mean_error_um']) for n in [125,250,375,500]]
        ax.bar(x+(i-1)*.24,ys,width=.24,label=r'$T_{\mathrm{detection}}='+str(seconds)+r'$ s')
    ax.set_xticks(x,[125,250,375,500]);ax.set_ylim(0,1);finish(fig,ax,out,'fig12.png')
    fig,ax=axes('Detection window length $l$','Localization accuracy (%)');ls=[50,100,150,200];x=np.arange(4)
    for i,n in enumerate([2,4]):
        ys=[accuracy(parameter('rx2m250',l)) if n==2 else float(mainrow('cnn',l)['accuracy_percent']) for l in ls]
        bars=ax.bar(x+(i-.5)*.32,ys,width=.32,label=rf'$\nu={n}$')
        ax.bar_label(bars,fmt='%.2f',padding=2,fontsize=6)
    ax.set_xticks(x,ls);ax.set_ylim(0,100);finish(fig,ax,out,'fig13a.png','upper left')
    return [str(p) for p in sorted(out.glob('fig*.png'))]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
    print(json.dumps(run(parser.parse_args().output),indent=2))
