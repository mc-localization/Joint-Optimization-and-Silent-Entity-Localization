"""Render PNG paper figures from committed raw experimental outputs."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator
from experiment_common import ROOT,load,save,sha,read_csv,write_csv

MODELS=['cnn','dnn','gru','transformer','mle']
LABELS=['CNN','DNN','GRU','Transformer','MLE']
COLORS=['#d62728','#1f77b4','#d8a700','#2ca02c','#7f7f7f']
MARKERS=['o','s','^','D','v']
def axes(xlabel,ylabel):
    available={x.name for x in font_manager.fontManager.ttflist}
    font=next((f for f in ['Times New Roman','Liberation Serif','DejaVu Serif'] if f in available),'serif')
    plt.rcParams.update({'font.family':font,'font.size':10,'legend.fontsize':8,
        'mathtext.fontset':'stix','xtick.labelsize':9,'ytick.labelsize':9})
    fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.18,right=.97,bottom=.20,top=.96)
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.set_axisbelow(True)
    ax.grid(which='major',linestyle=':',linewidth=.5,color='#b8b8b8')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both',direction='in',top=True,right=True)
    return fig,ax
def line(ax,x,y,index,label,markers=True):
    ax.plot(x,y,color=COLORS[index],marker=MARKERS[index] if markers else None,
        linestyle='--' if index==4 else '-',label=label,linewidth=1.25,
        markersize=3.5,markerfacecolor='white',markeredgewidth=1.,clip_on=False)
def finish(fig,ax,out,name):
    legend=ax.legend(loc='upper right',ncol=1,framealpha=1,edgecolor='black',handlelength=1.6)
    legend.get_frame().set_linewidth(.6)
    out.mkdir(parents=True,exist_ok=True);path=out/(name+'.png')
    fig.savefig(path,dpi=600);plt.close(fig);return dict(file=str(path),sha256=sha(path))
def baseline(length):return read_csv(ROOT/'results/cnn_optimized'/f'l{length}/final_test.csv')[0]
def parameter(case,length):return read_csv(ROOT/'parameter_experiments'/case/'results'/f'l{length}/final_test.csv')[0]
def error_ceiling(values):return max(.5,math.ceil(max(values)*2)/2)

def run(job):
    figure=job['figure'];out=ROOT/'paper_figures'/figure;outputs=[]
    if figure=='fig7':
        histories={m:read_csv(out/m/'curve.csv') for m in MODELS[:4]}
        sources={m:sha(out/m/'curve.csv') for m in histories}
        for m,rows in histories.items():assert [int(r['epoch']) for r in rows]==list(range(1,61))
        for field,name,ylabel,smooth in [('train_mse_um2','fig7_training_loss','Training loss ($\\mu$m$^2$)',False),
                ('val_mse_um2','fig8_validation_loss','Validation loss ($\\mu$m$^2$)',True)]:
            fig,ax=axes('Epoch',ylabel);all_values=[]
            for i,m in enumerate(MODELS[:4]):
                ys=[float(r[field]) for r in histories[m]]
                if smooth:
                    for k in range(1,len(ys)):ys[k]=.2*ys[k]+.8*ys[k-1]
                all_values+=ys;line(ax,range(1,61),ys,i,LABELS[i],False)
            ax.set(xlim=(0,60),ylim=(0,error_ceiling(all_values)),xticks=range(0,61,10))
            outputs.append(finish(fig,ax,out,name))
        save(out/'figure_manifest.json',dict(sources_sha256=sources,epochs=60,loss='mse_physical_um2',
            training_smoothing=False,validation_ema_alpha=.2,outputs=outputs))
        return
    if figure=='fig6':
        length=job['length'];thresholds=np.unique(np.r_[np.arange(0,2.0001,.05),.25,.4])
        fig,ax=axes(r'Localization error threshold $\chi$ ($\mu$m)','Localization success rate (%)')
        rows=[];sources={}
        for i,m in enumerate(MODELS):
            p=ROOT/'results'/f'{m}_optimized'/f'l{length}'/'predictions.npz';sources[m]=sha(p)
            with np.load(p,allow_pickle=False) as z:
                errors=np.linalg.norm(z['predictions']-z['truth'],axis=1)
                assert len(errors)==2000 and np.allclose(errors,z['errors_um'],rtol=1e-10,atol=1e-10)
            values=[100*float((errors<t).mean()) for t in thresholds]
            line(ax,thresholds,values,i,LABELS[i],False)
            rows += [dict(model=m,length=length,threshold_um=float(t),success_percent=v) for t,v in zip(thresholds,values)]
        ax.set(xlim=(0,2),ylim=(0,100),xticks=np.arange(0,2.01,.5),yticks=range(0,101,20))
        outputs.append(finish(fig,ax,out,'fig6_accuracy_threshold_l100'))
        write_csv(out/'threshold_results.csv',rows)
        save(out/'figure_manifest.json',dict(length=length,predictions_sha256=sources,outputs=outputs));return
    lengths=list(range(25,251,25));rows=[]
    if figure in ['fig11','fig13']:
        fig,ax=axes(r'Detection window length $l$',r'Mean localization error ($\mu$m)');values=[]
        cases=[('q10000','$Q=10000$'),(None,'$Q=20000$'),('q30000','$Q=30000$')] if figure=='fig11' else [('rx2m250','Two receivers'),(None,'Four receivers')]
        for i,(case,label) in enumerate(cases):
            records=[parameter(case,l) if case else baseline(l) for l in lengths]
            ys=[float(r['mean_error_um']) for r in records];values+=ys;rows+=records
            line(ax,lengths,ys,i,label)
        ax.set(xlim=(0,250),xticks=range(0,251,50),ylim=(0,error_ceiling(values)))
    elif figure=='fig12':
        fig,ax=axes(r'Sampled sequence length $M$',r'Mean localization error ($\mu$m)');values=[]
        for i,duration in enumerate([2,3,4]):
            records=[baseline(int(250*duration/5)) if M==250 else parameter('m'+str(M),int(M*duration/5)) for M in [125,250,375,500]]
            ys=[float(r['mean_error_um']) for r in records];values+=ys;rows+=records
            line(ax,[125,250,375,500],ys,i,r'$T_{\mathrm{window}}='+str(duration)+r'$ s')
        ax.set(xlim=(0,500),xticks=[0,125,250,375,500],ylim=(0,error_ceiling(values)))
    else:raise ValueError(figure)
    write_csv(out/'source_results.csv',rows)
    outputs.append(finish(fig,ax,out,figure))
    save(out/'figure_manifest.json',dict(outputs=outputs,source_results_sha256=sha(out/'source_results.csv')))
