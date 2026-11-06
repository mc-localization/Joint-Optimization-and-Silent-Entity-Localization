"""Draw Fig. 5 from the currently accepted main metrics on the paper canvas."""
from pathlib import Path
import argparse,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
ROOT=Path(__file__).resolve().parents[1]

def draw(output):
    with (ROOT/'reference/main_metrics.csv').open(encoding='utf-8',newline='') as f:
        rows={(r['model'],int(r['length'])):r for r in csv.DictReader(f)}
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'mathtext.fontset':'stix'})
    fig,ax=plt.subplots(figsize=(4,3))
    fig.subplots_adjust(left=.18,right=.97,bottom=.18,top=.95)
    ax.set(xlabel=r'Detection window length $l$',ylabel=r'Mean localization error ($\mu$m)')
    ax.set_axisbelow(True);ax.grid(which='major',linestyle=':',linewidth=.5,color='#b8b8b8')
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():spine.set_linewidth(.8)
    lengths=list(range(25,251,25))
    for model,label,color,marker in [('cnn','CNN','#d62728','o'),('dnn','DNN','#1f77b4','s'),
            ('gru','GRU','#d8a700','^'),('transformer','Transformer','#2ca02c','D'),('mle','MLE','#7f7f7f','v')]:
        ax.plot(lengths,[float(rows[model,l]['mean_error_um']) for l in lengths],label=label,
            color=color,marker=marker,linestyle='--' if model=='mle' else '-',linewidth=1.25,
            markersize=3.8,markerfacecolor='white',markeredgewidth=1.,clip_on=False)
    ax.set(xlim=(0,250),ylim=(0,1.5),xticks=range(0,251,25),yticks=[0,.5,1,1.5])
    legend=ax.legend(loc='upper right',ncol=1,framealpha=1,edgecolor='black',handlelength=1.8)
    legend.get_frame().set_linewidth(.6)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=600);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    draw(p.parse_args().output)
