"""Dependencies and scheduling priorities for the requested paper figures."""
from experiment_common import ROOT,load,save,sha,mark_done
LENGTHS=list(range(25,251,25))
MODELS=['cnn','dnn','gru','transformer']

def extend_static(jobs,add):
    for model in MODELS:
        add('convergence60_'+model,'convergence',['data_generate','policy_'+model],
            priority=-20,gpu_gib=2.5,memory_gib=7.,model=model)
    add('figure_fig7','paper_figure',['convergence60_'+m for m in MODELS],
        resource='cpu_light',priority=-20,gpu_gib=0,memory_gib=1.,figure='fig7')
    add('mle_supplement25','mle_supplement',['figure_fig7'],resource='cpu_light',
        priority=0,gpu_gib=0,memory_gib=2.)
    expected=['figure_fig7','mle_supplement25']
    expected += [f'eval_{m}_optimized_l{l}' for m in MODELS for l in LENGTHS]
    expected += [f'eval_cnn_{role}_l{l}' for role in ['fixed','coarse'] for l in LENGTHS]
    add('main_supplement_complete','barrier',expected,resource='cpu_light',priority=0,gpu_gib=0,memory_gib=.2)
    add('figure_fig6','paper_figure',['main_supplement_complete'],resource='cpu_light',
        priority=30,gpu_gib=0,memory_gib=1.,figure='fig6',length=100)
    cases=load(ROOT/'paper_suite_protocol.json')['parameter_cases']
    for case in cases:
        add('parameter_policy_'+case,'parameter_policy',['main_supplement_complete','dataset_ready_'+case],
            resource='control',priority=30,gpu_gib=.5,memory_gib=1.,model='cnn',case_id=case)
    groups={'fig11':['q10000','q30000'],'fig12':['m125','m375','m500'],'fig13':['rx2m250']}
    for figure,selected in groups.items():
        deps=[f'parameter_eval_{c}_l{l}' for c in selected for l in cases[c]['lengths']]
        add('figure_'+figure,'paper_figure',deps+['main_supplement_complete'],resource='cpu_light',
            priority=30,gpu_gib=0,memory_gib=1.,figure=figure)

def reconcile(jobs):
    cases=load(ROOT/'paper_suite_protocol.json')['parameter_cases']
    for case,spec in cases.items():
        directory=ROOT/spec['dataset_directory'];ready=directory/'readiness.json'
        key='dataset_ready_'+case
        if ready.exists() and not (ROOT/'done'/f'{key}.json').exists():
            record=load(ready)
            assert record['passed'] and record['signature']==spec['generation_signature']
            assert record['features']==spec['M']
            assert sha(directory/'dataset_metadata.json')==record['metadata_sha256']
            mark_done(key,case_id=case,readiness_sha256=sha(ready))
        frozen_path=ROOT/'parameter_experiments'/case/'search/frozen_optimizer.json'
        if not (ROOT/'done'/f'parameter_policy_{case}.json').exists():continue
        frozen=load(frozen_path);assert sha(frozen['optimizer_path'])==frozen['optimizer_sha256']
        for length in spec['lengths']:
            start=int(frozen['starts'][str(length)])
            fit=f'parameter_fit_{case}_l{length}_s{start}'
            jobs.setdefault(fit,dict(key=fit,kind='parameter_fit',deps=['parameter_policy_'+case],
                resource='gpu',priority=30,gpu_gib=2.5,memory_gib=9.,model='cnn',
                case_id=case,length=length,start=start))
            key=f'parameter_eval_{case}_l{length}'
            jobs.setdefault(key,dict(key=key,kind='parameter_evaluate',deps=[fit],resource='gpu',
                priority=29,gpu_gib=2.,memory_gib=2.,model='cnn',case_id=case,length=length))
    # GRU candidate fits must remain eligible because its frozen optimizer is a
    # prerequisite for the fourth Fig7 convergence run. All formal supplements wait.
    for job in jobs.values():
        if job['kind'] in ['formal_fit','evaluate','coarse_select']:
            if 'figure_fig7' not in job['deps']:job['deps'].append('figure_fig7')

def accept_request(job):
    if job['kind']!='parameter_search_fit':return False
    case=job['case_id'];spec=load(ROOT/'paper_suite_protocol.json')['parameter_cases'][case]
    assert job['model']=='cnn' and 1<=job['length']<spec['M']
    assert 1<=job['start']<=spec['M']-job['length']+1
    assert job['key']==f"parameter_search_{case}_l{job['length']}_s{job['start']}"
    return True

def rank(job):
    if job['kind']=='convergence' or job['key']=='figure_fig7':group=0
    elif job.get('model')=='gru' and job['kind'] in ['policy','search_fit']:group=1
    elif job['kind'].startswith('parameter_') or job['kind']=='paper_figure':group=3
    else:group=2
    evaluation_first=0 if job['kind'] in ['evaluate','parameter_evaluate'] else 1
    return group,evaluation_first,job['priority'],job['key']
