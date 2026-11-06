"""Run the remaining parameter studies with batch 256 at every length."""
import copy
import paper_suite_workers as original
from experiment_common import ROOT,load,sha

NATIVE_CFG=original.cfg_for
CASES={'m125','m375','m500','rx2m250'}

def cfg_for(case,stage='final'):
    cfg=NATIVE_CFG(case,stage)
    if case in CASES:
        cfg=copy.deepcopy(cfg)
        cfg['training'].update(batch_size=256,long_window_batch_size=256)
    return cfg

def run(job):
    assert job['case_id'] in CASES
    assert job['kind'] in ['parameter_policy','parameter_search_fit','parameter_fit','parameter_evaluate']
    p=load(ROOT/'remaining_experiments_protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    original.cfg_for=cfg_for
    original.run(job)
