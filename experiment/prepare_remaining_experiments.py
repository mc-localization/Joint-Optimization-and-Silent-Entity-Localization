"""Keep completed measurements; resume only unmet original paper-suite jobs."""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
target=ROOT/'run_remaining_paper_experiments.py'
assert not target.exists()
s=(ROOT/'run_paper_suite.py').read_text(encoding='utf-8')
needle="    if job['kind'] in ['convergence','parameter_policy','parameter_search_fit','parameter_fit','parameter_evaluate','mle_supplement','paper_figure','barrier']:"
assert s.count(needle)==1
s=s.replace(needle,"    if job.get('case_id') in ['m125','m375','m500','rx2m250'] and job['kind'] in ['parameter_policy','parameter_search_fit','parameter_fit','parameter_evaluate']:\n        import remaining_parameter_workers;remaining_parameter_workers.run(job)\n    elif"+needle[len('    if'):])
needle="    for name,digest in suite['source_sha256'].items():assert sha(ROOT/name)==digest,name"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n    pending=load(ROOT/'remaining_experiments_protocol.json')\n    for name,digest in pending['source_sha256'].items():assert sha(ROOT/name)==digest,name")
ast.parse(s);target.write_text(s,encoding='utf-8',newline='\n')
protocol=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
    remaining_cases=['m125','m375','m500','rx2m250'],figures=[12,13],batch_size_all_lengths=256,
    search_localizer_seed=2026,formal_localizer_seed=2025,deterministic=False,
    condition_specific_start_optimizer_training=True,baseline_results_reused=True,
    accepted_main_comparison_source='Local user-confirmed per-length table; do not overwrite original baseline CSVs',
    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['remaining_parameter_workers.py',target.name]})
(ROOT/'remaining_experiments_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(protocol))
