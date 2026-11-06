"""Regenerate the baseline using the original priority-first simulation order."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--run',action='store_true',help='Actually start the full Monte Carlo generation')
    a=p.parse_args()
    if not a.run:p.error('Full generation is expensive; pass --run to start it')
    if a.workers<1:p.error('--workers must be positive')
    if a.workspace.exists():p.error('Use a new workspace; existing server experiment directories are not supported here')
    a.workspace.mkdir(parents=True);workspace=a.workspace.resolve()
    for n in ['deployment_protocol.json','candidate_position_plan.json','position_plan.npz']:shutil.copy2(ROOT/'experiment'/n,workspace/n)
    protocol=json.loads((workspace/'deployment_protocol.json').read_text(encoding='utf-8'))
    protocol['resources']['micro_generation_workers']=a.workers
    protocol['resources']['full_generation_workers']=a.workers
    (workspace/'deployment_protocol.json').write_text(json.dumps(protocol,indent=2))
    sys.path[:0]=[str(ROOT/'experiment'),str(ROOT/'experiment/runtime')]
    import experiment_common,priority_data
    experiment_common.ROOT=priority_data.ROOT=workspace
    priority_data.run()

if __name__=='__main__':main()
