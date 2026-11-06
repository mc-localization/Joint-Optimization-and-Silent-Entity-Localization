"""Check exact archived source hashes and recompute the 50 paper result rows."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def run():
    sources=json.loads((ROOT/'reference/source_hashes.json').read_text(encoding='utf-8'))
    for item in sources:assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    with (ROOT/'reference/main_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        p=ROOT/'reference/raw'/Path(row['source']).parent/'predictions.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['predictions_sha256']
        with np.load(p,allow_pickle=False) as z:
            errors=np.linalg.norm(z['predictions']-z['truth'],axis=1)
            assert len(errors)==2000 and np.allclose(errors,z['errors_um'],rtol=1e-10,atol=1e-10)
        for value,key in [(errors.mean(),'mean_error_um'),(np.median(errors),'median_error_um'),(100*(errors<.8).mean(),'accuracy_percent')]:assert abs(float(value)-float(row[key]))<1e-9,(row['model'],row['length'],key)
    return dict(exact_source_files=len(sources),main_result_rows=len(rows),metrics_checked=len(rows)*3,passed=True)

if __name__=='__main__':print(json.dumps(run(),indent=2))
