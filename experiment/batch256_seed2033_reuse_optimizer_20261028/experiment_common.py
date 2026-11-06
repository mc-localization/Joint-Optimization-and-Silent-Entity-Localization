"""Atomic records and identities for this isolated experiment."""
from pathlib import Path
import csv,datetime,hashlib,json,os
ROOT=Path(__file__).resolve().parent

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid):
    try:
        a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=a[19],state=a[0])
    except FileNotFoundError:return None
def alive(who):
    now=identity(who['pid']) if who else None
    return bool(now and now['state']!='Z' and now['start_ticks']==who['start_ticks'])
def report(key,phase,**kw):
    row=dict(phase=phase,pid=os.getpid(),at_utc=utc(),**kw)
    save(ROOT/'progress'/f'{key}.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
def mark_done(key,**kw):save(ROOT/'done'/f'{key}.json',dict(completed_at_utc=utc(),**kw))
def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(set(r) for r in rows))))
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cpu_quota():
    text=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    return len(os.sched_getaffinity(0)) if text[0]=='max' else max(1,int(text[0])//int(text[1]))
