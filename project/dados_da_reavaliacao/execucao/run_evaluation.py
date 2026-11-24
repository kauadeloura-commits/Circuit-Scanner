from pathlib import Path
import json,subprocess,sys,time,hashlib,datetime
ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'cases_frozen.json').read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def start():
    ready=ROOT/'worker_ready.json'
    if ready.exists():ready.unlink()
    logfile=open(ROOT/f'worker_{int(time.time())}.log','w',encoding='utf8')
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py')],stdin=subprocess.PIPE,stdout=logfile,stderr=subprocess.STDOUT,text=True,encoding='utf8',cwd=ROOT)
    started=time.monotonic()
    while not ready.exists():
        if p.poll() is not None:raise RuntimeError(f'Worker startup failed; see {logfile.name}')
        if time.monotonic()-started>180:p.kill();raise RuntimeError('Worker import timeout')
        time.sleep(.5)
    return p,logfile
if __name__=='__main__':
    p,logfile=start()
    try:
        for i,c in enumerate(cases):
            out=ROOT/'results'/c['id'];out.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists():continue
            assert sha(Path(c['input']))==c['input_sha256'],'Frozen input changed'
            if p.poll() is not None:p,logfile=start()
            p.stdin.write(json.dumps(c,ensure_ascii=False)+'\n');p.stdin.flush();begin=time.monotonic()
            while not (out/'result.json').exists():
                if p.poll() is not None or time.monotonic()-begin>300:
                    timeout=p.poll() is None
                    if timeout:p.kill();p.wait()
                    dump(out/'result.json',dict(case_id=c['id'],status='timeout' if timeout else 'worker_crash',total_seconds=time.monotonic()-begin,manual_edits=0,prediction_overrides=0,input_sha256=c['input_sha256']))
                    break
                time.sleep(.5)
            result=json.loads((out/'result.json').read_text(encoding='utf8'))
            print(f'{i+1}/{len(cases)} {c["id"]} {result["status"]} {result["total_seconds"]:.2f}s',flush=True)
            dump(ROOT/'progress_summary.json',dict(completed=i+1,total=len(cases),last_case=c['id'],last_status=result['status'],updated_at=datetime.datetime.now().astimezone().isoformat()))
    finally:
        if p.poll() is None:p.stdin.close();p.wait(timeout=30)
        logfile.close()
