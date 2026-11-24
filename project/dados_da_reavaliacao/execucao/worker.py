"""Unmodified application handlers in a separate data directory. No prediction hooks."""
from pathlib import Path
import sys, os, json, time, asyncio, traceback, importlib.metadata, platform
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
os.environ['CIRCUIT_SCANNER_DB']=str(ROOT/'evaluation.sqlite3')
sys.path.insert(0,str(RUNTIME))
t0=time.perf_counter()
from backend import main
from backend.performance import DEFER_INITIAL_GEOMETRY
from backend.cpu_inference import start_model_warmup
from fastapi import UploadFile,HTTPException
import backend.cpu_inference as cpu
start_model_warmup()
if cpu._WARMUP_THREAD:cpu._WARMUP_THREAD.join()
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')
import torch,cv2
packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','opencv-python','numpy','fastapi','pillow','scipy']}
write(ROOT/'worker_ready.json',dict(startup_seconds=time.perf_counter()-t0,python=sys.version,executable=sys.executable,platform=platform.platform(),logical_cpus=os.cpu_count(),cpu=platform.processor(),torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),cuda_available=torch.cuda.is_available(),packages=packages,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
print('READY',flush=True)
def run(case):
    cid=case['id'];out=ROOT/'results'/cid;out.mkdir(parents=True,exist_ok=True)
    result=dict(case_id=cid,input_sha256=case['input_sha256'],started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stages=[],manual_edits=0,prediction_overrides=0)
    write(out/'case_started.json',result)
    begin=time.perf_counter();pid=None;folder=None
    def stage(name,fn):
        start=time.perf_counter()
        try:
            response=fn();row=dict(stage=name,ok=True,seconds=time.perf_counter()-start)
            write(out/f'{name}_response.json',response)
        except Exception as e:
            row=dict(stage=name,ok=False,seconds=time.perf_counter()-start,error_type=type(e).__name__,http_status=getattr(e,'status_code',None),error=getattr(e,'detail',str(e)))
            write(out/f'{name}_error.json',dict(row,traceback=traceback.format_exc()))
            response=None
        result['stages'].append(row);write(out/'progress.json',result);print(cid,name,round(row['seconds'],2),row['ok'],flush=True)
        return response
    async def create():
        with open(case['input'],'rb') as f:
            return await main.salvar_novo_projeto(UploadFile(filename=Path(case['input']).name,file=f),name='EVAL '+cid)
    response=stage('upload',lambda:asyncio.run(create()))
    if response:
        pid=response['project_id'];folder=main.PROJECTS_DIR/pid;result['project_id']=pid;write(out/'case_started.json',result)
        stage('process',lambda:main.processar_projeto(pid))
        def detect():
            token=DEFER_INITIAL_GEOMETRY.set(True)
            try:return main.reconhecer_componentes(pid)
            finally:DEFER_INITIAL_GEOMETRY.reset(token)
        stage('detect',detect)
        det=folder/'detections.json'
        if det.exists():write(out/'detections_before_ocr.json',json.loads(det.read_text(encoding='utf8')))
        stage('ocr_and_automatic_geometry',lambda:main.ler_referencias_e_valores(pid))
        if case['cohort']=='whole':
            stage('junctions',lambda:main.detectar_juncoes_projeto(pid))
            # Raw diagnostic graph only; does not resolve or confirm pending reviews.
            stage('connectivity_diagnostic',lambda:main.construir_redes_projeto(pid))
            stage('circuit_diagnostic',lambda:main.gerar_circuito_projeto(pid))
            stage('simulation_gate',lambda:main._circuito_confirmado(pid))
            circuit_file=folder/'circuit.json'
            if circuit_file.exists():
                c=json.loads(circuit_file.read_text(encoding='utf8'))
                stage('review_issues_readonly',lambda:main.review_issues(c))
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in ['.json','.png','.txt']:
                import shutil
                shutil.copy2(p,out/p.name)
    result.update(total_seconds=time.perf_counter()-begin,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='completed',project_id=pid)
    write(out/'result.json',result)
    print('DONE',cid,round(result['total_seconds'],2),flush=True)
for line in sys.stdin:
    try:run(json.loads(line))
    except Exception as e:
        print('HARNESS_ERROR',repr(e),traceback.format_exc(),flush=True)
