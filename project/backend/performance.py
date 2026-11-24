"""Content-checked project caches and timing; CPU-only, no vision shortcuts."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import lru_cache,wraps
from inspect import signature,unwrap
from pathlib import Path
from threading import RLock
from time import perf_counter,sleep
from time import time_ns
from uuid import uuid4
import hashlib,json,sys,os,ast,logging,shutil
from copy import deepcopy
from importlib.metadata import version,PackageNotFoundError

DEFER_INITIAL_GEOMETRY=ContextVar('defer_initial_geometry',default=False)
_IO_LOCK=RLock()
_LOCKS={}
_TIMING_STACK=ContextVar('processing_timing_stack',default=())
_LOGGER=logging.getLogger(__name__)

def project_lock(folder):
    with _IO_LOCK:return _LOCKS.setdefault(str(folder.resolve()),RLock())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

@lru_cache(maxsize=8192)
def _file_hash(path,stamp):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def file_hash(path):
    path=Path(path)
    try:
        s=path.stat()
        # Windows can give successive equal-length writes the same timestamp.
        # Verify recently saved files directly; stable assets keep fast hashes.
        if time_ns()-max(s.st_mtime_ns,s.st_ctime_ns)<1_000_000_000:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        # The bank contains more than 1,000 files. A smaller LRU evicted every
        # entry on each ordered sweep, hashing all assets again at every step.
        # stat already follows links; a lexical absolute key also avoids an
        # extra Windows filesystem round trip per unchanged asset.
        return _file_hash(os.path.abspath(path),(s.st_mtime_ns,s.st_ctime_ns,s.st_size))
    except FileNotFoundError:return None

@lru_cache(maxsize=1)
def runtime_versions():
    result={}
    for name in ('numpy','opencv-python','torch','ultralytics','easyocr'):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result

@lru_cache(maxsize=8)
def _analysis_main(path,content_hash):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and
           (n.name.startswith('_') or n.name in {'processar_projeto','reconhecer_componentes','ler_referencias_e_valores'})]
    return hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in nodes).encode()).hexdigest()


def runtime_signature(base,*,include_assets=True,analysis=False):
    base=Path(base)
    ui={'guided_page.py','production_interface.py','production_page.py','simulation_visual.py'}
    files=[p for p in (base/'backend').glob('*.py') if not p.name.startswith('test_') and not(analysis and p.name in ui)]
    for folder in ('models','backend/handwriting_references','terminal_training') if include_assets else ():
        directory=base/folder
        files.extend(p for p in directory.glob('*') if p.is_file() and
                     (p.suffix in {'.json','.npz'} or folder=='backend/handwriting_references' and p.suffix in {'.png','.jpg','.onnx','.ts'} or
                      p.name in {'circuit_scanner_yolo.pt','component_consensus.pt','terminal_keypoints.pt','all_component_terminal_keypoints.pt'}))
    if include_assets:
        for folder in ('terminal_keypoints/raw/meta','terminal_training_all_components/raw'):
            files.extend((base/folder).rglob('*.json'))
    if include_assets:files.append(base/'dataset/classes.json')
    hashes={p.relative_to(base).as_posix():file_hash(p) for p in sorted(set(files))}
    if analysis and 'backend/main.py' in hashes:
        hashes['backend/main.py']=_analysis_main(str(base/'backend/main.py'),hashes['backend/main.py'])
    command=next((Path(p) for p in (os.environ.get('TESSERACT_CMD','').strip().strip('"'),shutil.which('tesseract'),r'C:\Program Files\Tesseract-OCR\tesseract.exe',r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe') if p and Path(p).is_file()),None)
    printed_reader=dict(binary=file_hash(command),language=file_hash(command.parent/'tessdata/eng.traineddata')) if command else None
    return digest(dict(python=sys.version,libraries=runtime_versions(),disable_easyocr=os.environ.get('CIRCUIT_SCANNER_DISABLE_EASYOCR'),files=hashes,printed_reader=printed_reader))


def _shared_root(base):return Path(base)/'.analysis-cache'


def shared_restore(base,stage,key,folder,outputs,*,project_id,stored_filename):
    """Restore only checksum-verified automatic results for identical inputs."""
    root=_shared_root(base);prefix=stage+'_'+key
    with _IO_LOCK:
        entry=read_json(root/(prefix+'.cache.json'))
        hashes=entry.get('outputs')
        if entry.get('key')!=key or entry.get('stage')!=stage or not isinstance(hashes,dict) or set(hashes)!=set(outputs):return False
        try:payloads={name:(root/(prefix+'_'+name)).read_bytes() for name in outputs}
        except OSError:return False
        if any(hashlib.sha256(data).hexdigest()!=hashes[name] for name,data in payloads.items()):return False
        old_file=entry.get('stored_filename')
        def rebound(value):
            if isinstance(value,dict):
                return {k:project_id if k=='project_id' else stored_filename if k in {'source_image','source_filename'} and v==old_file else rebound(v) for k,v in value.items()}
            if isinstance(value,list):return [rebound(v) for v in value]
            return value
        try:
            payloads={name:json.dumps(rebound(json.loads(data)),ensure_ascii=False,indent=2,allow_nan=False).encode('utf8') if name.endswith('.json') else data for name,data in payloads.items()}
        except (ValueError,TypeError):return False
        for name,data in payloads.items():
            target=Path(folder)/name;temporary=target.with_name(target.name+'.'+uuid4().hex+'.tmp')
            temporary.write_bytes(data);temporary.replace(target)
        os.utime(root/(prefix+'.cache.json'),None)
        return True


def shared_store(base,stage,key,folder,outputs,*,stored_filename):
    root=_shared_root(base);root.mkdir(parents=True,exist_ok=True);prefix=stage+'_'+key
    with _IO_LOCK:
        hashes={}
        for name in outputs:
            original=Path(folder)/name
            if not original.is_file():return
            data=original.read_bytes();temporary=root/(prefix+'_'+name+'.'+uuid4().hex+'.tmp')
            temporary.write_bytes(data);temporary.replace(root/(prefix+'_'+name));hashes[name]=hashlib.sha256(data).hexdigest()
        write_json(root/(prefix+'.cache.json'),dict(stage=stage,key=key,stored_filename=stored_filename,outputs=hashes))
        # Cache files are owned artifacts; never delete uploaded/project files.
        records=sorted(root.glob('*.cache.json'),key=lambda p:p.stat().st_mtime_ns)
        for old in records[:-16]:
            old_entry=read_json(old);old_prefix=old.name[:-len('.cache.json')]
            old_outputs=old_entry.get('outputs',{})
            for name in old_outputs if isinstance(old_outputs,dict) else ():
                if Path(name).name!=name:continue
                target=(root/(old_prefix+'_'+name)).resolve()
                if target.parent==root.resolve():target.unlink(missing_ok=True)
            old.unlink(missing_ok=True)

@lru_cache(maxsize=64)
def _document(path,stamp):return json.loads(Path(path).read_text(encoding='utf8'))


def read_document(path):
    try:
        path=Path(path);s=path.stat()
        if time_ns()-max(s.st_mtime_ns,s.st_ctime_ns)<1_000_000_000:
            return json.loads(path.read_text(encoding='utf8'))
        return deepcopy(_document(str(path.resolve()),(s.st_mtime_ns,s.st_ctime_ns,s.st_size)))
    except (OSError,ValueError):return None


def read_json(path):
    value=read_document(path)
    return value if isinstance(value,dict) else {}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        # On Windows a reader/antivirus can briefly hold the old file without
        # FILE_SHARE_DELETE. Preserve its valid bytes and retry the atomic
        # replacement; never truncate the destination to work around the lock.
        for attempt,delay in enumerate((0,.02,.04,.08,.16,.25)):
            if delay:sleep(delay)
            try:tmp.replace(path);break
            except PermissionError as error:
                if getattr(error,'winerror',None) not in {None,5,32,33} or attempt==5:raise
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:_LOGGER.debug('Temporary JSON cleanup deferred: %s',tmp)

def cache_valid(folder,stage,key,outputs):
    entry=read_json(Path(folder)/'processing_cache.json').get(stage,{})
    if not isinstance(entry,dict) or not isinstance(entry.get('outputs'),dict):return False
    return entry.get('key')==key and bool(outputs) and all(
        entry.get('outputs',{}).get(name) is not None and _output_hash(folder,stage,name)==entry['outputs'][name]
        for name in outputs)

def cache_store(folder,stage,key,outputs):
    hashes={name:_output_hash(folder,stage,name) for name in outputs}
    if any(v is None for v in hashes.values()):return
    with _IO_LOCK:
        path=Path(folder)/'processing_cache.json';cache=read_json(path)
        cache[stage]=dict(key=key,outputs=hashes)
        try:write_json(path,cache)
        except OSError:_LOGGER.warning('Could not persist optional processing cache: %s',path,exc_info=True)

def _output_hash(folder,stage,name):
    if stage=='geometry' and name=='wires.json':
        data=read_document(Path(folder)/name)
        if not isinstance(data,dict):return None
        # Netlists and simulation are derived from this same immutable graph.
        # Revising them must not force another expensive image analysis.
        return digest({k:v for k,v in data.items() if k not in
                       {'stage10','stage11','stage12','stage10_11_warning','electrical_pipeline_order'}})
    return file_hash(Path(folder)/name)

def record_timing(folder,stage,seconds,*,reused=False,success=True,details=None,work_seconds=None):
    entry=dict(stage=stage,duration_seconds=round(seconds,4),reused=bool(reused),success=bool(success))
    if work_seconds is not None:entry['work_seconds']=round(max(0,work_seconds),4)
    if details:entry['details']=details
    with _IO_LOCK:
        path=Path(folder)/'performance.json';data=read_json(path)
        if not isinstance(data.get('stages'),dict):data['stages']={}
        if not isinstance(data.get('recent_runs'),list):data['recent_runs']=[]
        data.setdefault('stages',{})[stage]=entry
        data['recent_runs']=(data.get('recent_runs',[])+[entry])[-40:]
        data['schema_version']='performance-1.0'
        try:write_json(path,data)
        except OSError:
            # Metrics are auxiliary: a completed recognition/save must remain
            # successful even when its timing file is held by another process.
            _LOGGER.warning('Could not persist optional processing timings: %s',path,exc_info=True)
            entry['timing_persisted']=False
    return entry

@contextmanager
def timed(folder,stage,**details):
    started=perf_counter();success=False;result=dict(reused=False,details=details)
    parents=_TIMING_STACK.get();node=dict(child_seconds=0.)
    token=_TIMING_STACK.set(parents+(node,))
    try:
        yield result;success=True
    finally:
        elapsed=perf_counter()-started;_TIMING_STACK.reset(token)
        if parents:parents[-1]['child_seconds']+=elapsed
        if not result.get('skip_persist'):
            record_timing(folder,stage,elapsed,reused=result['reused'],success=success,
                          details=result.get('details'),work_seconds=elapsed-node['child_seconds'])

def cached_project_stage(stage):
    """Decorates an endpoint while preserving its FastAPI parameter signature."""
    outputs={'process':('processed.png','processing.json'),
             'detect':('detections.png','detections.json'),
             'ocr':('ocr.png','components.json','detections.png','detections.json')}[stage]
    def decorate(fn):
        sig=signature(fn);g=unwrap(fn).__globals__
        @wraps(fn)
        def wrapped(*args,**kwargs):
            bound=sig.bind(*args,**kwargs);bound.apply_defaults();params=dict(bound.arguments)
            pid=params.pop('project_id');force=params.pop('force',False)
            project=g['buscar_projeto'](pid)
            if not project:return fn(*args,**kwargs)
            folder=g['PROJECTS_DIR']/pid
            def key():
                inputs={'original':file_hash(folder/project['stored_filename']),'drawing_settings':file_hash(folder/'drawing_settings.json')}
                if stage=='detect' and params.get('source')=='processed':inputs['processed']=file_hash(folder/'processed.png')
                # Formatting/newlines and the project filename do not alter
                # detection geometry. Restore writes atomic UTF-8 bytes; the
                # normal Windows writer can use CRLF for the same JSON.
                if stage=='ocr':
                    detection_input=read_json(folder/'detections.json')
                    inputs['detections']=digest({k:v for k,v in detection_input.items() if k not in {'project_id','source_image','source_filename','elapsed_seconds','performance'}})
                return digest(dict(runtime=runtime_signature(g['BASE_DIR'],include_assets=stage!='process',analysis=True),inputs=inputs,options=params,
                                   deferred=DEFER_INITIAL_GEOMETRY.get() if stage=='detect' else False))
            with project_lock(folder),timed(folder,stage) as measurement,project_lock(_shared_root(g['BASE_DIR'])/(stage+'_'+(input_key:=key()))):
                shareable=os.environ.get('SCANNER_SHARED_ANALYSIS_CACHE','1')!='0' and stage in {'detect','ocr'} and not project.get('detections_info' if stage=='detect' else 'ocr_info')
                shared=False
                local=not force and cache_valid(folder,stage,input_key,outputs)
                if not force and not local and shareable:
                    shared=shared_restore(g['BASE_DIR'],stage,input_key,folder,outputs,project_id=pid,stored_filename=project['stored_filename'])
                    if shared:
                        detections=read_json(folder/'detections.json')
                        g['atualizar_deteccoes'](project_id=pid,status='reconhecido',detection_filename='detections.png',detections_info=json.dumps(detections,ensure_ascii=False))
                        if stage=='ocr':g['atualizar_ocr'](project_id=pid,status='ocr_concluido',ocr_filename='ocr.png',ocr_info=json.dumps(read_json(folder/'components.json'),ensure_ascii=False))
                        cache_store(folder,stage,key() if stage=='ocr' else input_key,outputs)
                if local or shared:
                    measurement['reused']=True
                    geometry_warning=None
                    if stage=='ocr':
                        current=g['buscar_projeto'](pid);components=read_json(folder/'components.json')
                        try:g['_gerar_terminais_automaticamente'](pid,current,components,trigger='ocr_cache')
                        except Exception as error:geometry_warning=str(error)
                    response=g['preparar_resposta_projeto'](g['buscar_projeto'](pid))
                    if shared and stage=='detect' and not DEFER_INITIAL_GEOMETRY.get():
                        try:g['_gerar_terminais_automaticamente'](pid,g['buscar_projeto'](pid),g['_components_info_from_detections'](read_json(folder/'detections.json')),trigger='detection_shared_cache')
                        except Exception as error:geometry_warning=str(error)
                    response.update(mensagem='Resultados válidos reaproveitados.',processing_reused=True,shared_processing_reused=shared)
                    if geometry_warning:response['terminal_detection_warning']=geometry_warning
                    return response
                result=fn(*args,**kwargs)
                # OCR can refine detection boxes; key the saved result to that final input.
                cache_store(folder,stage,key() if stage=='ocr' else input_key,outputs)
                if shareable:
                    try:shared_store(g['BASE_DIR'],stage,input_key,folder,outputs,stored_filename=project['stored_filename'])
                    except OSError:_LOGGER.warning('Could not persist optional shared processing cache',exc_info=True)
                if isinstance(result,dict):result['processing_reused']=False
                return result
        return wrapped
    return decorate

def time_project_operation(stage):
    def decorate(fn):
        @wraps(fn)
        def wrapped(project_id,*args,**kwargs):
            folder=unwrap(fn).__globals__['PROJECTS_DIR']/project_id
            with project_lock(folder),timed(folder,stage) as measurement:
                try:result=fn(project_id,*args,**kwargs)
                except Exception as error:
                    # Rejected edits must leave all project artifacts intact.
                    # Keep timing failures for internal errors, but avoid a
                    # write merely because validation rejected user input.
                    if getattr(error,'status_code',None) in {400,404,409,422}:measurement['skip_persist']=True
                    raise
                measurement['reused']=bool(isinstance(result,dict) and result.get('processing_reused'))
                return result
        return wrapped
    return decorate

GEOMETRY_OUTPUTS=('wires.json','wires.png','wires_mask.png','wires_skeleton.png','wires_erased_mask.png',
                  'terminals.json','terminals.png','junctions.json','junctions.png')

def geometry_key(base,folder,project,components):
    from copy import deepcopy
    payload=deepcopy(components)
    # Electrical values do not affect any image geometry or ink mask.
    for c in payload.get('components',[]):
        for name in ('value','model','electrical_parameters','parameter_policy','parameter_pending','review_required','review_reasons'):
            c.pop(name,None)
        c.pop('ocr_tokens',None)
        if isinstance(c.get('reference'),dict):c['reference']={'text':c['reference'].get('text')}
    payload={k:v for k,v in payload.items() if k in ('components',)}
    ocr=read_json(Path(folder)/'components.json') or read_json_value(project.get('ocr_info'))
    manual=[dict(id=c.get('component_id'),bbox=c.get('bbox'),terminals=c.get('terminals'))
            for c in read_json(Path(folder)/'terminals.json').get('components',[])
            if c.get('terminal_detection',{}).get('status')=='manual_ground_truth']
    return digest(dict(runtime=runtime_signature(base,analysis=True),image=file_hash(Path(folder)/project['stored_filename']),
                       drawing_settings=file_hash(Path(folder)/'drawing_settings.json'),components=payload,text_boxes=[t.get('bbox') for t in ocr.get('tokens',[])],
                       detections=read_json_value(project.get('detections_info')).get('detections',[]),
                       manual=manual,corrections=file_hash(Path(folder)/'terminal_corrections.json')))

def read_json_value(value):
    try:return json.loads(value) if isinstance(value,str) else value or {}
    except (ValueError,TypeError):return {}
