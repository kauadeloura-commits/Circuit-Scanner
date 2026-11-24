"""Persistent guided workflow; computations advance, reviews require the user."""
import hashlib
import json
import math
from threading import Lock
from uuid import uuid4
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from backend.circuit_review import revision
from backend.circuit_connectivity import source_signature
from backend.guided_page import guided_page, editor_page
from backend.digital_twin import read_geometry,VERSION as LAYOUT_VERSION
from backend.schematic_views import diagram_for
from backend.circuitjs_export import export_circuitjs
from backend.visual_edit import saved_native
from backend.performance import DEFER_INITIAL_GEOMETRY,record_timing,read_json as performance_json
from time import perf_counter
from backend.processing_progress import REPORTER

PHASES = ('process','detect','components','ocr','terminals','models','junctions',
          'simulation','production','kicad','export','done')
AUTOMATIC = {'process','detect','ocr','kicad','export'}
EDITORS = {'components':'project.html','terminals':'terminal-trainer','models':'ocr-models-view',
           'junctions':'junction-review','circuit':'circuit-review-view',
           'simulation':'simulation-view','production':'production-view','done':'completion-view'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()

def register_guided(app, state):
    locks={}
    # Keep CPU recognition out of a long-lived HTTP request. A reload polls
    # this same job and cannot start a duplicate computation.
    workers=ThreadPoolExecutor(max_workers=1,thread_name_prefix='scanner-flow')
    jobs={}
    def lock(pid):return locks.setdefault(pid,Lock())
    def folder_project(pid):
        p=state.buscar_projeto(pid)
        if not p:raise HTTPException(404,'Projeto não encontrado.')
        return state.PROJECTS_DIR/pid,p
    @lru_cache(maxsize=128)
    def cached_inputs(folder,stored_filename,stamp):
        terms=state._json_projeto(folder,'terminals.json') or {}
        wires=state._json_projeto(folder,'wires.json') or {}
        circuit=state._json_projeto(folder,'circuit.json') or {}
        image=folder/stored_filename
        return dict(drawing_settings=state._json_projeto(folder,"drawing_settings.json"),image=hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None,
                    detections=digest(state._json_projeto(folder,'detections.json')),
                    models=digest(state._json_projeto(folder,'components.json')),
                    terminals=digest([dict(id=c.get('component_id') or c.get('detection_id'),type=c.get('type'),bbox=c.get('bbox'),
                                          points=[{k:t.get(k) for k in ('pin','position','electrical_role')} for t in c.get('terminals',[])])
                                      for c in terms.get('components',[])]),
                    wires=source_signature(wires,terms),circuit=revision(circuit) if circuit else None)
    def inputs(folder,p):
        # Reuse expensive hashes only while every relevant file is unchanged.
        paths=[folder/p['stored_filename']]+[folder/n for n in ('drawing_settings.json','detections.json','components.json','terminals.json','wires.json','circuit.json')]
        def stamp(path):
            try:
                s=path.stat();return (s.st_mtime_ns,s.st_ctime_ns,s.st_size)
            except FileNotFoundError:return None
        return cached_inputs(folder,p['stored_filename'],tuple(stamp(path) for path in paths)).copy()
    def write(folder,flow,p):
        flow['cursor']=uuid4().hex
        flow['inputs']=inputs(folder,p)
        path=folder/'guided_workflow.json';tmp=folder/'guided_workflow.tmp'
        tmp.write_text(json.dumps(flow,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)
        return flow
    def initial(folder,p):
        _,c=state._resultados_eletricos_atuais(p)
        if c:phase='simulation'
        elif (folder/'components.json').is_file() and (folder/'terminals.json').is_file():phase='terminals'
        elif (folder/'detections.json').is_file():phase='ocr'
        elif p.get('processed_filename') and (folder/p['processed_filename']).is_file():phase='detect'
        else:phase='process'
        return write(folder,dict(schema_version=1,phase=phase,error=None,options={'simulation':{'engine':'falstad'}}),p)
    def read(pid,repair=True):
        folder,p=folder_project(pid)
        flow=state._json_projeto(folder,'guided_workflow.json')
        guard=lock(pid)
        if flow and flow.get('phase')=='circuit':
            flow.update(phase='simulation',error=None);write(folder,flow,p)
        if flow and flow.get('phase')=='validation':
            flow.update(phase='done',error=None);write(folder,flow,p)
        if flow and flow.get('options',{}).get('simulation',{}).get('engine')!='falstad':
            flow.setdefault('options',{})['simulation']={'engine':'falstad'};write(folder,flow,p)
        if not flow or flow.get('phase') not in PHASES:
            if not repair or not guard.acquire(False):raise HTTPException(423,'O fluxo está sendo preparado em outra tela. Aguarde.')
            try:
                flow=state._json_projeto(folder,'guided_workflow.json')
                if not flow or flow.get('phase') not in PHASES:flow=initial(folder,p)
            finally:guard.release()
        if repair and guard.acquire(False):
            try:
                # Reload after acquiring: another tab may just have advanced.
                flow=state._json_projeto(folder,'guided_workflow.json')
                reconcile(folder,p,flow,pid)
            finally:guard.release()
        return folder,p,flow
    def reconcile(folder,p,flow,pid):
            if flow.get('files_entry') and flow['phase']=='production':return
            current=inputs(folder,p);before=flow.get('inputs',{});i=PHASES.index(flow['phase']);reset=None
            if before.get('image')!=current['image']:reset='process'
            elif i>PHASES.index('ocr') and before.get('detections')!=current['detections']:reset='ocr' if not (folder/'components.json').is_file() else 'terminals'
            elif i>PHASES.index('models') and before.get('models')!=current['models']:reset='models'
            elif i>PHASES.index('models') and before.get('terminals')!=current['terminals']:reset='terminals'
            elif i>PHASES.index('junctions') and before.get('wires')!=current['wires']:reset='junctions'
            elif i>PHASES.index('simulation') and before.get('circuit')!=current['circuit']:reset='simulation'
            if reset is None and i>=PHASES.index('production'):
                ready=flow.get('falstad') or {};path=folder/'circuitjs.txt'
                if (ready.get('source_revision')!=current['circuit'] or ready.get('layout_version')!=LAYOUT_VERSION
                    or not path.is_file() or ready.get('circuit_sha256')!=hashlib.sha256(path.read_text(encoding='utf8').encode()).hexdigest()):reset='simulation'
            if reset is None and flow['phase']=='done':
                from backend.project_export import export_is_current
                _,c=state._resultados_eletricos_atuais(p)
                if not c or not c.get('confirmed_by_user'):reset='simulation'
                elif not export_is_current(folder,c):reset='production'
            if reset:
                flow.update(phase=reset,error='Os dados foram alterados. Confira novamente esta revisão.',models_reviewed=None)
                write(folder,flow,p)
    def output(pid,flow):
        job=jobs.get(pid)
        progress=dict(job,elapsed_seconds=round(perf_counter()-job['started'],1)) if job else None
        if progress:progress.pop('started',None)
        return dict(project_id=pid,name=(state.buscar_projeto(pid) or {}).get('name',pid),phase=flow['phase'],cursor=flow['cursor'],error=flow.get('error'),
                    automatic=flow['phase'] in AUTOMATIC,busy=lock(pid).locked(),processing=progress,
                    options=flow.get('options',{}),editor=EDITORS.get(flow['phase']),performance=performance_json(state.PROJECTS_DIR/pid/'performance.json').get('stages',{}))
    def call(path,pid,payload=None):
        route=next(r for r in app.routes if getattr(r,'path',None)==path and 'POST' in getattr(r,'methods',set()))
        return route.endpoint(pid,payload) if payload is not None else route.endpoint(pid)
    def validate_terminals(folder):
        terms=state._json_projeto(folder,'terminals.json') or {}
        components=terms.get('components') or []
        if not components:raise HTTPException(422,'Nenhum componente com terminais. Confira o reconhecimento da imagem.')
        for c in components:
            schema=state._schema_for_component(c);points={t.get('pin'):t.get('position') or {} for t in c.get('terminals',[])}
            b=c.get('bbox') or {}
            for pin,_ in schema:
                q=points.get(pin,{})
                try:
                    x,y=float(q['x']),float(q['y'])
                    valid=math.isfinite(x) and math.isfinite(y) and float(b['x1'])<x<float(b['x2']) and float(b['y1'])<y<float(b['y2'])
                except (KeyError,ValueError,TypeError):valid=False
                if not valid:raise HTTPException(422,f"{c.get('component_id')}.{pin}: marque o terminal sobre o fio, dentro da caixa.")
    @app.get('/projects/{project_id}/guided/status')
    def status(project_id:str):
        _,_,flow=read(project_id);return output(project_id,flow)
    @app.post('/projects/{project_id}/guided/open-saved')
    def open_saved(project_id:str):
        guard=lock(project_id)
        if not guard.acquire(False):raise HTTPException(423,'Aguarde a etapa em execução.')
        try:
            folder,p=folder_project(project_id)
            flow=state._json_projeto(folder,'guided_workflow.json') or dict(schema_version=1,options={})
            flow.setdefault('options',{})['simulation']={'engine':'falstad'}
            flow.update(phase='production',files_entry=True,error=None)
            write(folder,flow,p);return dict(output(project_id,flow),busy=False)
        finally:guard.release()
    @app.post('/projects/{project_id}/guided/step')
    def step(project_id:str,payload:dict):
        # Nonblocking lock: a second tab/reload cannot duplicate a long detection.
        folder,p,flow=read(project_id)
        guard=lock(project_id)
        if not guard.acquire(False):raise HTTPException(423,'Esta etapa já está em execução. Aguarde sua conclusão.')
        return perform_step(project_id,payload,guard)
    @app.post('/projects/{project_id}/guided/run',status_code=202)
    def run(project_id:str,payload:dict):
        folder,p,flow=read(project_id)
        guard=lock(project_id)
        if not guard.acquire(False):return output(project_id,flow)
        try:
            if payload.get('cursor')!=flow['cursor'] or payload.get('phase')!=flow['phase']:
                raise HTTPException(409,'O fluxo avançou em outra tela. Atualize para continuar.')
            if flow['phase']=='done':raise HTTPException(409,'O fluxo já foi concluído.')
            job=dict(phase=flow['phase'],state='queued',started=perf_counter())
            jobs[project_id]=job
            def work():
                def progress(label,completed,total):
                    job.update(label=label,completed=completed,total=total)
                job['state']='running';token=REPORTER.set(progress)
                try:perform_step(project_id,payload,guard)
                except Exception:pass  # perform_step persists the error for status/retry.
                finally:
                    REPORTER.reset(token)
                    if jobs.get(project_id) is job:jobs.pop(project_id,None)
            workers.submit(work)
        except Exception:
            jobs.pop(project_id,None);guard.release();raise
        # Always poll once after acceptance, even if a tiny job already ended.
        # A stale phase here could otherwise submit the same stage twice.
        return dict(output(project_id,flow),busy=True)
    def perform_step(project_id,payload,guard):
        folder=state.PROJECTS_DIR/project_id;flow={}
        started=perf_counter();phase=None;completed=False
        try:
            folder,p,flow=read(project_id,repair=False)
            if payload.get('cursor')!=flow['cursor'] or payload.get('phase')!=flow['phase']:
                raise HTTPException(409,'O fluxo avançou em outra tela. Atualize para continuar.')
            phase=flow['phase'];next_phase=None
            if phase=='process':state.processar_projeto(project_id);next_phase='detect'
            elif phase=='detect':
                token=DEFER_INITIAL_GEOMETRY.set(True)
                try:state.reconhecer_componentes(project_id)
                finally:DEFER_INITIAL_GEOMETRY.reset(token)
                next_phase='ocr'
            elif phase=='components':next_phase='ocr'
            elif phase=='ocr':
                state.ler_referencias_e_valores(project_id)
                terms=state._json_projeto(folder,'terminals.json') or {}
                if not terms.get('components'):raise HTTPException(422,'Nenhum componente com terminais. Corrija o reconhecimento antes de continuar.')
                next_phase='terminals'
            elif phase=='terminals':
                validate_terminals(folder)
                next_phase='junctions' if flow.get('models_reviewed')==inputs(folder,p)['models'] else 'models'
            elif phase=='models':
                info=state._json_projeto(folder,'components.json')
                if not info or not info.get('components'):raise HTTPException(422,'Leia o OCR e preencha os valores e modelos primeiro.')
                flow['models_reviewed']=inputs(folder,p)['models']
                next_phase='junctions'
            elif phase=='junctions':
                wires=state._json_projeto(folder,'wires.json') or {}
                if not wires.get('stage9'):state.detectar_juncoes_projeto(project_id)
                wires=state._json_projeto(folder,'wires.json') or {}
                if (wires.get('stage9') or {}).get('review_count',0):raise HTTPException(422,'Resolva as junções pendentes antes de continuar.')
                # Keep valid human circuit edits when merely revisiting this step.
                connectivity,c=state._resultados_eletricos_atuais(state.buscar_projeto(project_id))
                if not connectivity:state.construir_redes_projeto(project_id)
                if not c:state.gerar_circuito_projeto(project_id)
                next_phase='simulation'
            elif phase=='simulation':
                options=payload.get('options') or {'engine':'falstad'}
                if options.get('engine','falstad')!='falstad':raise HTTPException(422,'O simulador disponível é Falstad.')
                _,c=state._circuito_confirmado(project_id)
                try:result=saved_native(state,folder,c) or export_circuitjs(c,diagram_for(folder,c))
                except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,'Falstad: '+str(error))
                (folder/'circuitjs.txt').write_text(result['circuit'],encoding='utf8')
                (folder/'circuitjs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
                flow['falstad']=dict(source_revision=revision(c),layout_version=LAYOUT_VERSION,
                    circuit_sha256=hashlib.sha256(result['circuit'].encode()).hexdigest(),import_ready=True)
                next_phase='production'
            elif phase=='production':
                # Browsing saved files never claims an unreviewed circuit is
                # ready. New files still require the reviewed circuit.
                if flow.get('files_entry'):
                    _,c=state._circuito_confirmado(project_id)
                    result=saved_native(state,folder,c) or export_circuitjs(c,diagram_for(folder,c))
                    (folder/'circuitjs.txt').write_text(result['circuit'],encoding='utf8')
                    flow['falstad']=dict(source_revision=revision(c),layout_version=LAYOUT_VERSION,
                        circuit_sha256=hashlib.sha256(result['circuit'].encode()).hexdigest(),import_ready=True)
                    flow.pop('files_entry',None)
                options=payload.get('options') or {}
                if set(options)-{'pcb','footprints','base_url'}:raise HTTPException(422,'Opções de arquivos inválidas.')
                flow.setdefault('options',{})['production']=options;next_phase='kicad'
            elif phase=='kicad':
                _,c=state._circuito_confirmado(project_id);opts=flow['options'].get('production',{})
                call('/projects/{project_id}/kicad',project_id,dict(base_revision=revision(c),pcb=opts.get('pcb',False),footprints=opts.get('footprints',{})))
                next_phase='export'
            elif phase=='export':
                _,c=state._circuito_confirmado(project_id)
                call('/projects/{project_id}/export',project_id,dict(base_revision=revision(c),base_url=flow['options'].get('production',{}).get('base_url')))
                next_phase='done'
            else:raise HTTPException(409,'O fluxo já foi concluído.')
            completed=True
            record_timing(folder,'flow_'+phase,perf_counter()-started,reused=False,details={'includes_substeps':True,**({'engine':flow.get('options',{}).get('simulation',{}).get('engine','falstad')} if phase=='simulation' else {})})
            flow.update(phase=next_phase,error=None)
            write(folder,flow,state.buscar_projeto(project_id))
            return dict(output(project_id,flow),busy=False)
        except HTTPException as e:
            if e.status_code!=409:
                flow['error']=str(e.detail);write(folder,flow,state.buscar_projeto(project_id))
            raise
        except Exception as e:
            flow['error']='Não foi possível concluir esta etapa: '+str(e);write(folder,flow,state.buscar_projeto(project_id))
            raise HTTPException(500,flow['error']) from e
        finally:
            if phase and not completed:record_timing(folder,'flow_'+phase,perf_counter()-started,success=False)
            guard.release()
    @app.post('/projects/{project_id}/guided/back')
    def back(project_id:str,payload:dict):
        folder,p,flow=read(project_id);guard=lock(project_id)
        if not guard.acquire(False):raise HTTPException(423,'Aguarde a etapa em execução.')
        try:
            if payload.get('cursor')!=flow['cursor']:raise HTTPException(409,'O fluxo mudou. Atualize a tela.')
            destination=payload.get('phase')
            if destination=='circuit':destination='simulation'
            if destination not in {'components','terminals','models','junctions','simulation','production'} or PHASES.index(destination)>PHASES.index(flow['phase']):raise HTTPException(422,'Revisão anterior inválida.')
            flow.pop('files_entry',None)
            flow.update(phase=destination,error=None);write(folder,flow,p);return dict(output(project_id,flow),busy=False)
        finally:guard.release()
    @app.get('/projects/{project_id}/guided/editor/{phase}',response_class=HTMLResponse)
    def editor(project_id:str,phase:str):
        folder_project(project_id)
        if phase not in EDITORS:raise HTTPException(404,'Revisão desconhecida.')
        if phase=='components':
            html=(state.BASE_DIR/'frontend/project.html').read_text(encoding='utf8')
            html=html.replace('href="css/style.css"','href="/guided-assets/css/style.css"').replace('src="js/','src="/guided-assets/js/')
            return editor_page(html,phase)
        path='/projects/{project_id}/'+EDITORS[phase]
        route=next(r for r in app.routes if getattr(r,'path',None)==path and 'GET' in getattr(r,'methods',set()))
        result=route.endpoint(project_id)
        html=result.body.decode('utf8') if isinstance(result,HTMLResponse) else result
        return editor_page(html,phase)
    @app.get('/guided-assets/{asset:path}')
    def asset(asset:str):
        if asset not in {'css/style.css','js/project.js','js/model_editor.js'}:raise HTTPException(404,'Arquivo não encontrado.')
        return FileResponse(state.BASE_DIR/'frontend'/asset)
    return guided_page
