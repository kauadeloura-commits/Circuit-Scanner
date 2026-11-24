"""API integration for the final production stages."""
import hashlib
import html
import json
import re
import secrets
from pathlib import Path
from fastapi import HTTPException,Request
from fastapi.responses import FileResponse,HTMLResponse
from backend.circuit_review import revision
from backend.kicad_export import ARTIFACTS,PRESETS,write_kicad,export_options
from backend.project_export import EXPORT_ARTIFACTS,generate_export,export_is_current
from backend.production_page import production_page
from backend.digital_twin import read_geometry, make_diagram, diagram_svg, VERSION
from backend.circuitjs_export import export_circuitjs,preview_circuitjs
from fastapi.responses import Response
from backend.schematic_views import diagram_for
from backend.performance import file_hash
from backend.pcb_router import routing_is_current

def register_production(app,state):
    from backend.visual_edit import register_visual,saved_native
    register_visual(app,state)
    def circuit(project_id):return state._circuito_para_revisao(project_id)
    def kicad_current(project_id,c):
        folder=state.PROJECTS_DIR/project_id;manifest=state._json_projeto(folder,'kicad.json')
        if not manifest or manifest.get('source_revision')!=revision(c) or manifest.get('layout_version')!=VERSION or not routing_is_current(manifest):return None
        for name,digest in manifest.get('hashes',{}).items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or not (folder/name).is_file():return None
            if file_hash(folder/name)!=digest:return None
        return manifest
    @app.get('/projects/{project_id}/production')
    def production(project_id:str):
        project=state.buscar_projeto(project_id)
        if not project:raise HTTPException(404,'Projeto não encontrado.')
        try:project,c=circuit(project_id)
        except HTTPException as error:
            if error.status_code not in {400,404,409,422}:raise
            return dict(revision=None,confirmed=False,components=[],kicad=None,export=None,
                        presets=PRESETS,share_enabled=False,circuit_absent=True)
        folder=state.PROJECTS_DIR/project_id
        share=state._json_projeto(folder,'share.json')
        return dict(revision=revision(c),confirmed=c.get('confirmed_by_user',False),components=c['components'],
                    kicad=kicad_current(project_id,c),export=export_is_current(folder,c),presets=PRESETS,
                    share_enabled=bool(share and share.get('enabled')))
    @app.get('/projects/{project_id}/production-view',response_class=HTMLResponse)
    def page(project_id:str):
        if state.buscar_projeto(project_id) is None:raise HTTPException(404,'Projeto não encontrado.')
        return production_page(project_id)
    @app.get('/projects/{project_id}/completion-view',response_class=HTMLResponse)
    def completed(project_id:str):
        circuit(project_id)
        return production_page(project_id,completed=True)
    @app.get('/sharing/address')
    def sharing_address(request:Request):
        from backend.network_share import share_address
        return share_address(str(request.base_url))
    @app.post('/projects/{project_id}/kicad')
    def generate(project_id:str,payload:dict={}):
        with state._circuit_review_lock:
            _,c=circuit(project_id)
            if payload.get('base_revision') is not None and payload['base_revision']!=revision(c):raise HTTPException(409,'O circuito mudou. Atualize a tela.')
            try:
                options={k:v for k,v in payload.items() if k!='base_revision'}
                normalized=export_options(c,options)
                current=kicad_current(project_id,c)
                if current and current.get('options')==normalized:
                    return dict(current,processing_reused=True)
                manifest,files=state.build_kicad(c,options,geometry=read_geometry(state.PROJECTS_DIR/project_id))
                diagram=json.loads(files['schematic_layout.json'])
                try:
                    falstad=saved_native(state,state.PROJECTS_DIR/project_id,c) or export_circuitjs(c,diagram)
                    files['circuitjs.txt']=falstad['circuit']
                    files['circuitjs.json']=json.dumps(falstad,ensure_ascii=False,indent=2)
                    manifest['files']=sorted(files)
                except ValueError as error:manifest['warnings'].append('Falstad: '+str(error))
            except (ValueError,TypeError) as e:raise HTTPException(422,str(e))
            folder=state.PROJECTS_DIR/project_id
            state._remover_artefatos_desatualizados(folder,ARTIFACTS+EXPORT_ARTIFACTS)
            # All names originate in the generator, not in an upload or user-selected path.
            for name,text in files.items():
                path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf8')
            manifest['hashes']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files}
            (folder/'kicad.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
            return manifest
    @app.get('/projects/{project_id}/kicad/download/{artifact}')
    def download_kicad(project_id:str,artifact:str):
        _,c=circuit(project_id);manifest=kicad_current(project_id,c)
        if not manifest:raise HTTPException(409,'Gere os arquivos para a versão atual do circuito.')
        names={'schematic':'schematic.kicad_sch','pcb':'board.kicad_pcb','preview':'schematic.svg'}
        if artifact not in names or names[artifact] not in manifest['files']:raise HTTPException(404,'Arquivo não gerado.')
        name=names[artifact];return FileResponse(state.PROJECTS_DIR/project_id/name,filename=name,media_type='image/svg+xml' if artifact=='preview' else None)
    @app.get('/projects/{project_id}/digital-twin')
    def twin(project_id:str):
        project,c=circuit(project_id)
        try:
            diagram=diagram_for(state.PROJECTS_DIR/project_id,c)
            project=state.buscar_projeto(project_id)
            return dict(source_revision=revision(c),diagram=diagram,svg=diagram_svg(diagram,c),confirmed=c.get('confirmed_by_user',False),circuit=c,open_pins=[m for n in c['nets'] if len(n['members'])==1 for m in n['members']],photo_url='/projects-files/'+project_id+'/'+project['stored_filename'])
        except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,'Confira a geometria do circuito: '+str(error))
    @app.get('/projects/{project_id}/circuitjs')
    def falstad(project_id:str):
        _,c=state._circuito_para_revisao(project_id)
        try:
            edited=saved_native(state,state.PROJECTS_DIR/project_id,c)
            return edited or preview_circuitjs(c,diagram_for(state.PROJECTS_DIR/project_id,c))
        except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,str(error))
    @app.get('/projects/{project_id}/circuitjs/download')
    def download_falstad(project_id:str):
        data=falstad(project_id)
        return Response(data['circuit'],media_type='text/plain',headers={'Content-Disposition':'attachment; filename="circuitjs.txt"'})
    @app.post('/projects/{project_id}/export')
    def export(project_id:str,payload:dict={},request:Request=None):
        with state._circuit_review_lock:
            project,c=circuit(project_id);manifest=kicad_current(project_id,c)
            if payload.get('base_revision') is not None and payload['base_revision']!=revision(c):raise HTTPException(409,'O circuito mudou. Atualize a tela.')
            if not manifest:raise HTTPException(409,'Gere o esquemático da versão atual antes de exportar.')
            if set(payload)-{'base_revision','base_url'}:raise HTTPException(422,'Opções de exportação inválidas.')
            try:
                folder=state.PROJECTS_DIR/project_id
                address=__import__('backend.network_share',fromlist=['share_address']).share_address(payload.get('base_url') or (str(request.base_url) if request else 'http://127.0.0.1:8010'))['address']
                current=export_is_current(folder,c);share=state._json_projeto(folder,'share.json') or {}
                if current and share.get('enabled') and current.get('share_url')==address.rstrip('/')+'/share/'+current.get('token',''):
                    return dict(current,processing_reused=True)
                return generate_export(folder,c,manifest,original_filename=project.get('stored_filename'),base_url=address)
            except (ValueError,TypeError) as e:raise HTTPException(422,str(e))
    def current_export(project_id):
        _,c=circuit(project_id);manifest=export_is_current(state.PROJECTS_DIR/project_id,c)
        if not manifest:raise HTTPException(409,'Gere o ZIP para a versão atual do projeto.')
        return manifest
    @app.get('/projects/{project_id}/export/download')
    def download_export(project_id:str):
        current_export(project_id);return FileResponse(state.PROJECTS_DIR/project_id/'project.zip',filename=project_id+'.zip',media_type='application/zip')
    @app.get('/projects/{project_id}/export/qr')
    def download_qr(project_id:str):
        current_export(project_id);return FileResponse(state.PROJECTS_DIR/project_id/'share_qr.svg',media_type='image/svg+xml')
    @app.post('/projects/{project_id}/share/revoke')
    def revoke(project_id:str):
        with state._circuit_review_lock:
            circuit(project_id);folder=state.PROJECTS_DIR/project_id
            (folder/'share.json').write_text(json.dumps(dict(enabled=False)),encoding='utf8')
            state._remover_artefatos_desatualizados(folder,EXPORT_ARTIFACTS)
        return dict(enabled=False)
    def shared(token):
        if not re.fullmatch(r'[A-Za-z0-9_-]{32}',token):raise HTTPException(404,'Compartilhamento não encontrado.')
        for folder in state.PROJECTS_DIR.iterdir():
            if not folder.is_dir() or folder.is_symlink():continue
            share=state._json_projeto(folder,'share.json')
            if share and share.get('enabled') and secrets.compare_digest(str(share.get('token','')),token):
                project=state.buscar_projeto(folder.name)
                if not project:break
                return project,current_export(folder.name)
        raise HTTPException(404,'Link desativado ou inexistente.')
    @app.get('/share/{token}',response_class=HTMLResponse)
    def share_page(token:str):
        project,manifest=shared(token);report=manifest['report']
        from backend.simulation_visual import simulation_page
        return simulation_page(base='/share/'+token)
    @app.get('/share/{token}/digital-twin')
    def shared_twin(token:str):
        project,_=shared(token);data=twin(project['project_id']);data['photo_url']='/share/'+token+'/photo';return data
    @app.get('/share/{token}/photo')
    def shared_photo(token:str):
        project,_=shared(token);return FileResponse(state.PROJECTS_DIR/project['project_id']/project['stored_filename'])
    @app.get('/share/{token}/circuitjs')
    def shared_falstad(token:str):
        project,_=shared(token);return falstad(project['project_id'])
    @app.get('/share/{token}/circuitjs/download')
    def shared_native_download(token:str):
        project,_=shared(token);return download_falstad(project['project_id'])
    @app.get('/share/{token}/download')
    def shared_download(token:str):
        project,_=shared(token);return FileResponse(state.PROJECTS_DIR/project['project_id']/'project.zip',filename=project['project_id']+'.zip',media_type='application/zip')
