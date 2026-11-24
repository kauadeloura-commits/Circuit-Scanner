"""Persist canvas edits into the circuit used by simulation and schematic exports."""
import copy,json,math
from fastapi import HTTPException
from fastapi.responses import Response
from backend.circuit_review import revision,review_issues
from backend.circuit_document import validar_circuito
from backend.digital_twin import make_diagram,read_geometry
from backend.circuitjs_export import export_circuitjs,preview_circuitjs

def pin_geometry(diagram):
    return dict(components=[dict(component_id=c['id'],type=c['type'],terminals=[dict(pin=pin,position=dict(x=p[0]*3.125,y=p[1]*3.125)) for pin,p in c['pins'].items()]) for c in diagram['components']],nets=[])

def apply_actions(current,diagram,actions):
    if not isinstance(actions,list) or len(actions)>200:raise ValueError('Lista de alterações inválida.')
    c=copy.deepcopy(current);geometry=pin_geometry(diagram)
    known={part['id']+'.'+pin for part in c['components'] for pin in part['pins']}
    for action in actions:
        mode=action.get('action')
        if mode in {'connect','disconnect'}:
            a=action.get('from');b=action.get('to')
            if a not in known or mode=='connect' and b not in known:raise ValueError('Selecione terminais do desenho.')
            na=next(n for n in c['nets'] if a in n['members'])
            if mode=='connect':
                nb=next(n for n in c['nets'] if b in n['members'])
                if na is nb:continue
                target,other=(nb,na) if nb['name'] in {'GND','0'} else (na,nb)
                target['members']=sorted(set(target['members']+other['members']));c['nets']=[n for n in c['nets'] if n is not other]
            else:
                na['members'].remove(a);c['nets']=[n for n in c['nets'] if n['members']]
                name='EDIT_'+str(len(c['nets'])+1)
                while any(n['name']==name for n in c['nets']):name+='_' 
                c['nets'].append(dict(name=name,members=[a]))
        elif mode=='move':
            item=next((p for p in geometry['components'] if p['component_id']==action.get('component_id')),None)
            if item is None:raise ValueError('Componente não encontrado.')
            dx,dy=float(action['dx']),float(action['dy'])
            if not math.isfinite(dx+dy) or max(abs(dx),abs(dy))>1000:raise ValueError('Deslocamento inválido.')
            for p in item['terminals']:p['position']['x']+=dx*3.125;p['position']['y']+=dy*3.125
        elif mode=='value':
            item=next((p for p in c['components'] if p['id']==action.get('component_id')),None)
            if item is None:raise ValueError('Componente não encontrado.')
            value=action.get('value')
            if not isinstance(value,str) or len(value)>100 or any(ord(x)<32 for x in value):raise ValueError('Valor inválido.')
            item.update(value=value,value_text=value)
        else:raise ValueError('Alteração visual não suportada.')
    c.setdefault('review',{}).update(visual_connections_edited=True,edit_source='schematic_canvas')
    c['warnings']=review_issues(c);c['validation']=dict(structurally_valid=True,component_count=len(c['components']),net_count=len(c['nets']),pin_count=len(known),review_required=bool(c['warnings']))
    if not validar_circuito(c)['valid']:raise ValueError('A alteração produziu uma ligação inválida.')
    return c,geometry

def saved_native(state,folder,c):
    native=state._json_projeto(folder,'falstad_edit.json') or {}
    if native.get('source_revision')!=revision(c):return None
    from backend.falstad_presentation import clean_presentation
    from backend.falstad_grid import fit_grid
    from backend.falstad_compact import compact_native
    from backend.falstad_edit import check_native_connections
    return check_native_connections(c,compact_native(c,fit_grid(c,clean_presentation(c,native))))

def register_visual(app,state):
    def save(pid,project,c,geometry,native=None):
        # Native import has already checked element types and connectivity.
        # Keep the user's snapshot intact; normalize only when rendering/exporting.
        data=None
        if native:
            data=dict(engine='CircuitJS/Falstad',circuit=native,source_revision=revision(c),edited=True,
                      components=geometry['native_bindings'],warnings=[],layout_version='native-user-edit',
                      model_mode='reviewed_parameters_with_documented_approximations')
        state._salvar_revisao(pid,project,c)
        folder=state.PROJECTS_DIR/pid
        (folder/'visual_geometry.json').write_text(json.dumps(dict(source_revision=revision(c),connectivity=geometry),ensure_ascii=False,indent=2),encoding='utf8')
        if native:
            (folder/'falstad_edit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        flow=state._json_projeto(folder,'guided_workflow.json')
        if flow and flow.get('phase') in {'circuit','simulation'}:
            flow.setdefault('inputs',{})['circuit']=revision(c)
            temporary=folder/'guided_visual.tmp';temporary.write_text(json.dumps(flow,ensure_ascii=False,indent=2),encoding='utf8');temporary.replace(folder/'guided_workflow.json')
        return dict(saved=True,revision=revision(c),confirmed=c.get('confirmed_by_user',False),
                    open_pins=[m for n in c['nets'] if len(n['members'])==1 for m in n['members']])

    @app.post('/projects/{project_id}/visual-edit')
    def visual(project_id:str,payload:dict):
        with state._circuit_review_lock:
            project,c=state._circuito_para_revisao(project_id)
            if payload.get('base_revision')!=revision(c):raise HTTPException(409,'O circuito mudou. Atualize o desenho antes de salvar.')
            try:
                d=make_diagram(c,*read_geometry(state.PROJECTS_DIR/project_id))
                changed,geometry=apply_actions(c,d,payload.get('actions'))
                return save(project_id,project,changed,geometry)
            except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))

    @app.post('/projects/{project_id}/falstad-edit')
    def native_edit(project_id:str,payload:dict):
        with state._circuit_review_lock:
            project,c=state._circuito_para_revisao(project_id);folder=state.PROJECTS_DIR/project_id
            if payload.get('base_revision')!=revision(c):raise HTTPException(409,'O circuito mudou. Atualize o simulador antes de salvar.')
            try:
                baseline=saved_native(state,folder,c) or {}
                if baseline.get('source_revision')!=revision(c):
                    from backend.schematic_views import diagram_for
                    baseline=preview_circuitjs(c,diagram_for(folder,c))
                from backend.falstad_edit import import_edits
                changed,geometry,text=import_edits(c,baseline,payload.get('circuit'))
                from backend.electrical_values import value_number
                from backend.semiconductor_models import falstad_fit
                for part in changed['components']:
                    if part['type'] in {'resistor','capacitor','capacitor_polarized','inductor','source_dc','battery','vcc'} and part.get('value'):
                        value_number(part['value'],part['type'],part.get('unit'))
                    elif part['type'].startswith(('transistor_','mosfet_')) or part['type']=='opamp':falstad_fit(part)
                return save(project_id,project,changed,geometry,text)
            except (ValueError,KeyError,TypeError,IndexError) as e:raise HTTPException(422,'Não foi possível salvar a edição do Falstad: '+str(e))

    @app.get('/projects/{project_id}/schematic/download')
    def schematic_download(project_id:str,format:str='svg'):
        _,c=state._circuito_para_revisao(project_id)
        from backend.kicad_export import build_kicad
        if format not in {'svg','kicad'}:raise HTTPException(422,'Escolha SVG ou KiCad.')
        _,files=build_kicad(c,dict(pcb=False),geometry=read_geometry(state.PROJECTS_DIR/project_id))
        if format=='svg':return Response(files['schematic.svg'],media_type='image/svg+xml',headers={'Content-Disposition':'attachment; filename="esquematico.svg"'})
        import io,zipfile
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name,text in files.items():z.writestr(name,text)
        return Response(stream.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="esquematico_editavel.zip"'})
