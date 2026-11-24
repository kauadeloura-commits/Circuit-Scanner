"""Stage 14: bounded ngspice process, raw parsing and circuit measurements."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE_DIR=Path(__file__).resolve().parent.parent
MAX_RAW_BYTES=32*1024*1024
ARTIFACTS=('simulation.cir','spice.json','simulation.raw','simulation.log','simulation.csv','simulation.json')


def locate_ngspice():
    configured=os.environ.get('NGSPICE_CMD','').strip().strip('"')
    candidates=[configured,BASE_DIR/'tools/ngspice/Spice64/bin/ngspice_con.exe',
                BASE_DIR/'tools/ngspice/Spice64/bin/ngspice.exe',shutil.which('ngspice_con'),shutil.which('ngspice')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():return str(Path(candidate).resolve())
    return None


def simulator_status():
    command=locate_ngspice()
    return dict(ready=bool(command),engine='ngspice',path=command,
                message='Simulador disponível.' if command else 'ngspice não encontrado. Extraia também a pasta tools do ZIP ou configure NGSPICE_CMD.')


def deck_hash(text):return hashlib.sha256(text.encode('utf8')).hexdigest()


def parse_raw(path):
    path=Path(path)
    if not path.exists() or path.stat().st_size>MAX_RAW_BYTES:raise ValueError('O simulador não produziu resultados válidos dentro do limite de tamanho.')
    text=path.read_text(encoding='ascii',errors='strict')
    if '\nValues:' not in text or '\nVariables:' not in text:raise ValueError('Resultado do simulador incompleto.')
    header,values=text.split('Values:',1)
    metadata={k.strip():v.strip() for line in header.split('\nVariables:',1)[0].splitlines() if ':' in line for k,v in [line.split(':',1)]}
    if metadata.get('Flags')!='real':raise ValueError('Esta versão aceita somente resultados reais de DC ou transiente.')
    size=int(metadata.get('No. Variables','0'));points=int(metadata.get('No. Points','0'))
    if not 1<=size<=12000 or not 1<=points<=100000:raise ValueError('Quantidade de resultados fora dos limites.')
    variables=[]
    for line in header.split('\nVariables:',1)[1].strip().splitlines():
        parts=line.split()
        if len(parts)!=3 or int(parts[0])!=len(variables):raise ValueError('Colunas do simulador inválidas.')
        variables.append(dict(name=parts[1].lower(),unit=parts[2]))
    if len(variables)!=size:raise ValueError('Colunas do resultado incompletas.')
    tokens=values.split()
    if len(tokens)!=points*(size+1):raise ValueError('Dados de simulação incompletos.')
    rows=[]
    for i in range(points):
        offset=i*(size+1)
        if int(tokens[offset])!=i:raise ValueError('Sequência de resultados inválida.')
        row=[float(v) for v in tokens[offset+1:offset+1+size]]
        if any(not math.isfinite(v) for v in row):raise ValueError('Simulação produziu um valor não finito.')
        rows.append(row)
    return variables,rows


def measurements(manifest,variables,rows):
    indices={v['name']:i for i,v in enumerate(variables)}
    def vector(row,*names):
        for name in names:
            if name.lower() in indices:return row[indices[name.lower()]]
        return None
    def voltage(row,node):
        if node=='0':return 0.
        value=vector(row,'v('+node+')',node)
        if value is None:raise ValueError('Falta a tensão do nó '+node+' no resultado.')
        return value
    samples=[];current_meta=[]
    for b in manifest['components']:
        pins=b.get('current_pins') or b['pins'];current_meta.append(dict(component_id=b['component_id'],from_pin=pins[0],to_pin=pins[1],unit='A'))
    for row in rows:
        nets={name:voltage(row,node) for name,node in manifest['nodes'].items()}
        currents={}
        for b in manifest['components']:
            name=b['spice_name'].lower();p=name[0]
            if p=='r':
                pins=b['pins'];value=(voltage(row,b['nodes'][pins[0]])-voltage(row,b['nodes'][pins[1]]))/b['value_si']
            elif p in {'d','q','m'}:
                key='ic' if p=='q' else 'id';value=vector(row,'@'+name+'['+key+']','i(@'+name+'['+key+'])')
                # MOS1 reports PMOS id in its normalized device convention.
                if value is not None and b['type']=='mosfet_p':value=-value
            elif p in {'c','l'}:value=vector(row,'@'+name+'[i]','i(@'+name+'[i])','i('+name+')',name+'#branch')
            else:value=vector(row,'i('+name+')',name+'#branch')
            if value is None:raise ValueError('Falta a corrente de '+b['component_id']+' no resultado.')
            currents[b['component_id']]=value
        samples.append(dict(time_s=vector(row,'time'),voltages_v=nets,currents_a=currents))
    return dict(samples=samples,last=samples[-1],current_directions=current_meta,point_count=len(samples))


def error_message(log):
    lowered=log.lower()
    if any(s in lowered for s in ('singular matrix','timestep too small','no convergence','convergence failed')):
        return 'O circuito não convergiu. Confira terra, redes abertas, curtos e ligações das fontes.'
    if 'unknown parameter' in lowered or 'unknown device' in lowered or 'could not find a valid model' in lowered:
        return 'Um componente ou modelo não foi aceito pelo ngspice. Confira a netlist gerada.'
    return 'O ngspice não concluiu a simulação. Consulte o registro salvo e confira as conexões e valores.'


def run_simulation(manifest,folder,*,command=None,timeout=45):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    deck=manifest['netlist'];(folder/'simulation.cir').write_text(deck,encoding='utf8')
    executable=command or locate_ngspice();start=time.monotonic()
    base=dict(schema_version='stage14-simulation-1.0',engine='ngspice',source_revision=manifest['source_revision'],
              netlist_sha256=deck_hash(deck),options=manifest['options'],warnings=manifest['warnings'])
    if not executable:
        base.update(success=False,error_code='simulator_missing',message=simulator_status()['message'])
    else:
        try:
            process=subprocess.run([str(executable),'-n','-b','-o','simulation.log','simulation.cir'],cwd=str(folder),
                                   capture_output=True,text=True,errors='replace',timeout=timeout,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,check=False)
            log_path=folder/'simulation.log'
            log=(log_path.read_text(encoding='utf8',errors='replace') if log_path.exists() else '')
            log+='\n'+(process.stdout or '')+'\n'+(process.stderr or '')
            log_path.write_text(log,encoding='utf8')
            base['exit_code']=process.returncode
            # ngspice may quit with code zero after an aborted analysis and still
            # write a partial .raw file. That is never a completed simulation.
            if process.returncode!=0 or 'run simulation(s) aborted' in log.lower():
                base.update(success=False,error_code='simulator_error',message=error_message(log))
            else:
                variables,rows=parse_raw(folder/'simulation.raw')
                data=measurements(manifest,variables,rows)
                if manifest['options']['analysis']=='tran':
                    final=data['last']['time_s'];stop=manifest['options']['stop']
                    if final is None or final<stop-max(stop*1e-6,1e-15):raise ValueError('A análise temporal parou antes da duração solicitada. Confira convergência e redes abertas no registro.')
                base.update(data,success=True,message='Simulação concluída.')
                with (folder/'simulation.csv').open('w',newline='',encoding='utf8') as handle:
                    writer=csv.writer(handle);net_names=list(manifest['nodes']);ids=[b['component_id'] for b in manifest['components']]
                    writer.writerow(['time_s']+['V('+n+') [V]' for n in net_names]+['I('+c+') [A]' for c in ids])
                    for sample in data['samples']:writer.writerow([sample['time_s']]+[sample['voltages_v'][n] for n in net_names]+[sample['currents_a'][c] for c in ids])
        except subprocess.TimeoutExpired:
            base.update(success=False,error_code='timeout',message=f'A simulação excedeu {timeout:g} segundos e foi interrompida. Reduza o circuito ou a duração.')
        except (OSError,ValueError,UnicodeError) as error:
            base.update(success=False,error_code='result_error',message='Não foi possível concluir a simulação: '+str(error)[:300])
    base['elapsed_seconds']=round(time.monotonic()-start,3)
    (folder/'simulation.json').write_text(json.dumps(base,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return base
