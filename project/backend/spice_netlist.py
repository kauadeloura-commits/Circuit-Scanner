"""Stage 13: generate a bounded, validated SPICE deck from confirmed JSON."""
import math
import re
from backend.circuit_document import validar_circuito
from backend.circuit_review import revision
from backend.circuit_types import PIN_SCHEMAS
from backend.semiconductor_models import semiconductor, parameters

SUPPORTED=set(PIN_SCHEMAS)
GENERATOR_VERSION='reviewed-rails-complete-analysis-2'
UNITS={'resistor':('ohm','ohms','Ω','Ω'), 'capacitor':('F','farad','farads'),'capacitor_polarized':('F','farad','farads'),
       'inductor':('H','henry','henries'),'source_dc':('V','volt','volts'),'battery':('V','volt','volts'),'time':('s','second','seconds')}
SCALES={'':1.,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'K':1e3,'M':1e6,'meg':1e6,'g':1e9,'G':1e9,'t':1e12,'T':1e12,'R':1.,'r':1.}
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def value_number(value, kind, unit=None):
    if not isinstance(value,str) or not value.strip() or len(value)>100:
        raise ValueError('Informe um valor numérico em '+kind+'.')
    text=re.sub(r'\s+','',value).replace('µ','u').replace('μ','u').replace(',','.')
    valid_units=UNITS[kind]
    if unit and str(unit).strip().lower() not in {u.lower() for u in valid_units}:
        raise ValueError('Unidade incompatível com '+kind+': '+str(unit))
    for suffix in sorted(valid_units,key=len,reverse=True):
        if text.lower().endswith(suffix.lower()):
            text=text[:-len(suffix)];break
    # Printed 4k7 / 4R7 / 2M2 notation is a decimal separator plus multiplier.
    embedded=re.fullmatch(r'([+-]?\d+)([RrKkMmUuNnPp])(\d+)',text)
    if embedded:
        text=embedded[1]+'.'+embedded[3]+embedded[2]
    match=re.fullmatch('('+NUMBER+r')(meg|[pnumkKMGgTtRr]?)',text,re.IGNORECASE if text.lower().endswith('meg') else 0)
    if not match:
        raise ValueError('Valor inválido: '+value+'. Use, por exemplo, 10k, 4.7k ou 100uF.')
    suffix=match[2]
    if suffix.lower()=='meg':suffix='meg'
    number=float(match[1])*SCALES[suffix]
    if not math.isfinite(number) or abs(number)>1e15 or (number and abs(number)<1e-18):
        raise ValueError('Valor fora da faixa permitida: '+value)
    if kind not in {'source_dc','battery'} and number<=0:
        raise ValueError('O valor deve ser positivo em '+kind+'.')
    return number


def analysis_options(options=None):
    options=options or {}
    if not isinstance(options,dict) or set(options)-{'analysis','step','stop','initial_zero'}:
        raise ValueError('Opções de simulação inválidas.')
    kind=options.get('analysis','op')
    if kind=='op':return {'analysis':'op'}
    if kind!='tran':raise ValueError('Escolha ponto de operação ou análise no tempo.')
    step=value_number(options.get('step','10u'),'time')
    stop=value_number(options.get('stop','10m'),'time')
    if step>stop or stop/step>20000 or stop>1000:
        raise ValueError('Use passo menor que a duração e até 20.000 intervalos, com duração máxima de 1000 s.')
    initial=options.get('initial_zero',False)
    if not isinstance(initial,bool):raise ValueError('A condição inicial deve ser verdadeira ou falsa.')
    return dict(analysis='tran',step=step,stop=stop,initial_zero=initial)


def build_netlist(circuit,options=None):
    valid=validar_circuito(circuit)
    if not valid['valid']:raise ValueError('; '.join(valid['errors']))
    if circuit.get('confirmed_by_user') is not True or circuit.get('status')!='confirmed':
        raise ValueError('Revise e confirme o circuito na etapa 12 antes de gerar SPICE.')
    opts=analysis_options(options)
    unknown=[c['id']+' ('+c['type']+')' for c in circuit['components'] if c['type'] not in SUPPORTED]
    if unknown:
        raise ValueError('Modelo não disponível para: '+', '.join(unknown)+'. Nenhum componente foi omitido.')
    member_net={m:n['name'] for n in circuit['nets'] for m in n['members']}
    ground={n['name'] for n in circuit['nets'] if n['name'].upper() in {'GND','0'}}
    for c in circuit['components']:
        if c['pins']!=PIN_SCHEMAS[c['type']]:
            if set(c['pins'])!=set(PIN_SCHEMAS[c['type']]):raise ValueError('Confira os pinos de '+c['id']+'.')
        if c['type']=='gnd':ground.add(member_net[c['id']+'.GND'])
    if not ground:raise ValueError('Defina uma rede GND ou adicione o componente terra na revisão.')
    node_map={n['name']:('0' if n['name'] in ground else 'n'+str(i+1)) for i,n in enumerate(circuit['nets'])}
    from backend.power_rails import rail_sources
    rails=rail_sources(circuit)
    if not rails and not any(c['type'] in {'source_dc','battery'} for c in circuit['components']):
        raise ValueError('Adicione uma fonte, bateria ou VCC com tensão explícita.')
    # All emitted names and syntax are generated internally. User strings
    # stay in metadata, never in commands, include paths or model directives.
    lines=['Circuit Scanner - confirmed circuit','.options numdgt=15 reltol=1e-6 abstol=1e-12 vntol=1e-9']
    bindings=[];models=set();warnings=[];counts={}
    for c in sorted(circuit['components'],key=lambda c:c['id']):
        kind=c['type'];pins=PIN_SCHEMAS[kind]
        nodes={p:node_map[member_net[c['id']+'.'+p]] for p in pins}
        if kind in {'gnd','vcc'}:continue
        if len(set(nodes.values()))==1:
            raise ValueError('Todos os terminais de '+c['id']+' estão na mesma rede. Corrija o possível curto.')
        prefix={'resistor':'R','capacitor':'C','capacitor_polarized':'C','inductor':'L','source_dc':'V','battery':'V','diode':'D','led':'D',
                'transistor_npn':'Q','transistor_pnp':'Q','mosfet_n':'M','mosfet_p':'M','opamp':'E'}[kind]
        counts[prefix]=counts.get(prefix,0)+1;name=prefix+str(counts[prefix])
        binding=dict(component_id=c['id'],spice_name=name,type=kind,pins=pins,nodes=nodes)
        if prefix in {'Q','M','E'}:
            result=semiconductor(c,name,nodes,opts);lines.extend(result['lines']);warnings.extend(result['warnings'])
            binding.update(electrical_model=result['model'],current_pins=result['current_pins'])
        elif prefix=='D':
            model='CS_LED' if kind=='led' else 'CS_DIODE';models.add(model)
            lines.append(f'{name} {nodes["A"]} {nodes["K"]} {model}')
            binding['model']=model
        else:
            value=c.get('value')
            if kind=='inductor' and not value:
                _,resolved=parameters(c)
                if 'inductance' in resolved['parameters']:value=format(resolved['parameters']['inductance']['value'],'.15g')
            number=value_number(value,kind,c.get('unit'));binding['value_si']=number
            spec=format(number,'.15g')
            if prefix=='V':spec='DC '+spec
            if prefix in {'C','L'} and opts.get('initial_zero'):spec+=' IC=0'
            if kind=='inductor':
                _,resolved=parameters(c)
                initial=resolved['parameters'].get('initial_current')
                if initial and opts['analysis']=='tran':
                    spec=spec.replace(' IC=0','')+f' IC={initial["value"]:.15g}'
                    binding['initial_current_a']=initial['value']
                    if not opts.get('initial_zero'):warnings.append(c['id']+': habilite condições iniciais na análise temporal para aplicar a corrente inicial.')
                if 'saturation_current' in resolved['parameters']:
                    binding['saturation_current_a']=resolved['parameters']['saturation_current']['value']
                    warnings.append(c['id']+': corrente de saturação registrada como referência; o indutor desta simulação é linear.')
            first=nodes[pins[0]]
            if kind in {'inductor','capacitor','capacitor_polarized','source_dc','battery'} and c.get('metadata',{}).get('electrical_parameters'):
                code,resolved=parameters(c);param=resolved['parameters'].get('series_resistance')
                if param and param['value']>0:
                    first='series_'+name;lines.append(f'Rseries{name} {nodes[pins[0]]} {first} {param["value"]:.15g}')
                    binding['series_resistance_ohm']=param['value']
                if code:binding['model_code']=code
            lines.append(f'{name} {first} {nodes[pins[1]]} {spec}')
        bindings.append(binding)
    for rail in rails:
        counts['V']=counts.get('V',0)+1;name='V'+str(counts['V']);node=node_map[rail['net']]
        lines.append(f'{name} {node} 0 DC {rail["value"]:.15g}')
        bindings.append(dict(component_id=rail['component_id'],type='vcc',spice_name=name,pins=['VCC','GND'],nodes={'VCC':node,'GND':'0'},value_si=rail['value']))
        warnings.append(rail['component_id']+f': alimentação ideal de {rail["value"]:g} V em relação ao GND, usando a tensão explícita revisada.')
    if 'CS_DIODE' in models:lines.append('.model CS_DIODE D(IS=1e-14 N=1 RS=0.1)')
    if 'CS_LED' in models:lines.append('.model CS_LED D(IS=1e-20 N=2 RS=2)')
    if models:warnings.append('Diodos e LEDs usam modelos genéricos; o resultado não representa um modelo comercial específico.')
    if any(c['type']=='vcc' and not c.get('value') for c in circuit['components']):warnings.append('VCC sem tensão permanece como marcador; sua alimentação deve vir de uma fonte ou bateria.')
    for net in circuit['nets']:
        if len(net['members'])==1:warnings.append(net['name']+': apenas um terminal conectado ('+net['members'][0]+'); confira se a ligação aberta é intencional.')
    if opts['analysis']=='op' and any(c['type'] in {'capacitor','capacitor_polarized'} for c in circuit['components']):warnings.append('Análise DC: capacitores ficam abertos. Para observar carga, descarga ou oscilação, selecione Evolução no tempo.')
    if len(ground)>1:warnings.append('As redes com símbolos GND foram unidas ao nó de referência 0.')
    saves=['all']
    for b in bindings:
        p=b['spice_name'][0]
        if p in {'C','D','L','Q','M'}:saves.append('@'+b['spice_name'].lower()+('[ic]' if p=='Q' else '[id]' if p in {'D','M'} else '[i]'))
    lines.append('.save '+' '.join(saves))
    lines.append('.op' if opts['analysis']=='op' else '.tran '+format(opts['step'],'.15g')+' '+format(opts['stop'],'.15g')+(' uic' if opts['initial_zero'] else ''))
    lines+=['.control','set filetype=ascii','run','write simulation.raw all','quit','.endc','.end']
    return dict(schema_version='stage13-spice-1.0',source_revision=revision(circuit),generator_version=GENERATOR_VERSION,
                options=opts,nodes=node_map,components=bindings,warnings=warnings,netlist='\n'.join(lines)+'\n')
