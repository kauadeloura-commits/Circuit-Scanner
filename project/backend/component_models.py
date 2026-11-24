"""OCR model identification and opt-in datasheet estimates. No SPICE integration."""
from copy import deepcopy
import math
import re
from backend.measurement_text import parse_measurement

PARAMETERS={
 'mos_kp':dict(label='Beta (β / KP)',unit='A/V²',positive=True),
 'gate_drain_capacitance':dict(label='Capacitância porta-dreno',unit='F',positive=True),
 'gate_source_capacitance':dict(label='Capacitância porta-fonte',unit='F',positive=True),
 'channel_lambda':dict(label='Lambda (λ)',unit='1/V',positive=True),
 'initial_current':dict(label='Corrente inicial',unit='A'),
 'hfe':dict(label='Ganho de corrente hFE / β',unit='A/A',positive=True),
 'vbe_on':dict(label='Tensão base-emissor VBE',unit='V'),
 'vgs_threshold':dict(label='Limiar VGS(th)',unit='V'),
 'gate_drive_voltage':dict(label='Tensão de acionamento VGS (carga)',unit='V'),
 'gm':dict(label='Transcondutância gm (ganho do MOSFET)',unit='S',positive=True),
 'rds_on':dict(label='Resistência RDS(on)',unit='ohm',positive=True),
 'open_loop_gain':dict(label='Ganho do AOP em malha aberta',unit='V/V',positive=True),
 'gbw':dict(label='Produto ganho-largura de banda',unit='Hz',positive=True),
 'slew_rate':dict(label='Slew rate',unit='V/us',positive=True),
 'offset_voltage':dict(label='Tensão de offset',unit='V'),
 'series_resistance':dict(label='Resistência série / DCR / ESR',unit='ohm',positive=True),
 'saturation_current':dict(label='Corrente de saturação do indutor',unit='A',positive=True),
 'rated_current':dict(label='Corrente nominal',unit='A',positive=True),
 'tolerance':dict(label='Tolerância',unit='%',positive=True),
 'inductance':dict(label='Indutância nominal do datasheet',unit='H',positive=True),
 'forward_voltage':dict(label='Tensão direta VF',unit='V',positive=True),
 'junction_capacitance':dict(label='Capacitância de junção',unit='F',positive=True),
}
FIELDS={
 'transistor_npn':['hfe','vbe_on'],'transistor_pnp':['hfe','vbe_on'],
 'mosfet_n':['vgs_threshold','gate_drive_voltage','gm','rds_on','mos_kp','gate_drain_capacitance','gate_source_capacitance','channel_lambda'],'mosfet_p':['vgs_threshold','gate_drive_voltage','gm','rds_on','mos_kp','gate_drain_capacitance','gate_source_capacitance','channel_lambda'],
 'opamp':['open_loop_gain','gbw','slew_rate','offset_voltage'],
 'inductor':['inductance','series_resistance','saturation_current','rated_current','tolerance','initial_current'],
 'capacitor':['series_resistance','tolerance'],'capacitor_polarized':['series_resistance','tolerance'],
 'resistor':['tolerance'],'source_dc':['series_resistance'],'battery':['series_resistance'],
 'diode':['forward_voltage','junction_capacitance'],'led':['forward_voltage'],
}
UI_FIELDS=deepcopy(FIELDS)
UI_FIELDS.update(mosfet_n=['mos_kp','vgs_threshold','gate_drain_capacitance','gate_source_capacitance','channel_lambda'],
                 mosfet_p=['mos_kp','vgs_threshold','gate_drain_capacitance','gate_source_capacitance','channel_lambda'],
                 inductor=['saturation_current','initial_current'],resistor=[],capacitor=[],capacitor_polarized=[],source_dc=[],battery=[],led=['forward_voltage'])
NO_MODEL={'resistor','capacitor','capacitor_polarized','led','source_dc','battery','vcc','gnd'}
DEFAULT_MODELS={'transistor_npn':'BC547B','transistor_pnp':'BC557B','mosfet_n':'BS170','mosfet_p':'IRF9540N','opamp':'LM358','inductor':'SRR6028-100Y','diode':'1N4148'}
CATALOG={}
def _spec(unit,condition,minimum=None,typical=None,maximum=None):
    return dict(unit=unit,condition=condition,min=minimum,typ=typical,max=maximum)

def _add(names,kind,url,manufacturer,parameters,page):
    for name in names:
        CATALOG[name]=dict(model=name,type=kind,manufacturer=manufacturer,url=url,page=page,
                           checked_at='2026-10-01',parameters=deepcopy(parameters))

# Exact family and gain group are retained. A/B/C are not stripped as packaging.
for kind,stems,url,sign in [
 ('transistor_npn',['BC546','BC547','BC548','BC549','BC550'],'https://m.onsemi.com/download/data-sheet/pdf/bc550-d.pdf',1),
 ('transistor_pnp',['BC556','BC557','BC558','BC559','BC560'],'https://www.onsemi.com/pdf/datasheet/bc556bta-d.pdf',-1)]:
    groups={'BC546':'ABC','BC547':'ABC','BC548':'ABC','BC549':'BC','BC550':'BC',
            'BC556':'AB','BC557':'AB','BC558':'B','BC559':'BC','BC560':'C'}
    for stem in stems:
        for suffix,low,high in [('',110,800),('A',110,220),('B',200,450),('C',420,800)]:
            if suffix and suffix not in groups[stem]:continue
            _add([stem+suffix],kind,url,'onsemi',{
                'hfe':_spec('A/A',f'VCE = {sign*5} V; IC = {sign*2} mA; '+('TC' if sign==1 else 'TA')+' = 25 °C',low,None,high),
                'vbe_on':_spec('V',f'VCE = {sign*5} V; IC = {sign*2} mA; '+('TC' if sign==1 else 'TA')+' = 25 °C',
                                  .58 if sign==1 else -.75,.66*sign,.70 if sign==1 else -.60)},2)
_add(['2N3904'],'transistor_npn','https://www.onsemi.com/pdf/datasheet/2n3904-d.pdf','onsemi',
     {'hfe':_spec('A/A','IC = 10 mA; VCE = 1 V; TA = 25 °C',100,None,300)},2)
_add(['2N3906'],'transistor_pnp','https://www.onsemi.com/pdf/datasheet/2n3906-d.pdf','onsemi',
     {'hfe':_spec('A/A','|IC| = 10 mA; |VCE| = 1 V; TA = 25 °C',100,None,300)},2)
_add(['1N4148'],'diode','https://www.vishay.com/docs/81857/1n4148.pdf','Vishay',{
    'forward_voltage':_spec('V','IF = 10 mA; TA = 25 °C; somente máximo especificado',maximum=1),
    'junction_capacitance':_spec('F','VR = 0 V; f = 1 MHz; VHF = 50 mV; TA = 25 °C; somente máximo especificado',maximum=4e-12)},2)
CATALOG['1N4148']['checked_at']='2026-10-05'
_add(['1N400'+str(n) for n in range(1,8)],'diode','https://www.vishay.com/docs/88503/1n4001.pdf','Vishay',{
    'forward_voltage':_spec('V','IF = 1 A; TA = 25 °C; VF máximo, não valor nominal fixo',maximum=1.1),
    'junction_capacitance':_spec('F','VR = 4 V; f = 1 MHz; TA = 25 °C; valor típico da tabela',typical=15e-12)},2)
for n in range(1,8):
    CATALOG['1N400'+str(n)].update(checked_at='2026-11-09',datasheet_revision='88503, 2020-04-29')
_add(['IRLZ44N'],'mosfet_n','https://www.infineon.com/part/IRLZ44N','Infineon',
     {'vgs_threshold':_spec('V','Faixa VGS(th) publicada na página do fabricante',1,None,2)},None)
_add(['IRF9540N'],'mosfet_p','https://www.infineon.com/part/IRF9540N','Infineon',
     {'vgs_threshold':_spec('V','Faixa VGS(th) publicada na página do fabricante',-4,None,-2)},None)
_add(['IRFZ44N'],'mosfet_n','https://www.infineon.com/part/IRFZ44N','Infineon',
     {'vgs_threshold':_spec('V','Faixa VGS(th) publicada na página do fabricante',2,None,4)},None)
_add(['IRF4905'],'mosfet_p','https://www.infineon.com/part/IRF4905','Infineon',
     {'vgs_threshold':_spec('V','Faixa VGS(th) publicada na página do fabricante',-4,None,-2)},None)
_add(['IRF640N'],'mosfet_n','https://www.infineon.com/assets/row/public/documents/24/49/infineon-irf640n-datasheet-en.pdf','Infineon',
     {'vgs_threshold':_spec('V','VDS = VGS; ID = 250 µA; TJ = 25 °C',2,None,4),
      'gate_drive_voltage':_spec('V','VGS do ensaio RDS(on), ID = 11 A; referência para esta condição, não recomendação universal',typical=10),
      'gm':_spec('S','VDS = 50 V; ID = 11 A; TJ = 25 °C',6.8),
      'rds_on':_spec('ohm','VGS = 10 V; ID = 11 A; TJ = 25 °C',maximum=.15)},2)
_add(['BS170','MMBF170'],'mosfet_n','https://www.onsemi.com/pdf/datasheet/mmbf170-d.pdf','onsemi',
     {'vgs_threshold':_spec('V','VDS = VGS; ID = 1 mA; TA = 25 °C',.8,2.1,3),
      'gate_drive_voltage':_spec('V','VGS do ensaio RDS(on), ID = 200 mA; referência para esta condição, não recomendação universal',typical=10),
      'gm':_spec('S','ID = 200 mA; BS170: VDS = 10 V; MMBF170: VDS >= 2 VDS(on); TA = 25 °C',typical=.32),
      'rds_on':_spec('ohm','VGS = 10 V; ID = 200 mA; TA = 25 °C',typical=1.2,maximum=5)},1)
_add(['IRF540N'],'mosfet_n','https://www.infineon.com/assets/row/public/documents/24/49/infineon-irf540n-datasheet-en.pdf','Infineon',
     {'vgs_threshold':_spec('V','VDS = VGS; ID = 250 µA; TJ = 25 °C',2,None,4),
      'gate_drive_voltage':_spec('V','VGS do ensaio RDS(on), ID = 16 A; referência para esta condição, não recomendação universal',typical=10),
      'gm':_spec('S','VDS = 50 V; ID = 16 A; TJ = 25 °C',21),
      'rds_on':_spec('ohm','VGS = 10 V; ID = 16 A; TJ = 25 °C',maximum=.044)},2)
for name,low,high,condition,rdsmax in [('2N7000',.8,3,'VDS = VGS; ID = 1 mA',5),
                                     ('2N7002',1,2.5,'VDS = VGS; ID = 250 µA',7.5),
                                     ('NDS7002A',1,2.5,'VDS = VGS; ID = 250 µA',7.5)]:
    _add([name],'mosfet_n','https://www.onsemi.com/pdf/datasheet/nds7002a-d.pdf','onsemi',
         {'vgs_threshold':_spec('V',condition+'; TC = 25 °C',low,2.1,high),
          'gate_drive_voltage':_spec('V','VGS do ensaio RDS(on), ID = 500 mA; referência para esta condição, não recomendação universal',typical=10),
          'rds_on':_spec('ohm','VGS = 10 V; ID = 500 mA; TC = 25 °C',typical=1.2,maximum=rdsmax)},2)
_add(['LM158','LM158A','LM258','LM258A','LM358','LM358A','LM2904','LM2904V'],'opamp',
     'https://www.ti.com/lit/ds/symlink/lm358.pdf','Texas Instruments',{
      'open_loop_gain':_spec('V/V','VS = 15 V; VO = 1..11 V; RL >= 2 kohm; TA = 25 °C',typical=100000),
      'gbw':_spec('Hz','TA = 25 °C; variante clássica, sem sufixo B',typical=700000),
      'slew_rate':_spec('V/us','G = +1; TA = 25 °C; variante clássica',typical=.3)},'10–14')
_add(['LM324'],'opamp','https://www.ti.com/lit/ds/symlink/lm324.pdf','Texas Instruments',{
      'open_loop_gain':_spec('V/V','VCC = 15 V; VO = 1..11 V; RL >= 2 kohm; TA = 25 °C; LM324 clássico',25000,100000),
      'gbw':_spec('Hz','VCC = +/-15 V; RL = 1 Mohm; CL = 20 pF; TA = 25 °C; seção 6.10',typical=1200000),
      'slew_rate':_spec('V/us','VCC = +/-15 V; G = 1; RL = 1 Mohm; CL = 30 pF; VI = +/-10 V; TA = 25 °C',typical=.5),
      'offset_voltage':_spec('V','VCC = 5 V; VIC = VICR mínimo; VO = 1,4 V; TA = 25 °C; LM324 clássico',typical=.003,maximum=.007)},'11, 13')
CATALOG['LM324'].update(checked_at='2026-11-09',datasheet_revision='SLOS066AE, 2025-09')
_add(['LM358B','LM358BA','LM2904B','LM2904BA'],'opamp',
     'https://www.ti.com/lit/ds/symlink/lm358.pdf','Texas Instruments',{
      'open_loop_gain':_spec('V/V','VS = 15 V; VO = 1..11 V; RL >= 10 kohm ligado a V-; TA = 25 °C',70000,140000),
      'gbw':_spec('Hz','TA = 25 °C; variante B/BA',typical=1200000),
      'slew_rate':_spec('V/us','G = +1; TA = 25 °C; variante B/BA',typical=.5)},'6–8')
_add(['NE5532','NE5532A','SA5532','SA5532A'],'opamp','https://www.ti.com/lit/ds/symlink/ne5532.pdf','Texas Instruments',{
      'open_loop_gain':_spec('V/V','VCC = +/-15 V; RL >= 2 kohm; VO = +/-10 V; TA = 25 °C',typical=100000),
      'gbw':_spec('Hz','Ganho unitário; VCC = +/-15 V; TA = 25 °C; revisão K',typical=12000000),
      'slew_rate':_spec('V/us','Ganho unitário; VCC = +/-15 V; TA = 25 °C; revisão K',typical=5),
      'offset_voltage':_spec('V','VO = 0; VCC = +/-15 V; TA = 25 °C',typical=.0005)},4)
_add([stem+suffix for stem in ('TL071','TL072','TL074') for suffix in ('','A','B','C','AC','BC','I','M')],
     'opamp','https://www.ti.com/lit/ds/symlink/tl071.pdf','Texas Instruments',{
      'open_loop_gain':_spec('V/V','VS = +/-15 V; VO = 0; TA = 25 °C; versões clássicas, sem H',typical=200000)},15)
# Passive parameters retain their table meaning: limits are not typical values.
coil_rows=[('1R5',1.5,14,3.4,3.5),('2R5',2.5,25,2.8,3),('3R0',3,28,3,2.6),
 ('3R3',3.3,28,2.5,2.6),('3R9',3.9,32,2.45,2.3),('5R0',5,36,2.4,2.1),
 ('6R0',6,40,2.3,2),('7R3',7.3,52,2.2,1.85),('8R6',8.6,56,2.1,1.82),
 ('100',10,65,2,1.7),('120',12,70,1.9,1.5),('150',15,78,1.7,1.3),
 ('180',18,85,1.5,1.1),('220',22,115,1.4,1.08),('270',27,130,1.2,1),
 ('330',33,155,1.15,.92),('390',39,195,1,.75),('470',47,235,.92,.72),
 ('560',56,260,.85,.65),('680',68,300,.8,.6),('820',82,400,.7,.56),
 ('101',100,500,.58,.55),('151',150,680,.5,.42),('221',220,820,.45,.36),
 ('331',330,1400,.37,.27),('471',470,2100,.27,.22),('681',680,3100,.22,.2),('102',1000,4500,.15,.17)]
for code,uh,mohm,irms,isat in coil_rows:
    _add(['SRR6028-'+code+'Y'],'inductor','https://www.bourns.com/docs/Product-Datasheets/SRR6028.pdf','Bourns',{
      'inductance':_spec('H','Indutância nominal da tabela; ensaio a 100 kHz',typical=uh*1e-6),
      'tolerance':_spec('%','Tolerância da indutância nominal',typical=30),
      'series_resistance':_spec('ohm','DCR máxima da tabela; não é um valor típico',maximum=mohm/1000),
      'rated_current':_spec('A','Irms da tabela; limite térmico',maximum=irms),
      'saturation_current':_spec('A','Isat da tabela; redução típica de 35% da indutância',typical=isat)},1)

# Fixed-capacitance / square-law approximations, identified as derived values.
# BS170/MMBF170 electrical table, onsemi MMBF170/D rev.9, p.2 (2026-02).
for name in ('BS170','MMBF170'):
    item=CATALOG[name];item.update(page=2,checked_at='2026-10-05')
    item['parameters'].update(
        mos_kp=dict(_spec('A/V²','Estimativa de baixo VDS: β = 1/[RDS(on) × (VGS − VGS(th))], usando 1,2 ohm, 10 V e 2,1 V; 25 °C',typical=1/(1.2*(10-2.1))),derived=True),
        gate_drain_capacitance=_spec('F','Crss típico; VDS = 10 V; VGS = 0 V; f = 1 MHz; 25 °C',typical=7e-12),
        gate_source_capacitance=dict(_spec('F','Estimativa Cgs = Ciss − Crss = 24 − 7 pF; VDS = 10 V; VGS = 0 V; f = 1 MHz; 25 °C',typical=17e-12),derived=True))

# Manufacturer tables checked 2026-10-06, page 2. Maximum RDS(on) is a
# boundary used for a labelled fit; it is never presented as a typical value.
for name,drive,current,rds,ciss,crss in [
    ('IRLZ44N',10,25,.022,1700e-12,150e-12),
    ('IRF9540N',-10,-11,.117,1300e-12,240e-12),
    ('IRFZ44N',10,25,.0175,1470e-12,88e-12),
    ('IRF4905',-10,-38,.020,3400e-12,640e-12)]:
    item=CATALOG[name];item.update(url='https://www.infineon.com/assets/row/public/documents/24/49/infineon-'+name.lower()+'-datasheet-en.pdf',page=2,checked_at='2026-10-06')
    condition=f'VGS = {drive} V; ID = {current} A; TJ = 25 °C'
    item['parameters'].update(gate_drive_voltage=_spec('V','Condição de ensaio de RDS(on): '+condition,typical=drive),
        rds_on=_spec('ohm',condition+'; máximo publicado, não típico',maximum=rds),
        gate_drain_capacitance=_spec('F','Crss típico; VGS = 0 V; |VDS| = 25 V; f = 1 MHz',typical=crss),
        gate_source_capacitance=dict(_spec('F','Cgs = Ciss − Crss; VGS = 0 V; |VDS| = 25 V; f = 1 MHz',typical=ciss-crss),derived=True))


for item in CATALOG.values():
    if not item['type'].startswith('mosfet_'):continue
    p=item['parameters'];t=p.get('vgs_threshold',{});d=p.get('gate_drive_voltage',{});r=p.get('rds_on',{})
    threshold=t.get('typ') if t.get('typ') is not None else (t['min']+t['max'])/2 if t.get('min') is not None and t.get('max') is not None else None
    drive=d.get('typ');rds=r.get('typ') or r.get('max')
    if threshold is not None and drive is not None and rds and abs(drive)>abs(threshold):
        kp=1/(rds*(abs(drive)-abs(threshold)))
        p['mos_kp']=dict(_spec('A/V²',f'KP = 1/[RDS(on) × (|VGS| − |Vth|)]; RDS={rds:g} ohm, VGS={drive:g} V, Vth={threshold:g} V. Ajuste aproximado usando RDS máximo quando não há típico.',typical=kp),derived=True)


def automatic_model_fields(component):
    """Fill blank model fields for review and simulation; retain explicit inputs."""
    result=deepcopy(component);kind=result['type']
    if kind not in DEFAULT_MODELS:return result
    old=result.get('model') or {};old={'text':old,'source':'manual'} if isinstance(old,str) else old
    code=old.get('text','').strip() or DEFAULT_MODELS[kind]
    result['model']=dict(old,text=code,source=old.get('source','manual') if old.get('text') else 'suggested')
    policies=deepcopy(result.get('parameter_policy') or {})
    for key in FIELDS[kind]:
        if key!='initial_current':policies.setdefault(key,'datasheet')
    explicit={k:v for k,v in (result.get('electrical_parameters') or {}).items()
              if not isinstance(v,dict) or not v.get('source','').startswith('datasheet_')}
    resolved=resolve_parameters(kind,code,explicit,policies)
    values=resolved['parameters'];item=resolved['datasheet']
    reference=item or lookup_model(DEFAULT_MODELS[kind],kind)
    for key in FIELDS[kind]:
        if key in values or policies.get(key)!='datasheet':continue
        spec=(reference or {}).get('parameters',{}).get(key)
        if not spec:continue
        number=spec.get('typ')
        if number is None and spec.get('min') is not None and spec.get('max') is not None:number=(spec['min']+spec['max'])/2
        if number is None:continue
        values[key]=dict(value=number,unit=PARAMETERS[key]['unit'],source='datasheet_derived' if spec.get('derived') else 'datasheet_typical' if spec.get('typ') is not None else 'datasheet_midpoint',estimated=True,
                         datasheet=dict(model=reference['model'],url=reference['url'],manufacturer=reference['manufacturer'],page=reference['page'],checked_at=reference['checked_at'],**deepcopy(spec)))
        if not item:values[key]['datasheet']['reference_only']=True
    if kind.startswith('mosfet_') and ('mos_kp' not in values or values['mos_kp'].get('estimated')) and policies.get('mos_kp')=='datasheet':
        threshold=values.get('vgs_threshold',{}).get('value');drive=values.get('gate_drive_voltage',{}).get('value')
        spec=(reference or {}).get('parameters',{}).get('rds_on',{})
        rds=values.get('rds_on',{}).get('value',spec.get('typ') or spec.get('max'))
        if threshold is not None and drive is not None and rds and rds>0 and abs(drive)>abs(threshold):
            kp=1/(rds*(abs(drive)-abs(threshold)))
            values['mos_kp']=dict(value=kp,unit='A/V²',source='datasheet_derived',estimated=True,
                datasheet=dict(model=reference['model'],manufacturer=reference['manufacturer'],url=reference['url'],page=reference['page'],checked_at=reference['checked_at'],
                               min=None,max=None,typ=kp,reference_only=not bool(item),derived=True,
                               condition=f'KP = 1/[RDS(on) × (|VGS| − |Vth|)]; RDS={rds:g} ohm, VGS={drive:g} V, Vth={threshold:g} V. Ajuste aproximado; RDS máximo quando não há típico.'))
    result.update(electrical_parameters=values,parameter_policy=policies,
                  parameter_pending=[p for p in resolved['pending'] if p['parameter'] not in values])
    return result

def model_name(value):
    if not isinstance(value,str) or len(value)>80 or any(ord(c)<32 for c in value):
        raise ValueError('Modelo deve ter até 80 caracteres, sem quebras de linha.')
    return re.sub(r'\s+','',value).upper()

def lookup_model(value,kind):
    name=model_name(value);item=CATALOG.get(name)
    # Only explicit packaging aliases. Electrical suffixes remain significant.
    if not item and re.fullmatch(r'IR(?:F|LZ)\d+N(?:PBF)',name):item=CATALOG.get(name[:-3])
    if item and item['type']!=kind:raise ValueError('O modelo '+name+' não corresponde à classe '+kind+'.')
    return deepcopy(item) if item else None

def model_classes(text):
    name=model_name(text)
    if name in CATALOG:return {CATALOG[name]['type']}
    # Classification only: no electrical parameters are borrowed across package
    # variants. Manufacturer: Diodes BS250/BS250P, P-channel enhancement DMOS.
    if name in {'BS250','BS250P','BS250F','BS250KL'}:return {'mosfet_p'}
    if re.fullmatch(r'(?:BC|BD|TIP|2N)\d{3,}[A-Z0-9-]*',name):return {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p'}
    if re.fullmatch(r'(?:IRF|IRL|BS|BSS|AO|FQP|FQU|STP|SI|MMBF)[A-Z0-9-]*\d[A-Z0-9-]*',name):return {'mosfet_n','mosfet_p'}
    if re.fullmatch(r'(?:LM|TL|NE|SA|OPA|AD|MCP|UA|RC|MC)\d{2,}[A-Z0-9-]*',name):return {'opamp'}
    if re.fullmatch(r'(?:SRR|SRN|SDR|CDRH|LQH|LQM|IHLP|MSS|WE)[A-Z0-9-]*\d[A-Z0-9-]*',name):return {'inductor'}
    if re.fullmatch(r'(?:(?:1N|BAV|UF|FR)\d+|BAT\d{2,})[A-Z0-9-]*',name):return {'diode','led'}
    return set()

def model_ocr_reading(text):
    """Catalog-backed OCR hypotheses; preserve actual letters and gain suffixes.

    Only unique confusion matches inside a catalog model's numeric region are
    repaired automatically. A possible suffix B/8 or unknown valid model stays
    as read, with suggestions for review rather than guessed parameters.
    """
    name=model_name(text)
    if name in CATALOG:
        return dict(text=name,raw_text=text,classes={CATALOG[name]['type']},corrections=[],candidates=[],needs_review=False)
    groups=('0O','1IL|','2Z','5S','8B')
    def equivalent(a,b):return a==b or any(a in g and b in g for g in groups)
    matches=[]
    for candidate,item in CATALOG.items():
        if len(candidate)!=len(name):continue
        changes=[i for i,(a,b) in enumerate(zip(name,candidate)) if a!=b]
        if 1<=len(changes)<=2 and all(equivalent(a,b) for a,b in zip(name,candidate)):
            matches.append((candidate,item,changes))
    classes=model_classes(name)
    candidates=[m[0] for m in matches]
    if matches:classes=classes.union(*(set([item['type']]) for _,item,_ in matches))
    if len(matches)==1:
        candidate,item,changes=matches[0]
        # A letter between the prefix and another digit is strong digit evidence.
        # Final letters such as BC547B remain meaningful model suffixes.
        safe=all(candidate[i].isdigit() and (any(c.isdigit() for c in candidate[i+1:]) or
                                             name[i] in 'OIL|' and any(c.isdigit() for c in candidate[:i])) for i in changes)
        if safe:
            return dict(text=candidate,raw_text=text,classes={item['type']},
                        corrections=[dict(index=i,**{'from':name[i]},to=candidate[i]) for i in changes],
                        candidates=candidates,needs_review=True)
    return dict(text=name,raw_text=text,classes=classes,corrections=[],candidates=candidates,
                needs_review=bool(matches) or bool(classes and name not in CATALOG))

def parameter_number(value,key):
    if isinstance(value,bool) or not isinstance(value,(str,float,int)):raise ValueError('Parâmetro inválido: '+key)
    text=str(value).strip().replace(',','.').replace('µ','u').replace('μ','u')
    unit=PARAMETERS[key]['unit']
    for suffix in sorted({unit,'Ω' if unit=='ohm' else unit,'Ω' if unit=='ohm' else unit},key=len,reverse=True):
        if text.lower().endswith(suffix.lower()):text=text[:-len(suffix)].strip();break
    match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(p|n|u|m|k|K|M|meg|G)?',text)
    if not match:raise ValueError('Valor inválido em '+PARAMETERS[key]['label']+'. Use a unidade indicada.')
    scale={'':1,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'K':1e3,'M':1e6,'meg':1e6,'G':1e9}
    number=float(match[1])*scale[match[2] or '']
    if not math.isfinite(number) or abs(number)>1e15:raise ValueError('Parâmetro fora da faixa: '+key)
    if PARAMETERS[key].get('positive') and number<0:raise ValueError('O parâmetro '+key+' não pode ser negativo.')
    if key in {'hfe','gm','mos_kp','open_loop_gain','gbw','slew_rate','inductance','rated_current','saturation_current'} and number==0:
        raise ValueError('O parâmetro '+key+' deve ser maior que zero.')
    if key=='tolerance' and number>100:raise ValueError('Tolerância deve ficar entre 0 e 100%.')
    return number

def resolve_parameters(kind,model,parameters=None,policies=None):
    if kind not in FIELDS:
        if model or parameters or policies:raise ValueError('Esta classe não utiliza modelo ou parâmetros.')
        return dict(parameters={},policies={},pending=[],datasheet=None)
    if not isinstance(parameters or {},dict) or not isinstance(policies or {},dict):raise ValueError('Parâmetros e opções devem ser objetos.')
    parameters=parameters or {};policies=policies or {};fields=FIELDS[kind]
    if set(parameters)-set(fields) or set(policies)-set(fields):raise ValueError('Parâmetro incompatível com a classe '+kind+'.')
    if any(v not in ('manual','datasheet') for v in policies.values()):raise ValueError('Escolha informar ou usar datasheet.')
    item=lookup_model(model,kind) if model else None;resolved={};pending=[]
    for key in fields:
        entry=parameters.get(key);raw=entry.get('value') if isinstance(entry,dict) else entry
        source=entry.get('source','manual') if isinstance(entry,dict) else 'manual'
        # An estimate is always regenerated for the current part; submitted
        # source URLs/ranges never become trusted catalog provenance.
        if raw is not None and str(raw).strip() and source not in {'datasheet_typical','datasheet_midpoint','datasheet_limit','datasheet_derived'}:
            number=parameter_number(raw,key)
            if key in {'vgs_threshold','gate_drive_voltage'} and ((kind=='mosfet_n' and number<=0) or (kind=='mosfet_p' and number>=0)):
                raise ValueError('VGS deve ser positivo no canal N e negativo no canal P.')
            resolved[key]=dict(value=number,unit=PARAMETERS[key]['unit'],source='ocr' if source=='ocr' else 'manual',estimated=False)
            if source=='ocr' and isinstance(entry,dict):
                for evidence in ('raw_text','corrections','needs_review'):
                    if evidence in entry:resolved[key][evidence]=deepcopy(entry[evidence])
        elif policies.get(key)=='datasheet':
            spec=(item or {}).get('parameters',{}).get(key)
            if spec and spec['typ'] is not None:number=spec['typ'];source='datasheet_derived' if spec.get('derived') else 'datasheet_typical'
            elif spec and spec['min'] is not None and spec['max'] is not None:
                number=(spec['min']+spec['max'])/2;source='datasheet_midpoint'
            else:
                pending.append(dict(parameter=key,reason='only_limit' if spec else 'not_found',
                    message='O datasheet só informa um limite; informe um valor ou confira a condição.' if spec else 'Parâmetro sem datasheet verificado para este modelo.',bounds=deepcopy(spec)))
                continue
            resolved[key]=dict(value=number,unit=PARAMETERS[key]['unit'],source=source,estimated=True,
                               datasheet=dict(model=item['model'],manufacturer=item['manufacturer'],url=item['url'],page=item['page'],
                                              checked_at=item['checked_at'],**deepcopy(spec)))
    return dict(parameters=resolved,policies=deepcopy(policies),pending=pending,datasheet=item)

def model_edit(component,edit):
    old=component.get('model') or {};old_name=old.get('text','') if isinstance(old,dict) else str(old)
    raw=edit.get('model',old_name);name=model_name(raw)
    kind=component['type']
    if kind in NO_MODEL:name=''
    if not name and kind in DEFAULT_MODELS:name=DEFAULT_MODELS[kind]
    source=old.get('source','manual') if name==old_name and old.get('source')!='missing' else 'manual'
    parameters=deepcopy(edit.get('electrical_parameters',component.get('electrical_parameters')) or {})
    if kind in NO_MODEL:parameters={k:v for k,v in parameters.items() if k in UI_FIELDS.get(kind,[])}
    if isinstance(parameters,dict):
        for key,entry in parameters.items():
            prior=(component.get('electrical_parameters') or {}).get(key)
            if isinstance(entry,dict) and entry.get('source')=='ocr' and entry!=prior:entry['source']='manual'
    policies=deepcopy(edit.get('parameter_policy',component.get('parameter_policy')) or {})
    if kind in NO_MODEL:policies={k:v for k,v in policies.items() if k in UI_FIELDS.get(kind,[])}
    if name:
        for key in UI_FIELDS.get(kind,[]):
            if key!='initial_current':policies.setdefault(key,'datasheet')
    if not old_name and name==DEFAULT_MODELS.get(kind) and not raw:source='suggested'
    if not old_name and name==DEFAULT_MODELS.get(kind) and edit.get('model_source')=='suggested':source='suggested'
    resolved=resolve_parameters(kind,name,parameters,policies)
    return dict(model=dict(text=name,source=source if name else 'missing',confidence=old.get('confidence') if source=='ocr' else None),
                electrical_parameters=resolved['parameters'],parameter_policy=resolved['policies'],parameter_pending=resolved['pending'])

def preserve_manual_models(current,previous):
    by_id={c.get('detection_id'):c for c in (previous or {}).get('components',[])}
    for c in current.get('components',[]):
        old=by_id.get(c.get('detection_id'))
        if not old or old.get('type')!=c.get('type'):continue
        # Never carry a model edit into a different box after a new detection.
        if any(old.get('bbox',{}).get(k)!=c.get('bbox',{}).get(k) for k in ('x1','y1','x2','y2')):continue
        # Re-reading uncertain handwriting must not erase the user's edits.
        for field in ('reference','value'):
            if old.get(field,{}).get('source')=='manual':
                c[field]=deepcopy(old[field])
                if field=='reference':c['component_id']=old.get('component_id') or old[field].get('text')
        model=c.get('model',{})
        if old.get('model',{}).get('source')=='manual':model=old['model']
        parameters=deepcopy(c.get('electrical_parameters',{}))
        for key,entry in old.get('electrical_parameters',{}).items():
            if entry.get('source')=='manual':parameters[key]=deepcopy(entry)
        resolved=resolve_parameters(c['type'],model.get('text',''),parameters,old.get('parameter_policy',{}))
        c.update(model=model,electrical_parameters=resolved['parameters'],parameter_policy=resolved['policies'],parameter_pending=resolved['pending'])
    current['model_ocr_count']=sum(c.get('model',{}).get('source')=='ocr' for c in current.get('components',[]))
    current['reference_ocr_count']=sum(c.get('reference',{}).get('source')=='ocr' for c in current.get('components',[]))
    current['value_ocr_count']=sum(c.get('value',{}).get('source')=='ocr' for c in current.get('components',[]))
    return current

def catalog_summary():
    return dict(parameters=deepcopy(PARAMETERS),fields=deepcopy(FIELDS),model_count=len(CATALOG),
                ui_fields=deepcopy(UI_FIELDS),no_model=sorted(NO_MODEL),default_models=deepcopy(DEFAULT_MODELS),
                models=[dict(model=n,type=e['type'],manufacturer=e['manufacturer'],url=e['url'],page=e['page'],checked_at=e['checked_at'],parameters=deepcopy(e['parameters'])) for n,e in sorted(CATALOG.items())],
                simulation_integration=True,
                note='VGS(th) indica início de condução, não a tensão de acionamento para corrente de carga. gm é transcondutância em S, não hFE.')

def enrich_models(components,tokens,detections,distance):
    """Unique nearest compatible assignment, independent of required values."""
    by_id={c['detection_id']:c for c in components};pairs=[]
    from backend.reference_labels import annotation_group_owner,reference_classes
    references={}
    for c in components:
        if (c.get('reference') or {}).get('source')!='ocr':continue
        reference=next((t for t in c.get('ocr_tokens',[]) if c['type'] in reference_classes(t.get('text',''))),None)
        if reference:references[c['detection_id']]=reference
    for i,token in enumerate(tokens):
        try:reading=model_ocr_reading(token.get('text',''));classes=reading['classes']
        except ValueError:continue
        if not classes or float(token.get('confidence',0) or 0)<.35:continue
        compatible_refs={cid:ref for cid,ref in references.items() if by_id[cid]['type'] in classes}
        owner=annotation_group_owner(token,compatible_refs)
        for det in detections:
            if det['class'] not in classes:continue
            if owner is not None and owner!=det['detection_id']:continue
            score=distance(token,det)
            if score is not None:pairs.append((score,-float(token.get('confidence',0)),i,det['detection_id']))
    used_tokens=set();used_components=set()
    from backend.text_assignment import globally_assign
    model_pairs=[(score,confidence,i,cid,tokens[i]) for score,confidence,i,cid in pairs]
    for score,_,i,cid,_ in globally_assign(model_pairs):
        if i in used_tokens or cid in used_components or cid not in by_id:continue
        token=tokens[i];reading=model_ocr_reading(token['text']);by_id[cid]['model']=dict(text=reading['text'],source='ocr',confidence=token['confidence'],
                                                raw_text=token.get('raw_text',reading['raw_text']),corrections=reading['corrections'],
                                                candidates=reading['candidates'],needs_review=reading['needs_review'],
                                                token_id=token.get('token_id'),distance_score=round(score,4))
        if reading['needs_review']:
            by_id[cid]['review_required']=True
            reasons=by_id[cid].setdefault('review_reasons',[])
            if 'model_ocr_needs_confirmation' not in reasons:reasons.append('model_ocr_needs_confirmation')
        if not any(t.get('token_id')==token.get('token_id') for t in by_id[cid].get('ocr_tokens',[])):
            by_id[cid].setdefault('ocr_tokens',[]).append(dict(token,distance_score=round(score,4),association_reason='nearest_compatible_model'))
        used_tokens.add(i);used_components.add(cid)
    # Explicitly labelled parameters, never numbers alone or a model's digits.
    patterns={
      'hfe':r'(?:hfe|beta|β|B)\s*[:=]\s*(.+)',
      'vgs_threshold':r'(?:vgs\s*\(?th\)?|vth)\s*[:=]\s*(.+)',
      'gm':r'gm\s*[:=]\s*(.+)','open_loop_gain':r'(?:aol|avol)\s*[:=]\s*(.+)',
      'series_resistance':r'(?:dcr|esr)\s*[:=]\s*(.+)',
      'saturation_current':r'isat\s*[:=]\s*(.+)',
    }
    candidates=[]
    for i,token in enumerate(tokens):
        if i in used_tokens or float(token.get('confidence',0) or 0)<.45:continue
        for key,pattern in patterns.items():
            match=re.fullmatch(pattern,token.get('text',''),re.IGNORECASE)
            if not match:continue
            reading=parse_measurement(match[1])
            expected={'V':'volt','F':'farad','H':'henry','ohm':'ohm','A':'ampere'}.get(PARAMETERS[key]['unit'])
            if reading and reading['unit']!=expected:continue
            try:number=parameter_number(reading['normalized'] if reading else match[1],key)
            except ValueError:continue
            for det in detections:
                if key not in FIELDS.get(det['class'],[]):continue
                score=distance(token,det)
                if score is not None:candidates.append((score,i,det['detection_id'],key,number,reading))
    assigned=set()
    for _,i,cid,key,number,reading in sorted(candidates):
        if i in used_tokens or (cid,key) in assigned or cid not in by_id:continue
        by_id[cid].setdefault('electrical_parameters',{})[key]=dict(value=number,unit=PARAMETERS[key]['unit'],source='ocr',estimated=False)
        entry=by_id[cid]['electrical_parameters'][key]
        entry['raw_text']=tokens[i].get('raw_text',tokens[i]['text'])
        if tokens[i].get('contextual_corrections'):
            entry.update(corrections=tokens[i]['contextual_corrections'],needs_review=True,
                         detection_context=tokens[i].get('component_context'))
            by_id[cid]['review_required']=True
            reasons=by_id[cid].setdefault('review_reasons',[])
            if 'parameter_label_ocr_corrected' not in reasons:reasons.append('parameter_label_ocr_corrected')
        if reading and reading['corrections']:
            entry.update(corrections=reading['corrections'],needs_review=True)
            by_id[cid]['review_required']=True
            reasons=by_id[cid].setdefault('review_reasons',[])
            if 'parameter_ocr_corrected' not in reasons:reasons.append('parameter_ocr_corrected')
        used_tokens.add(i);assigned.add((cid,key))
    for c in components:
        c.setdefault('model',dict(text='',source='missing',confidence=None))
        c.setdefault('electrical_parameters',{});c.setdefault('parameter_policy',{});c.setdefault('parameter_pending',[])
    return used_tokens
