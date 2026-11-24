"""Explicit approximate electrical models fitted to the reviewed OCR parameters."""
import math
from backend.component_models import resolve_parameters, automatic_model_fields

def parameters(component):
    meta=component.get('metadata') or {};model=meta.get('model') or {}
    filled=automatic_model_fields(dict(type=component['type'],**{k:meta[k] for k in ('model','electrical_parameters','parameter_policy') if k in meta}))
    meta=filled;model=filled.get('model') or {}
    name=model.get('text','') if isinstance(model,dict) else str(model)
    result=resolve_parameters(component['type'],name,meta.get('electrical_parameters'),meta.get('parameter_policy'))
    # The automatic resolver has already regenerated estimates with provenance.
    if 'parameter_pending' in meta:result['parameters']=meta['electrical_parameters']
    result['pending']=meta.get('parameter_pending',result['pending'])
    return name,result
def falstad_fit(component):
    _,resolved=parameters(component)
    values={k:p['value'] for k,p in resolved['parameters'].items()}
    def need(key):
        if key not in values:raise ValueError(component['id']+': informe '+key+' em Valores e modelos.')
        return values[key]
    kind=component['type']
    if kind.startswith('transistor_'):return {'BF':need('hfe')}
    if kind=='opamp':return {'open_loop_gain':need('open_loop_gain')}
    threshold=need('vgs_threshold');kp=values.get('mos_kp')
    if kp is None:
        drive=abs(need('gate_drive_voltage'))-abs(threshold);rds=need('rds_on')
        if drive<=0 or rds<=0:raise ValueError(component['id']+': confira VGS de acionamento e RDS(on).')
        kp=1/(rds*drive)
    if not math.isfinite(kp) or not 0<kp<=1e8:raise ValueError('KP fora da faixa permitida.')
    return {'VTO':threshold,'KP':kp}
