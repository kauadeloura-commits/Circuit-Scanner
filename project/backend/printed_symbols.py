"""Digital symbol evidence shared by detection, OCR and terminal placement."""
import cv2,numpy as np
from backend.resistor_geometry import cached_vision

@cached_vision
def plate_kind(image,box):
 from backend.printed_capacitor import plate_polarity
 from backend.capacitor_geometry import capacitor_body_evidence
 shape=plate_polarity(image,box)
 if shape['valid']:return 'capacitor_polarized',shape
 ok,body=capacitor_body_evidence(image,box,maximum_gap_ratio=5)
 if not ok:return None,body
 lengths=body['outer_plate_lengths'];ratio=max(lengths)/max(1,min(lengths))
 profiles=shape.get('plates',[])
 if ratio>=1.40 and (len(profiles)==2 and all(p.get('straight') for p in profiles)):
  return 'battery',dict(body,length_ratio=ratio,method='unequal_straight_opposed_plates')
 return None,body

@cached_vision
def printed_emission(image,box):
 from backend.symbol_semantics import emission_arrows
 pairs=emission_arrows(image,box)
 if pairs:return [dict(point=a['point'].tolist(),bounds=[float(v) for v in a['bounds']]) for a in pairs[0]]
 return []


def correct_types(image,detections,class_ids):
 from backend.drawing_mode import digital
 if not digital():return detections
 result=[]
 for original in detections:
  d=dict(original);kind=d.get('class')
  if d.get('manual_reviewed'):result.append(d);continue
  target=None;evidence=None
  if kind in ('capacitor','capacitor_polarized','source_dc','battery'):target,evidence=plate_kind(image,d['bbox'])
  if not target and kind in ('capacitor','capacitor_polarized'):
   from backend.capacitor_geometry import capacitor_body_evidence
   ok,_=capacitor_body_evidence(image,d['bbox'])
   if not ok:
    from backend.recognition import _validar_classe_geometrica
    valid,score,proof=_validar_classe_geometrica(image,d['bbox'],'diode')
    if valid and proof.get('triangle_pair') and proof.get('bar_and_lead'):target,evidence='diode',proof
  if kind in ('transistor_npn','transistor_pnp'):
   from backend.printed_bjt import read_roles
   anchors,proof=read_roles(image,d['bbox'])
   if anchors and proof.get('detected_arrow_direction'):
    target='transistor_npn' if proof['detected_arrow_direction']=='outward' else 'transistor_pnp';evidence=proof
   from backend.printed_bounds import bjt_bounds
   d=bjt_bounds(d,anchors,proof)
  if kind in ('vcc','gnd','source_dc') and not target:
   alternatives=[a for a in d.get('alternatives',[]) if a.get('class') in ('transistor_npn','transistor_pnp') and a.get('confidence',0)>=.18]
   if alternatives:
    from backend.recognition import _validar_classe_geometrica
    own=_validar_classe_geometrica(image,d['bbox'],kind)
    if own and not own[0]:
     for alt in sorted(alternatives,key=lambda a:a.get('confidence',0),reverse=True):
      physical=_validar_classe_geometrica(image,d['bbox'],alt['class'])
      if physical and physical[0]:target,evidence=alt['class'],physical[2];break
  if kind in ('diode','led'):
   arrows=printed_emission(image,d['bbox'])
   if arrows:target,evidence='led',dict(method='printed_two_outward_emission_arrows',arrows=arrows)
  if target and target!=kind:
   d.update(resolved_from_class=kind,**{'class':target,'class_id':class_ids[target]},class_resolution='printed_physical_symbol_evidence',printed_type_evidence=evidence)
  result.append(d)
 return result


def refine_models_from_text(image,detections,tokens):
 from backend.drawing_mode import digital
 if not digital():return detections
 from backend.component_models import model_classes
 import math
 result=[dict(d) for d in detections]
 for token in tokens:
  if float(token.get('confidence',0))<.80:continue
  if __import__('re').fullmatch(r'L\d+',str(token.get('text','')),__import__('re').I):
   from backend.recognition import _validar_geometria_indutor
   b=token.get('bbox') or {};ranked=[]
   for d in result:
    if d.get('manual_reviewed') or d.get('class') not in {'resistor','inductor'}:continue
    a=d['bbox'];distance=math.hypot(max(a['x1']-b['x2'],b['x1']-a['x2'],0),max(a['y1']-b['y2'],b['y1']-a['y2'],0))
    if distance<=max(20,min(60,min(a['x2']-a['x1'],a['y2']-a['y1']))):ranked.append((distance,d))
   if ranked:
    d=min(ranked,key=lambda item:item[0])[1];valid,score,proof=_validar_geometria_indutor(image,d['bbox'])
    if valid and proof.get('connected_curve_with_two_leads'):
     d.update(resolved_from_class=d['class'],**{'class':'inductor','class_id':5},class_resolution='explicit_L_reference_and_continuous_coil',printed_type_evidence=proof)
   continue
  classes=model_classes(str(token.get('text','')))
  if len(classes)!=1:continue
  kind=next(iter(classes))
  if kind not in {'transistor_npn','transistor_pnp'}:continue
  b=token.get('bbox') or {};ranked=[]
  if not all(k in b for k in ('x1','y1','x2','y2')):continue
  for d in result:
   if d.get('manual_reviewed') or d.get('class') not in {'transistor_npn','transistor_pnp'}:continue
   a=d['bbox'];distance=math.hypot(max(a['x1']-b['x2'],b['x1']-a['x2'],0),max(a['y1']-b['y2'],b['y1']-a['y2'],0))
   if distance<=max(30,min(70,min(a['x2']-a['x1'],a['y2']-a['y1']))):ranked.append((distance,d))
  if ranked:
   d=min(ranked,key=lambda item:item[0])[1]
   d.update(resolved_from_class=d['class'],**{'class':kind,'class_id':10 if kind=='transistor_npn' else 11},class_resolution='explicit_ocr_model_and_local_catalog',printed_model_text=token['text'])
 return result
