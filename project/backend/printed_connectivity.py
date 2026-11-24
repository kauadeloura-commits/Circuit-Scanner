"""Digital nets traced on original conductor pixels outside component bodies.

No distance-only connection or completion from expected circuit behavior.
Disconnected crossings retain their two independent conductor directions.
"""
from copy import deepcopy
from datetime import datetime
import math,cv2,numpy as np


def build(image,wires,terminals):
 from backend.circuit_connectivity import _Groups,_outside_normal,source_signature
 from backend.circuit_types import PIN_SCHEMAS
 from backend.printed_capacitor import plate_ink
 from backend.junction_detection import _source_branch_evidence
 components=deepcopy(terminals['components']);mask=plate_ink(image)
 from backend.printed_enclosures import connector_barriers
 enclosure_mask,enclosures=connector_barriers(image)
 mask[enclosure_mask>0]=0
 # Symbols insulate their own pins. Real contact pixels at the outer edge
 # remain accessible, but the internal fork/plates never become a wire.
 for c in components:
  if c.get('type') in {'vcc','gnd'}:continue
  circle=c.get('printed_symbol_geometry') or {}
  if circle.get('circle_center') and circle.get('circle_radius'):
   cv2.circle(mask,tuple(np.rint(circle['circle_center']).astype(int)),int(math.ceil(circle['circle_radius']*1.14)),0,-1)
   continue
  b=c['bbox'];x1,y1=max(0,int(math.floor(b['x1']))+2),max(0,int(math.floor(b['y1']))+2)
  x2,y2=min(mask.shape[1],int(math.ceil(b['x2']))-1),min(mask.shape[0],int(math.ceil(b['y2']))-1)
  if x2>x1 and y2>y1:mask[y1:y2,x1:x2]=0
 split=[]
 for node in (wires.get('stage9') or {}).get('nodes',[]):
  p=node.get('position')
  if not p:continue
  proof=_source_branch_evidence(image,p)
  explicit=node.get('decision_source','').startswith('manual')
  status=node.get('electrical_status')
  separate=status=='crossing_unconnected' and explicit or not explicit and proof.get('branch_count')==4 and not proof.get('dot_evidence',{}).get('dot_present')
  if not separate:continue
  x,y=int(round(p['x'])),int(round(p['y']))
  radius=max(3,int(math.ceil(proof.get('dot_evidence',{}).get('center_radius_px',0)))+2) if explicit else 3
  if any(c['bbox']['x1']-4<=x<=c['bbox']['x2']+4 and c['bbox']['y1']-4<=y<=c['bbox']['y2']+4 for c in components):continue
  cv2.circle(mask,(x,y),radius,0,-1);split.append((x,y,radius))
 count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
 groups=_Groups();issues=[];connections=[];ground=[];net_labels={}
 def region_ids(p,r=2):
  x,y=np.rint(p).astype(int)
  return set(int(v) for v in np.unique(labels[max(0,y-r):y+r+1,max(0,x-r):x+r+1]) if v)
 for x,y,r in split:
  # Find the four observed arms on an angular ring. Pair opposite arms,
  # including diagonal and rotated drawings, without joining their crossing.
  ring=np.zeros(mask.shape,np.uint8)
  cv2.circle(ring,(x,y),r+3,255,3)
  ring=(ring>0)&(mask>0)
  n,arm_labels,_,_=cv2.connectedComponentsWithStats(ring.astype(np.uint8),8)
  arms=[]
  for i in range(1,n):
   yy,xx=np.nonzero(arm_labels==i)
   if len(xx)<2:continue
   direction=np.array([xx.mean()-x,yy.mean()-y]);direction/=max(1,np.linalg.norm(direction))
   regions=set(int(v) for v in np.unique(labels[yy,xx]) if v)
   if len(regions)==1:arms.append((direction,next(iter(regions))))
  if len(arms)==4:
   pairings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
   ranked=sorted((sum(float(arms[a][0]@arms[b][0]) for a,b in pairing),pairing) for pairing in pairings)
   _,pairing=ranked[0]
   if all(arms[a][0]@arms[b][0]<-.75 for a,b in pairing):
    for a,b in pairing:groups.union(['P:'+str(arms[a][1]),'P:'+str(arms[b][1])])
 wire_pixels={}
 for segment in wires.get('segments',[]):
  observed=set()
  for p in segment.get('points',[]):observed.update(region_ids((p['x'],p['y']),1))
  # A graph segment may cross two disconnected arms; do not join these by
  # segment identity. Use it for drawing attribution only.
  wire_pixels[str(segment['wire_id'])]=observed
 for c in components:
  cid=c['component_id'];pins=c.get('terminals',[])
  expected=PIN_SCHEMAS.get(c['type'])
  if expected is not None and {t['pin'] for t in pins}!=set(expected):issues.append(dict(code='pin_count_mismatch',component_id=cid,message='Terminais incompletos.'))
  if c.get('review_required') or c.get('terminal_detection',{}).get('review_required'):issues.append(dict(code='component_pending',component_id=cid,message='Confira a identidade dos terminais.'))
  for t in pins:
   ref=cid+':'+t['pin'];key='T:'+ref;groups.find(key)
   p=np.array([t['position']['x'],t['position']['y']],float);normal=_outside_normal(p,c['bbox'])
   circle=c.get('printed_symbol_geometry') or {}
   if circle.get('circle_center'):
    normal=p-np.asarray(circle['circle_center']);normal/=max(1,np.linalg.norm(normal))
   candidates=set()
   # Search the actual lead just outside this face, within three pixels
   # perpendicular to it. Never snap to a different nearby conductor.
   samples=[]
   for step in np.arange(0,7,.5):
    q=p+normal*step;x,y=np.rint(q).astype(int)
    for side in (-1,0,1):
     r=np.rint(q+np.array([-normal[1],normal[0]])*side).astype(int)
     if 0<=r[0]<labels.shape[1] and 0<=r[1]<labels.shape[0] and labels[r[1],r[0]]:
      samples.append((step,int(labels[r[1],r[0]])))
   if samples:
    first=min(s[0] for s in samples);candidates={v for s,v in samples if s<=first+1}
   roots={groups.find('P:'+str(i)) for i in candidates}
   attached=len(roots)==1
   if attached:groups.union([key,*roots])
   else:issues.append(dict(code='unconnected_pin' if not roots else 'ambiguous_pin',terminal_ref=ref,message='Ligação externa não confirmada nos pixels.'))
   record=dict(terminal_ref=ref,component_id=cid,pin=t['pin'],electrical_role=t.get('electrical_role'),position=deepcopy(t['position']),
               status='attached_original_pixels' if attached else 'unconnected' if not roots else 'ambiguous',wire_id=None,distance_px=float(first) if samples else None)
   connections.append(record)
   if c['type']=='gnd':ground.append(key)
   label=c.get('net_label') or t.get('net_label')
   if label:net_labels.setdefault(str(label),[]).append(key)
 groups.union(ground)
 for label,keys in net_labels.items():groups.union(keys+ground if label.upper() in ('GND','0') else keys)
 roots={}
 for record in connections:
  root=groups.find('T:'+record['terminal_ref']);roots.setdefault(root,dict(wire_ids=[],terminal_refs=[],labels=[]))['terminal_refs'].append(record['terminal_ref'])
 for root,net in roots.items():
  for wid,pixels in wire_pixels.items():
   if any(groups.find('P:'+str(i))==root for i in pixels):net['wire_ids'].append(wid)
 ground_root=groups.find(ground[0]) if ground else None
 nets=[];names={};number=0
 for root,net in sorted(roots.items(),key=lambda item:(item[0]!=ground_root,min(item[1]['terminal_refs']))):
  is_ground=root==ground_root
  if not is_ground:number+=1
  name='GND' if is_ground else 'NET_'+str(number);names[root]=name
  net.update(name=name,is_ground=is_ground,member_count=len(net['terminal_refs']),terminal_refs=sorted(net['terminal_refs']),wire_ids=sorted(set(net['wire_ids'])))
  nets.append(net)
 for record in connections:
  net=names[groups.find('T:'+record['terminal_ref'])];record['net']=net
  matching=next(n for n in nets if n['name']==net)['wire_ids']
  if matching:record['wire_id']=matching[0]
 unconnected=sum(r['status']=='unconnected' for r in connections);ambiguous=sum(r['status']=='ambiguous' for r in connections)
 return dict(schema_version='stage10-connectivity-1.0',created_at=datetime.now().astimezone().isoformat(timespec='seconds'),project_id=wires.get('project_id'),source_signature=source_signature(wires,terminals),
  method='printed_original_pixel_paths_with_body_insulation_and_split_crossings',components=components,nets=nets,terminal_connections=connections,warnings=issues,
  net_count=len(nets),terminal_count=len(connections),attached_terminal_count=len(connections)-unconnected-ambiguous,unconnected_terminal_count=unconnected,
  ambiguous_terminal_count=ambiguous,wire_only_net_count=0,review_count=len(issues),recovered_wire_bridges=[],recovered_observed_leads=[],recovered_coincident_contacts=[],
  disconnected_crossings=[dict(x=x,y=y) for x,y,r in split],auxiliary_insulating_enclosures=enclosures,
  criterion_met=bool(connections and not unconnected and not ambiguous and not any(i['code']=='pin_count_mismatch' for i in issues)))
