"""Insulate a clipped connector frame only with two explicit numbered ports.

A U-shaped conductor alone is never enough. Native frame pixels, parallel
incoming leads, a white frame interior and two confident OCR numbers must
agree. No expected circuit connections or component IDs enter this reader.
"""
import cv2,numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def connector_barriers(image):
 from backend.printed_capacitor import plate_ink
 from backend.local_word_ocr import read_words
 barrier=np.zeros(image.shape[:2],np.uint8);evidence=[]
 for quarter in range(4):
  view=np.ascontiguousarray(np.rot90(image,quarter));ink=plate_ink(view);h,w=ink.shape
  if min(h,w)<30:continue
  candidates=[]
  for x in range(max(12,w-min(10,max(4,int(w*.025)))),w-1):
   rows=np.flatnonzero(ink[:,x]>0)
   runs=np.split(rows,np.flatnonzero(np.diff(rows)>1)+1)
   for run in runs:
    if not 8<=len(run)<=60:continue
    y,z=int(run[0]),int(run[-1]);spacing=z-y
    if y<spacing or z>=h-1:continue
    def horizontal(level,left,right):
     section=ink[max(0,level-1):min(h,level+2),left:right]
     return float(np.mean(np.any(section>0,axis=0))) if section.size else 0.
    if min(horizontal(y,x-12,x-2),horizontal(z,x-12,x-2))<.90:continue
    if min(horizontal(y,x+1,w),horizontal(z,x+1,w))<.60:continue
    interior=ink[y+3:z-2,x+2:w]
    if interior.size<3 or np.mean(interior==0)<.80:continue
    if any(abs(x-c[0])<4 and abs(y-c[1])<4 and abs(z-c[2])<4 for c in candidates):continue
    candidates.append((x,y,z))
  for x,y,z in candidates:
   spacing=z-y;crop_h=max(7,int(round(spacing*.70)));crop_w=max(9,int(round(spacing*.90)))
   patches=[view[level-crop_h:level-1,max(0,x-crop_w):x-1] for level in (y,z)]
   rows=read_words(patches)
   if len(rows)!=2 or any(not r or not str(r.get('text','')).isdigit() or float(r.get('confidence',0))<.99 or float(r.get('min_character_confidence',0))<.97 for r in rows):continue
   numbers=[int(r['text']) for r in rows]
   if numbers[0]==numbers[1] or any(n>99 for n in numbers):continue
   cut=np.zeros(ink.shape,np.uint8)
   cut[max(0,y-1):min(h,z+2),max(0,x-1):w]=255
   barrier=np.maximum(barrier,np.ascontiguousarray(np.rot90(cut,-quarter)))
   evidence.append(dict(kind='clipped_connector_insulating_frame',port_numbers=numbers,
                        number_confidences=[r['confidence'] for r in rows],quarter_turns=quarter,
                        observed_frame=dict(x=x,y1=y,y2=z),method='visible_rectangular_wall_parallel_leads_and_two_ocr_port_numbers'))
 return barrier,evidence
