"""Keep a corroborated complete value ahead of a clipped prefix of it."""
import math
from backend.measurement_text import parse_measurement


def complete_span_priorities(tokens):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return {}
    priorities={}
    for full in tokens:
        text=''.join(full.get('text','').split())
        value=parse_measurement(text,'resistor')
        readers={r.get('engine') for r in full.get('word_evidence',[]) if r.get('engine')}
        if not value or not value.get('prefix') or full.get('confidence',0)<.90 or len(readers)<2:continue
        a=full['bbox'];angle=math.radians(float(full.get('text_rotation_degrees',0)))
        unit=(math.cos(angle),math.sin(angle));axis='x' if abs(unit[0])>=abs(unit[1]) else 'y'
        sign=unit[0] if axis=='x' else unit[1];end=axis+('2' if sign>=0 else '1')
        height=min(a['x2']-a['x1'],a['y2']-a['y1'])
        for fragment in tokens:
            if fragment is full:continue
            short=''.join(fragment.get('text','').split())
            if not text.lower().startswith(short.lower()) or not text[len(short):].isdigit():continue
            shorter=parse_measurement(short,'resistor')
            if not shorter or shorter.get('prefix')!=value['prefix']:continue
            b=fragment['bbox']
            inter=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
            area=min((a['x2']-a['x1'])*(a['y2']-a['y1']),(b['x2']-b['x1'])*(b['y2']-b['y1']))
            extension=(a[end]-b[end])*(1 if sign>=0 else -1)
            if inter/max(1,area)<.65 or extension<max(3,height*.20):continue
            priorities[id(full)]=1;priorities[id(fragment)]=-1
    return priorities
