"""Reference grammar and local annotation geometry, independent of OCR engine."""
import math
import re

TRANSISTORS = {'transistor_npn', 'transistor_pnp', 'mosfet_n', 'mosfet_p'}
PREFIXES = {
    'R': {'resistor'}, 'C': {'capacitor', 'capacitor_polarized'},
    'L': {'inductor'}, 'Q': TRANSISTORS, 'T': TRANSISTORS,
    'D': {'diode', 'led'}, 'LED': {'led'}, 'M': {'mosfet_n', 'mosfet_p'},
    'V': {'source_dc', 'battery', 'vcc'}, 'VS': {'source_dc', 'battery'},
    'BAT': {'battery'}, 'BT': {'battery'}, 'U': {'opamp'}, 'IC': {'opamp'},
    'CP': {'capacitor_polarized'},
}


def printed_reference_classes(text):
    """Explicit conventional printed designators, never a guessed prefix."""
    key=re.sub(r'[\s_{}]','',str(text)).upper()
    if re.fullmatch(r'R(?:B|C|E|L|S|IN|OUT)\d*',key):return {'resistor'}
    if re.fullmatch(r'C(?:IN|OUT|E|B|L)\d*',key):return {'capacitor','capacitor_polarized'}
    if key in {'Q','T'}:return set(TRANSISTORS)
    return set()


def reference_classes(text):
    """A complete identifier requires a real numeric suffix, not a model fragment."""
    key = re.sub(r'\s+', '', str(text)).upper()
    if re.fullmatch(r'(?:GND|[+-]?V(?:CC|EE|DD|SS))\d*', key):
        return {'gnd'} if key.startswith('GND') else {'vcc'}
    from backend.drawing_mode import digital
    if digital():
        conventional=printed_reference_classes(key)
        if conventional:return conventional
    match = re.fullmatch(r'(LED|BAT|BT|VS|CP|IC|[RCLQTDVMU])(\d{1,5})', key)
    if not match:
        return set()
    # A known component model always takes precedence over its leading letter.
    from backend.component_models import CATALOG, model_classes
    if key in CATALOG or model_classes(key):
        return set()
    return set(PREFIXES[match[1]])


def edge_distance(a, b):
    return math.hypot(max(a['x1']-b['x2'], b['x1']-a['x2'], 0),
                      max(a['y1']-b['y2'], b['y1']-a['y2'], 0))


def local_distance(token, detection):
    """Use the photographed text scale, with a bounded component neighborhood."""
    a, b = token['bbox'], detection['bbox']
    w, h = b['x2']-b['x1'], b['y2']-b['y1']
    th = max(4., a['y2']-a['y1'])
    distance = edge_distance(a, b)
    limit = min(260., max(60., 4*th, 1.25*max(w, h)))
    if distance > limit:
        return None
    center = math.hypot((a['x1']+a['x2']-b['x1']-b['x2'])/2,
                        (a['y1']+a['y2']-b['y1']-b['y2'])/2)
    return distance + .08*center


def annotation_group_owner(token, references):
    """Only a nearby, aligned reference may anchor the value beside/under it."""
    a = token['bbox']; options = []
    from backend.drawing_mode import digital
    printed=digital();degrees=float(token.get('text_quarter_turns',0))*90 if 'text_quarter_turns' in token else float(token.get('text_rotation_degrees',0));quarter=round(degrees/90)%4
    theta=math.radians(degrees);co,si=math.cos(theta),math.sin(theta)
    def oriented(box,polygon=None):
        points=polygon or [(box[x],box[y]) for x,y in [('x1','y1'),('x1','y2'),('x2','y1'),('x2','y2')]]
        points=[(co*x+si*y,-si*x+co*y) for x,y in points]
        return dict(x1=min(p[0] for p in points),y1=min(p[1] for p in points),x2=max(p[0] for p in points),y2=max(p[1] for p in points))
    a=oriented(a,token.get('text_polygon'))
    for cid, ref in references.items():
        if printed and int(ref.get('text_quarter_turns',0))%4!=quarter:continue
        b = oriented(ref['bbox'],ref.get('text_polygon')); h = max(4., a['y2']-a['y1'], b['y2']-b['y1'])
        dx = max(a['x1']-b['x2'], b['x1']-a['x2'], 0)
        dy = max(a['y1']-b['y2'], b['y1']-a['y2'], 0)
        ox = min(a['x2'], b['x2'])-max(a['x1'], b['x1'])
        oy = min(a['y2'], b['y2'])-max(a['y1'], b['y1'])
        same_phrase = token.get('parent_token_id') is not None and token.get('parent_token_id') == ref.get('parent_token_id')
        below=ox>0 and dy<=1.8*h and (not printed or b['y2']<=a['y1']+.5*h)
        after=oy>.25*h and dx<=1.8*h and (not printed or b['x2']<=a['x1']+.3*h)
        if same_phrase or below or after:
            options.append((edge_distance(a, b)/h, cid))
    options.sort()
    if not options or options[0][0] > 1.8:
        return None
    if len(options) > 1 and options[1][0]-options[0][0] < .6:
        return None
    return options[0][1]
