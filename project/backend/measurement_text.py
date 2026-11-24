"""Parse electrical values by their unit before repairing the numeric region."""
import math
import re
import unicodedata

DIMENSIONS={'resistor':'ohm','capacitor':'farad','capacitor_polarized':'farad',
            'inductor':'henry','source_dc':'volt','battery':'volt','vcc':'volt'}
SYMBOLS={'ohm':'Ω','farad':'F','henry':'H','volt':'V','ampere':'A'}
UNIT_NAMES={'ohm':('ohms','ohmios','ohm','Ω','Ω','ω','R'),
            'farad':('faradios','farads','farad','F'),
            'henry':('henries','henrys','henrios','henry','H'),
            'volt':('volts','voltios','volt','V'),
            'ampere':('amperes','ampere','amps','amp','A')}
SCALES={'':1.,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'M':1e6,'G':1e9,'meg':1e6,'R':1.}
PREFIX_WORDS={'pico':'p','nano':'n','micro':'u','mili':'m','milli':'m','quilo':'k','kilo':'k','mega':'M','giga':'G'}
CONFUSIONS={'O':'0','o':'0','I':'1','i':'1','l':'1','L':'1','|':'1','Z':'2','z':'2','S':'5','s':'5','B':'8'}


def clean_unicode(text):
    text=unicodedata.normalize('NFKC',str(text)).replace('−','-').replace('—','-')
    return ''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))


def unit_suffix(text):
    """Unit-only tokens are useful anchors even when their number was not read."""
    value=re.sub(r'\s+','',clean_unicode(text))
    candidates=[]
    for dimension,names in UNIT_NAMES.items():
        for name in names:
            if value.casefold().endswith(name.casefold()):
                candidates.append((len(name),dimension,value[:-len(name)],name))
    if not candidates:return None
    _,dimension,head,_=max(candidates,key=lambda c:c[0])
    return dimension,head


def _prefix(text):
    if text in ('µ','μ','u','U'):return 'u'
    if text in ('k','K'):return 'k'
    if text in ('r','R'):return 'R'
    if text in ('p','P','n','N','m','M','g','G'):return 'G' if text=='g' else text.lower() if text in ('P','N') else text
    if text.lower()=='meg':return 'meg'
    return PREFIX_WORDS.get(text.lower())


def unit_anchor(text):
    result=unit_suffix(text)
    if result is None:return None
    dimension,head=result
    if not head or _prefix(head) is not None:return dimension
    return None


def parse_measurement(text,kind=None):
    """Returns a value plus the untouched reading and explicit correction evidence.

    O/0, I/1, S/5, B/8 and Z/2 substitutions apply only to the numeric
    substring with an explicit physical unit. Model codes are never touched.
    Unlabelled values may use their component's dimension, without repairing
    letters. Milli and mega preserve their case.
    """
    raw=str(text).strip()
    # A rating pair (10 16V) is not the number 1016. Keep it unassigned.
    if re.search(r'\d\s+\d',raw):return None
    compact=re.sub(r'\s+','',clean_unicode(raw)).replace(',','.')
    suffix=unit_suffix(compact)
    explicit=suffix is not None
    if explicit:dimension,body=suffix
    else:dimension,body=DIMENSIONS.get(kind),compact
    if not dimension or kind and DIMENSIONS.get(kind)!=dimension:return None
    prefix=''
    # Long prefix words and meg must be tested before single-letter prefixes.
    words=sorted([*PREFIX_WORDS,'meg'],key=len,reverse=True)
    for word in words:
        if body.lower().endswith(word) and len(body)>len(word):
            prefix=_prefix(body[-len(word):]);body=body[:-len(word)];break
    else:
        if body and _prefix(body[-1]) is not None:
            prefix=_prefix(body[-1]);body=body[:-1]
    if prefix=='R' and dimension!='ohm':return None
    # IEC notation: 4R7, 4k7, 4n7F, 2u2H.
    embedded=re.fullmatch(r'([+-]?\d+)([RrKkMmUuµμNnPp])(\d+)',body)
    if embedded:
        if prefix:return None
        prefix=_prefix(embedded[2]);body=embedded[1]+'.'+embedded[3]
        if prefix=='R' and dimension!='ohm':return None
    corrections=[]
    repaired=''
    for i,c in enumerate(body):
        replacement=CONFUSIONS.get(c,c) if explicit else c
        if replacement!=c:corrections.append({'index':i,'from':c,'to':replacement})
        repaired+=replacement
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',repaired):return None
    try:number=float(repaired)*SCALES[prefix]
    except (ValueError,KeyError):return None
    if not math.isfinite(number) or abs(number)>1e15 or number and abs(number)<1e-18:return None
    if dimension not in {'volt','ampere'} and number<=0:return None
    # Keep readable engineering notation while normalized always carries SI.
    number_text=f'{float(repaired):.12g}'
    readable=number_text+prefix+(SYMBOLS[dimension] if explicit else '')
    return dict(text=readable,raw_text=raw,normalized=f'{number:.12g}',unit=dimension,
                unit_explicit=explicit,prefix=prefix,numeric_reading=body,corrections=corrections,
                reading_method='unit_first' if explicit else 'component_context',
                needs_review=bool(corrections) or not explicit)
