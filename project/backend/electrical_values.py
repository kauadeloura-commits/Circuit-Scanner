"""Parse reviewed electrical quantities in SI units."""
import math,re
UNITS={'resistor':('ohm','ohms','Ω','Ω'), 'capacitor':('F','farad','farads'),'capacitor_polarized':('F','farad','farads'),
       'inductor':('H','henry','henries'),'source_dc':('V','volt','volts'),'battery':('V','volt','volts'),'vcc':('V','volt','volts'),'time':('s','second','seconds')}
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
    if kind not in {'source_dc','battery','vcc'} and number<=0:
        raise ValueError('O valor deve ser positivo em '+kind+'.')
    return number

