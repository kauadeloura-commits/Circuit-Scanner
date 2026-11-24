"""Human-readable pin-to-net listing, without a second electrical simulator."""
from backend.kicad_export import model_text

def connection_preview(circuit):
    nets={m:n['name'] for n in circuit['nets'] for m in n['members']}
    lines=[]
    for c in circuit['components']:
        fields=[c['id'],c['type'],c.get('value_text') or c.get('value'),model_text(c)]
        lines.append(' | '.join(str(x) for x in fields if x)+' | '+', '.join(p+' → '+nets[c['id']+'.'+p] for p in c['pins']))
    return dict(netlist='\n'.join(lines),nodes={n['name']:n['name'] for n in circuit['nets']},
                components=[dict(component_id=c['id']) for c in circuit['components']],warnings=[])
