"""Reuse schematic presentation while the reviewed circuit/files are unchanged."""
import copy,json
from functools import lru_cache
from pathlib import Path
from backend.circuit_review import revision
from backend.digital_twin import make_diagram,read_geometry,VERSION

def stamps(folder,names):
    values=[]
    for name in names:
        try:
            s=(folder/name).stat();values.append((name,s.st_mtime_ns,s.st_ctime_ns,s.st_size))
        except FileNotFoundError:values.append((name,None))
    return tuple(values)

@lru_cache(maxsize=16)
def _diagram(folder,rev,body,versions,layout_version):
    return make_diagram(json.loads(body),*read_geometry(Path(folder)))

def diagram_for(folder,circuit):
    folder=Path(folder)
    versions=stamps(folder,('visual_geometry.json','connectivity.json','wires.json','circuit.json'))
    body=json.dumps(circuit,sort_keys=True,ensure_ascii=False)
    return copy.deepcopy(_diagram(str(folder.resolve()),revision(circuit),body,versions,VERSION))
