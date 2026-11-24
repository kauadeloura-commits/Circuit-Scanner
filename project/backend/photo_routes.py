"""Keep verified photographed wire bends; prune unanchored ink and ground buses."""
import math

def captured_paths(ids,segments,convert,terminals,grounds):
    if not ids:return []
    edges=set();nodes=set()
    def edge(a,b):
        if a!=b:edges.add(tuple(sorted((a,b))));nodes.update((a,b))
    for wid in set(ids):
        s=segments.get(wid,{})
        raw=s.get('points',[])
        if len(raw)<2:continue
        # Suppress raster stair steps while retaining substantial corners.
        import cv2,numpy as np
        raw=np.array([[p['x'],p['y']] for p in raw],np.float32)
        simplified=cv2.approxPolyDP(raw,5,False).reshape(-1,2)
        path=[convert(p) for p in simplified]
        for a,b in zip(path,path[1:]):
            corner=(b[0],a[1]) if abs(b[0]-a[0])>=abs(b[1]-a[1]) else (a[0],b[1])
            edge(a,corner);edge(corner,b)
    if not nodes:return []
    # Attach only the near end of the photographed branch to a reviewed pin.
    for p in terminals:
        q=min(nodes,key=lambda q:math.dist(p,q))
        if math.dist(p,q)<=12:
            corner=(p[0],q[1]);edge(p,corner);edge(corner,q)
    parent={};anchored={};grounded={}
    def find(p):
        parent.setdefault(p,p);anchored.setdefault(p,{p} if p in terminals else set());grounded.setdefault(p,p in grounds)
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    forest=[]
    for a,b in sorted(edges,key=lambda e:(math.dist(*e),e)):
        ra,rb=find(a),find(b)
        if ra==rb or grounded[ra] and grounded[rb]:continue
        parent[ra]=rb;anchored[rb]|=anchored[ra];grounded[rb]|=grounded[ra];forest.append((a,b))
    # A fragment cannot add a connection if fewer than two actual pins use it.
    forest=[(a,b) for a,b in forest if len(anchored[find(a)])>=2]
    while True:
        degree={}
        for a,b in forest:
            for p in (a,b):degree[p]=degree.get(p,0)+1
        leaves={p for p,d in degree.items() if d==1 and p not in terminals}
        if not leaves:break
        forest=[(a,b) for a,b in forest if a not in leaves and b not in leaves]
    return [[a,b] for a,b in forest]
