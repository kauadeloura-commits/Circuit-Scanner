"""Exact Zhang-Suen rules through two 256-entry neighborhood tables."""
import numpy as np


def _tables():
    indices=np.arange(256,dtype=np.uint8)
    p=[((indices>>i)&1) for i in range(8)]
    degree=sum(p);transitions=sum(((p[i]==0)&(p[(i+1)%8]==1)).astype(np.uint8) for i in range(8))
    common=(degree>=2)&(degree<=6)&(transitions==1)
    return (common&~(p[0]&p[2]&p[4]).astype(bool)&~(p[2]&p[4]&p[6]).astype(bool),
            common&~(p[0]&p[2]&p[6]).astype(bool)&~(p[0]&p[4]&p[6]).astype(bool))

_DELETE=_tables()


def thin_binary(mask,max_iter=200):
    image=(mask>0).astype(np.uint8)
    if not image.size:return image
    image[[0,-1],:]=0;image[:,[0,-1]]=0
    rows=np.flatnonzero(np.any(image,axis=1));cols=np.flatnonzero(np.any(image,axis=0))
    if not len(rows) or not len(cols):return image
    # Pixels outside the occupied rectangle are already zero. One zero border
    # retains exactly the same eight-neighbor rules and output coordinates.
    work=image[max(0,rows[0]-1):min(image.shape[0],rows[-1]+2),max(0,cols[0]-1):min(image.shape[1],cols[-1]+2)]
    center=work[1:-1,1:-1]
    iteration=0
    while max_iter is None or iteration<max_iter:
        iteration+=1
        changed=False
        for rule in _DELETE:
            neighbors=(work[:-2,1:-1],work[:-2,2:],work[1:-1,2:],work[2:,2:],
                       work[2:,1:-1],work[2:,:-2],work[1:-1,:-2],work[:-2,:-2])
            code=neighbors[0].copy()
            for index,neighbor in enumerate(neighbors[1:],1):code|=neighbor<<index
            remove=(center>0)&rule[code]
            if np.any(remove):center[remove]=0;changed=True
        if not changed:break
    return image
