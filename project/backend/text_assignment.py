"""One-to-one text association across the whole circuit, with abstention."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def globally_assign(pairs,*,value_pairs=False):
    """Return original pair tuples; never fabricate a token or a label.

    Reference tuples: distance, negative confidence, token id, detection id,
    token. Value tuples prepend unit priority. Dummies preserve unassigned
    components and texts when there is no compatible edge.
    """
    if not pairs:return []
    offset=int(value_pairs)
    tokens=list(dict.fromkeys(p[offset+2] for p in pairs))
    detections=list(dict.fromkeys(p[offset+3] for p in pairs))
    ti={t:i for i,t in enumerate(tokens)};di={d:i for i,d in enumerate(detections)}
    n,m=len(tokens),len(detections);forbidden=1e9;unmatched=10000.
    cost=np.full((n,m+n),forbidden,dtype=float)
    best={}
    strong_dimensioned={p[4] for p in pairs if value_pairs and p[0]<2 and -float(p[2])>=.90}
    for p in pairs:
        i,j=ti[p[offset+2]],di[p[offset+3]]
        priority=float(p[0]) if value_pairs else 0.
        # Explicit units remain preferred, then physical distance. Confidence
        # breaks close spatial ties without moving a remote annotation.
        # An uncertain glyph decoded as an explicit ohm sign must not defeat
        # a confidently read '10k' merely because that word omits the unit.
        # Units are a small tie-break after compatible dimension, locality and
        # observed reading confidence, all on the same numerical scale.
        bare_penalty=1000. if value_pairs and priority==2 and p[4] in strong_dimensioned else 0.
        score=bare_penalty+priority*.15+float(p[offset])+5*float(p[offset+1])
        if score<cost[i,j]:cost[i,j]=score;best[i,j]=p
    for i in range(n):cost[i,m+i]=unmatched
    rows,cols=linear_sum_assignment(cost)
    selected=[best[i,j] for i,j in zip(rows,cols) if j<m and (i,j) in best]
    return sorted(selected)
