# WL-kNN baseline + size-only screen. Label-free scores, fit on training normals only (majority class = normal),
# stratified 5-fold CV (random_state 42, same folds as gladGAN). Fixed settings, no tuning: 3 WL iterations
# on atom types, hashed + L2-normalised histograms, mean distance to the 5 nearest training normals.
#   python baselines/wl_knn.py BZR COX2 ...
import sys, numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
def load(ds):
    gi=np.loadtxt(f'{D}/{ds}/{ds}_graph_indicator.txt',dtype=int)
    y=np.loadtxt(f'{D}/{ds}/{ds}_graph_labels.txt',dtype=int)
    nl=np.loadtxt(f'{D}/{ds}/{ds}_node_labels.txt',dtype=int,delimiter=',',ndmin=2)[:,0]
    A=np.loadtxt(f'{D}/{ds}/{ds}_A.txt',dtype=int,delimiter=',')
    adj=defaultdict(list)
    for a,b in A: adj[a-1].append(b-1)
    return gi-1,y,nl,adj
def wl(gi,nl,adj,it=3):
    lab=[str(l) for l in nl]; feats=[Counter() for _ in range(gi.max()+1)]
    for t in range(it+1):
        for v,l in enumerate(lab): feats[gi[v]][f'{t}:{l}']+=1
        lab=[lab[v]+'|'+','.join(sorted(lab[u] for u in adj[v])) for v in range(len(lab))]
    return feats
for ds in sys.argv[1:]:
    gi,y,nl,adj=load(ds); n=np.bincount(gi).astype(float)
    vals,c=np.unique(y,return_counts=True); anom=(y!=vals[c.argmax()]).astype(int)
    X=normalize(FeatureHasher(2**18,input_type='dict').transform(wl(gi,nl,adj)))
    ln=np.log(np.maximum(n,1)); sz,dirsz,wk=[],[],[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(n,anom):
        nt=tr[anom[tr]==0]
        sz.append(roc_auc_score(anom[te],np.abs(ln[te]-ln[nt].mean())))
        a=roc_auc_score(anom[te],ln[te]); dirsz.append(max(a,1-a))
        nn=NearestNeighbors(n_neighbors=5).fit(X[nt]); wk.append(roc_auc_score(anom[te],nn.kneighbors(X[te])[0].mean(1)))
    print(f'{ds:13s} graphs={len(y):5d} anom={anom.mean():.2f} n={n.mean():5.1f}  size(label-free)={np.mean(sz):.3f}  size(best dir, oracle)={np.mean(dirsz):.3f}  WL-kNN={np.mean(wk):.3f} ± {np.std(wk,ddof=1):.3f}  size(label-free) sd={np.std(sz,ddof=1):.3f}',flush=True)
