"""Step 2 - crop every referee candidate (kit-colour rule per match) into one sheet per match
so that each crop can be checked by eye. Mistakes found here go into NOT_REF / DROP in step 3.

    python scripts/dataset/2_review_referee_candidates.py m3   -> review_m3.jpg
"""
import json, cv2, numpy as np, sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
P=json.load(open('pseudo.json'))
def is_ref(v,p):
    L,a,b=p['lab']
    if v=='m3': return L>125 and 112<a<124 and b>=147      # yellow shirt (GK is darker)
    return L<150 and 108<a<132 and b<150                   # black/dark kit
v=sys.argv[1]; items=[(n,i,p) for n,d in sorted(P.items()) if n.startswith(v) for i,p in enumerate(d['people']) if is_ref(v,p)]
tiles=[]; cache={}
for n,i,p in items:
    if n not in cache: cache[n]=cv2.imread(f'frames/{n}.jpg')
    f=cache[n]; x1,y1,x2,y2=p['b']; pad=int(0.25*(y2-y1))
    c=f[max(0,y1-pad):y2+pad,max(0,x1-pad):x2+pad]; t=np.zeros((112,60,3),np.uint8); t[:90]=cv2.resize(c,(60,90))
    cv2.putText(t,n[3:],(1,100),0,0.3,(255,255,255),1); cv2.putText(t,f'#{i}',(1,110),0,0.32,(0,255,255),1); tiles.append(t)
while len(tiles)%20: tiles.append(np.zeros((112,60,3),np.uint8))
cv2.imwrite(f'review_{v}.jpg',np.vstack([np.hstack(tiles[i:i+20]) for i in range(0,len(tiles),20)])); print(v,len(items))
