"""Step 3 - turn the proposals into YOLO labels.

class = referee if the kit matches the match's referee kit (black, or yellow in m3), else player;
then the visual-review corrections below are applied (goalkeepers, players in dark shadow,
coaches/bench staff, wrong ball candidates). Duplicate boxes (IoU > 0.6) are removed.
Validation = the last part of each video (never mixed with training frames).

    python scripts/dataset/3_build_labels.py   -> data/own_dataset
"""
import json, os, shutil, cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
P=json.load(open('pseudo.json'))
def is_ref(v,p):
    L,a,b=p['lab']
    if v=='m3': return L>125 and 112<a<124 and b>=147     # yellow shirt (GK is darker)
    return L<150 and 108<a<132 and b<150                  # black kit
# visual review results ---------------------------------------------------------
NOT_REF = {'m1_f180#11','m1_f180#22','m1_f195#13','m1_f195#24','m1_f210#2','m1_f210#8','m1_f210#23',
           'm1_f135#7','m1_f150#19','m1_f165#19','m1_f180#19','m1_f195#17','m1_f195#21','m1_f210#11','m1_f225#14',  # home GK
           'm2_f520#22','m2_f520#23','m2_f700#21','m3_f700#14','m3_f720#4','m4_f170#10'}
DROP = {'m2_f060#21','m2_f300#23','m2_f320#23'}             # coach / staff
BAD_BALL = {'m4_f000','m4_f010','m4_f020','m1_f540'}
VAL = lambda n: (n[:2]=='m1' and int(n[4:])>=645) or (n[:2] in ('m2','m3') and int(n[4:])>=640) or (n[:2]=='m4' and int(n[4:])>=230)
def iou(p,q):
    ix=max(0,min(p[2],q[2])-max(p[0],q[0])); iy=max(0,min(p[3],q[3])-max(p[1],q[1])); i=ix*iy
    return i/((p[2]-p[0])*(p[3]-p[1])+(q[2]-q[0])*(q[3]-q[1])-i)
root=os.path.abspath('../../data/own_dataset'); shutil.rmtree(root,ignore_errors=True)
for s in ('train','val'):
    os.makedirs(f'{root}/images/{s}'); os.makedirs(f'{root}/labels/{s}')
stats={s:[0,0,0,0] for s in ('train','val')}
for n,d in sorted(P.items()):
    v=n[:2]; W,H=d['W'],d['H']; keep=[]
    for i,p in sorted(enumerate(d['people']),key=lambda t:-t[1]['s']):
        key=f'{n}#{i}'; x1,y1,x2,y2=p['b']
        if key in DROP: continue
        if v=='m1' and (y2>990 or (x2<120 and y1>780)): continue
        ref = is_ref(v,p) and key not in NOT_REF
        if v=='m3' and y2>0.78*H and not ref: continue        # bench / staff below the touchline
        if any(iou(p['b'],q[0])>0.6 for q in keep): continue
        keep.append((p['b'],2 if ref else 1))
    if d['balls'] and n not in BAD_BALL:
        b=max(d['balls'],key=lambda t:t['s'])
        if b['s']>=0.2: keep.append((b['b'],0))
    s='val' if VAL(n) else 'train'
    lines=[f"{c} {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}" for b,c in keep]
    shutil.copy(f'frames/{n}.jpg',f'{root}/images/{s}/{n}.jpg'); open(f'{root}/labels/{s}/{n}.txt','w').write('\n'.join(lines)+'\n')
    stats[s][0]+=1
    for _,c in keep: stats[s][1+c]+=1
print('split: [images, ball, player, referee]',stats)
open(f'{root}/data.yaml','w').write(f"""# Self-collected soccer dataset: frames from 4 match videos, labeled semi-automatically + reviewed by eye
train: images/train
val: images/val
nc: 3
names: {{0: ball, 1: player, 2: referee}}
""")
