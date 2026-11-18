"""Step 1 - collect frames from 4 match videos and propose boxes.

- m1: Gladbach vs Wolfsburg (the demo match) - every 15th frame, frames 235-389 skipped
      (the 5-second demo clip is frames 250-374, so it never leaks into training)
- m2: Union Berlin vs Wolfsburg, m3: Mainz vs (black kits), m4: Man City vs Man United
Boxes are only *proposals* from a helper detector (used like Roboflow's auto-label);
classes are decided in step 3 from kit colours + visual review.

    python scripts/dataset/1_collect_frames.py --helper weights/public_stage2_best.pt
"""
import cv2, json, numpy as np, os, argparse
from ultralytics import YOLO
ap=argparse.ArgumentParser(); ap.add_argument('--helper',default='weights/public_stage2_best.pt'); ap.add_argument('--videos',default='videos')
A=ap.parse_args(); m=YOLO(A.helper)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
vd=os.path.abspath(os.path.join('../..',A.videos)) if not os.path.isabs(A.videos) else A.videos
V={'m1':(f'{vd}/08fd33_4.mp4',15,(235,390)),'m2':(f'{vd}/121364_0.mp4',20,None),
   'm3':(f'{vd}/cut_1.mp4',20,None),'m4':(f'{vd}/CityUtdR.mp4',10,None)}
os.makedirs('frames',exist_ok=True); out={}
def grass(f):
    h=f.shape[0]; return np.median(cv2.cvtColor(f[int(h*.3):int(h*.85)],cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float),0)
def jersey(f,b,g):
    x1,y1,x2,y2=b; h=y2-y1; w=x2-x1; t=f[y1+int(h*.15):y1+int(h*.45), x1+int(w*.2):max(x1+int(w*.2)+1,x2-int(w*.2))]
    if t.size==0: return [0,128,128]
    l=cv2.cvtColor(t,cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float); k=l[np.linalg.norm(l-g,axis=1)>25]
    return np.median(k if len(k)>3 else l,0).tolist()
for key,(path,step,excl) in V.items():
    cap=cv2.VideoCapture(path); i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if i%step==0 and not (excl and excl[0]<=i<excl[1]):
            name=f'{key}_f{i:03d}'; cv2.imwrite(f'frames/{name}.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,92])
            r=m.predict(f,imgsz=960 if f.shape[1]>1500 else 640,conf=0.25,classes=[1,2],agnostic_nms=True,verbose=False)[0]
            b=m.predict(f,imgsz=1280 if f.shape[1]>1500 else 960,conf=0.2,classes=[0],verbose=False)[0]
            g=grass(f); ppl=[]
            for bb,c,s in zip(r.boxes.xyxy.numpy().astype(int).tolist(),r.boxes.cls.numpy().astype(int).tolist(),r.boxes.conf.numpy().tolist()):
                ppl.append(dict(b=bb,c=c,s=s,lab=jersey(f,bb,g)))
            balls=[dict(b=bb,s=s) for bb,s in zip(b.boxes.xyxy.numpy().astype(int).tolist(),b.boxes.conf.numpy().tolist())]
            out[name]=dict(people=ppl,balls=balls,W=f.shape[1],H=f.shape[0])
        i+=1
json.dump(out,open('pseudo.json','w'))
import collections; print(collections.Counter(k[:2] for k in out), len(out))
