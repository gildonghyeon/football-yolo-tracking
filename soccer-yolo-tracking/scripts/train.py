"""Train YOLOv8n (COCO-pretrained) on the self-collected 3-class soccer dataset.

    pip install -r requirements.txt
    python scripts/train.py                                   # CPU settings used for the submitted model
    python scripts/train.py --device 0 --epochs 80 --batch 16 # with a GPU (e.g. Colab)
"""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8n.pt")
p.add_argument("--data", default="data/own_dataset/data.yaml")
p.add_argument("--epochs", type=int, default=35)
p.add_argument("--imgsz", type=int, default=960)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--device", default="cpu")
a = p.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            close_mosaic=5, seed=0, patience=0, project="runs", name="own", exist_ok=True)
print(model.val(data=a.data, imgsz=a.imgsz, device=a.device).results_dict)
