"""Train MSD-Net (new modules + structure + SAD-Loss) on NEU-DET."""

import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO
from ultralytics.utils.loss_sad import SADTrainer

YAML = r"G:\deeplearning\YOLO-small\yaml\MSD-Net.yaml"  # use MSD-Net-FB.yaml for the full rebuild
DATA = r"G:\deeplearning\YOLO-small\NEU-DET\NEU-DET.yaml"
PRETRAINED = r"G:\deeplearning\YOLO-small\yolo11n.pt"  # transfers the backbone only; set None to train from scratch

if __name__ == "__main__":
    model = YOLO(YAML)
    if PRETRAINED:
        model.load(PRETRAINED)  # partial transfer: backbone weights, new neck/head stay random
    model.train(
        trainer=SADTrainer,
        data=DATA,
        epochs=300,
        batch=12,
        imgsz=640,
        workers=0,
        device=0,
        optimizer="auto",
        amp=True,
        cache=False,
        pretrained=False,
        name="MSD-Net",
        patience=50,
        cos_lr=True,
        seed=0,
    )
