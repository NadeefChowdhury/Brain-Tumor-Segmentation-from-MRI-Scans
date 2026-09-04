Python 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# =========================
# CONFIG
# =========================
REPO_ID = "your-username/brain-tumor-models"

YOLO_FILE = "best.pt"
SEG_MODELS = [
    "resunet.pth",
    "attention_unet.pth",
    "unetplusplus.pth"
]

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODELS
# =========================
def load_models():
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=YOLO_FILE)
    yolo = YOLO(yolo_path)

    seg_models = []
    for file in SEG_MODELS:
        path = hf_hub_download(repo_id=REPO_ID, filename=file)

        model = torch.load(path, map_location=DEVICE)
        model.to(DEVICE)
        model.eval()

        seg_models.append(model)

    return yolo, seg_models

yolo, seg_models = load_models()

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0)
    return img.to(DEVICE)

# =========================
# ENSEMBLE
# =========================
def ensemble_predict(models, inp):
    preds = []
    with torch.no_grad():
        for m in models:
            p = torch.sigmoid(m(inp))
            preds.append(p)

    return torch.mean(torch.stack(preds), dim=0)

# =========================
# PIPELINE
# =========================
def run_pipeline(image):
    orig = np.array(image)
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    h, w = orig.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    results = yolo(orig)[0]

    if results.boxes is None:
        return image

    for box in results.boxes.xyxy:
...         x1, y1, x2, y2 = map(int, box)
... 
...         crop = orig[y1:y2, x1:x2]
...         if crop.size == 0:
...             continue
... 
...         inp = preprocess(crop)
... 
...         pred = ensemble_predict(seg_models, inp)
...         pred = pred.squeeze().cpu().numpy()
... 
...         mask = (pred > 0.5).astype(np.uint8)
...         mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
... 
...         full_mask[y1:y2, x1:x2] = np.maximum(
...             full_mask[y1:y2, x1:x2],
...             mask_resized
...         )
... 
...     # Overlay
...     overlay = orig.copy()
...     overlay[full_mask == 1] = [0, 0, 255]
... 
...     final = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
...     final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
... 
...     return final
... 
... # =========================
... # UI
... # =========================
... app = gr.Interface(
...     fn=run_pipeline,
...     inputs=gr.Image(type="pil"),
...     outputs=gr.Image(type="numpy"),
...     title="🧠 Brain Tumor Segmentation (YOLO + Ensemble)",
...     description="Upload MRI → Detect + Segment Tumor"
... )
... 
